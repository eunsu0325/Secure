"""Post-run aggregator: read full_results.json, write summary.json.

The runner calls ``write_summary(output_dir)`` after STEP 9. This module reads
``full_results.json`` plus per-condition ``score_distributions.json`` files and
emits a flat ``summary.json`` with the keys the plan §1 calls out.

5% FPIR (and any value in ``additional_fpirs`` other than the protocol's
``target_fpir``) is computed only when dev-score exposure is present:

  * Thresholds are recomputed from ``external_dev`` impostor max-scores.
  * Known TPIR is evaluated on known-test scores/predictions.
  * Achieved external FPIR is evaluated on external-test scores.
  * Older runs without dev-score exposure are marked deferred.

Calibration discipline: thresholds are always computed from external_dev
impostor max-scores. external_test scores are never used for thresholding.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from experiments.exp1_common.calibration import threshold_from_dev_scores


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_summary(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


# ---------------------------------------------------------------------------
# Lookup helpers (resilient to small key-naming differences across protocols)
# ---------------------------------------------------------------------------


_TPIR_KEYS = ("tpir_at_target_fpir", "tpir_at_1pct_fpir", "tpir", "TPIR")
_FPIR_KEYS = ("achieved_external_fpir", "external_test_fpir", "fpir", "FPIR")
_RANK1_KEYS = ("rank1_accuracy", "rank1", "rank_1", "rank1_acc", "Rank1")
_THRESHOLD_KEYS = ("threshold", "tau", "tau_initial", "threshold_initial")
_KNOWN_SCORE_KEYS = ("known_scores", "known_max_scores", "known")
_EXTERNAL_SCORE_KEYS = ("external_scores", "external_max_scores", "external")
_DEV_IMPOSTOR_KEYS = (
    "external_dev_impostor_max_scores",
    "external_dev_impostor_max_per_step",
    "external_dev_impostor_max",
    "dev_impostor_max_scores",
    "external_dev_max_impostor",
)


def _first_present(d: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _step_sort_key(key: Any) -> Tuple[int, Any]:
    try:
        return (0, int(str(key).split("_")[-1]))
    except Exception:
        return (1, str(key))


def _ordered_step_items(steps: Any) -> List[Tuple[Optional[str], Dict[str, Any]]]:
    if isinstance(steps, dict):
        return [
            (str(k), steps[k] or {})
            for k in sorted(steps.keys(), key=_step_sort_key)
            if isinstance(steps[k] or {}, dict)
        ]
    if isinstance(steps, list):
        return [
            (None, step or {})
            for step in steps
            if isinstance(step or {}, dict)
        ]
    return []


def _choose_step_item(items: List[Tuple[Optional[str], Dict[str, Any]]],
                      prefer_largest_gallery: bool) -> Tuple[Optional[str], Dict[str, Any]]:
    if not items:
        return None, {}
    if prefer_largest_gallery:
        with_gallery = [
            (key, step) for key, step in items
            if isinstance(step.get("gallery_size"), (int, float, np.integer, np.floating))
        ]
        if with_gallery:
            return max(with_gallery, key=lambda item: float(item[1]["gallery_size"]))
    return items[-1]


def _get_final_step_entry(condition_block: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    for container_key in ("per_step", "steps", "step_results"):
        steps = condition_block.get(container_key)
        items = _ordered_step_items(steps)
        if items:
            return _choose_step_item(items, prefer_largest_gallery=True)
    return None, condition_block


def _get_final_step(condition_block: Dict[str, Any]) -> Dict[str, Any]:
    """Return the dict for the final-gallery step of a sequential condition.

    ``condition_block`` is the per-condition payload as embedded in
    ``full_results.json`` (without ``score_distributions``). Sequential
    conditions store per-step results under ``per_step`` (or ``steps``); the
    final step is the one with the largest gallery_size or step index.
    """
    return _get_final_step_entry(condition_block)[1]


def _scalar(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, np.generic):
        return float(value)
    return None


# ---------------------------------------------------------------------------
# Additional FPIR helpers
# ---------------------------------------------------------------------------


def _as_float_array(values: Any, name: str) -> np.ndarray:
    if values is None:
        raise ValueError(f"{name} missing")
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError(f"{name} empty")
    return arr


def _as_int_array(values: Any, name: str) -> np.ndarray:
    if values is None:
        raise ValueError(f"{name} missing")
    arr = np.asarray(values, dtype=np.int64)
    if arr.size == 0:
        raise ValueError(f"{name} empty")
    return arr


def _true_ids_from_probe_keys(probe_keys: Any) -> np.ndarray:
    if probe_keys is None:
        raise ValueError("known_probe_keys missing")
    true_ids = []
    for item in probe_keys:
        if not isinstance(item, (list, tuple)) or len(item) < 1:
            raise ValueError("known_probe_keys malformed")
        true_ids.append(int(item[0]))
    arr = np.asarray(true_ids, dtype=np.int64)
    if arr.size == 0:
        raise ValueError("known_probe_keys empty")
    return arr


def _compute_metrics_at_fpir(blob: Dict[str, Any],
                             fpir: float) -> Tuple[float, float, float]:
    dev_scores = _first_present(blob, _DEV_IMPOSTOR_KEYS)
    known_scores = _first_present(blob, _KNOWN_SCORE_KEYS)
    external_scores = _first_present(blob, _EXTERNAL_SCORE_KEYS)
    pred_ids = blob.get("known_pred_ids")
    probe_keys = blob.get("known_probe_keys")

    dev_arr = _as_float_array(dev_scores, "external_dev_impostor_max_scores")
    tau = threshold_from_dev_scores(dev_arr, fpir)
    known_arr = _as_float_array(known_scores, "known_scores")
    external_arr = _as_float_array(external_scores, "external_scores")
    pred_arr = _as_int_array(pred_ids, "known_pred_ids")
    true_arr = _true_ids_from_probe_keys(probe_keys)

    if not (len(known_arr) == len(pred_arr) == len(true_arr)):
        raise ValueError(
            "known_scores, known_pred_ids, and known_probe_keys length mismatch"
        )
    accepted_correct = (pred_arr == true_arr) & (known_arr >= tau)
    tpir = float(np.mean(accepted_correct))
    achieved_fpir = float(np.mean(external_arr >= tau))
    return float(tau), tpir, achieved_fpir


def _condition_dir_name(condition_key: str) -> str:
    return "condition_" + condition_key


def _select_score_blob(sd: Dict[str, Any],
                       final_step: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(sd, dict) or not sd:
        return None

    preferred_keys = [
        final_step.get("step"),
        final_step.get("gallery_step_key"),
    ]
    for key in preferred_keys:
        if key is not None and str(key) in sd and isinstance(sd[str(key)], dict):
            return sd[str(key)]

    items = _ordered_step_items(sd)
    return _choose_step_item(items, prefer_largest_gallery=True)[1] if items else None


def _load_condition_score_blob(output_dir: Path,
                               condition_key: str,
                               final_step: Dict[str, Any],
                               condition_block: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    if isinstance(condition_block, dict):
        embedded = condition_block.get("score_distributions")
        selected = _select_score_blob(embedded, final_step) if isinstance(embedded, dict) else None
        if selected is not None:
            return selected

    sd_path = output_dir / _condition_dir_name(condition_key) / "score_distributions.json"
    if not sd_path.exists():
        return None
    sd = _load_json(sd_path)
    return _select_score_blob(sd, final_step)


def _get_b_final(full: Dict[str, Any], final_gallery_size: int) -> Tuple[Optional[str], Dict[str, Any]]:
    b_block = full.get("B", {}) or {}
    if not isinstance(b_block, dict) or not b_block:
        return None, {}

    preferred_key = str(final_gallery_size)
    if final_gallery_size and preferred_key in b_block:
        return preferred_key, b_block.get(preferred_key, {}) or {}

    numeric_items = []
    for key, value in b_block.items():
        try:
            numeric_items.append((int(key), str(key), value or {}))
        except Exception:
            continue
    if numeric_items:
        _, key, value = max(numeric_items, key=lambda item: item[0])
        return key, value if isinstance(value, dict) else {}

    key = sorted(b_block.keys())[-1]
    value = b_block.get(key, {}) or {}
    return str(key), value if isinstance(value, dict) else {}


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def write_summary(
    output_dir,
    *,
    dataset_name: Optional[str] = None,
    backbone_name: Optional[str] = None,
) -> Path:
    """Read ``full_results.json`` under output_dir and write ``summary.json``.

    ``dataset_name`` and ``backbone_name`` are optional — if omitted, they are
    read from ``full_results.json``'s embedded config.
    """
    output_dir = Path(output_dir)
    full_path = output_dir / "full_results.json"
    if not full_path.exists():
        raise FileNotFoundError(f"full_results.json not found at {full_path}")
    full = _load_json(full_path)
    cfg = full.get("config", {}) or {}
    dataset_name = dataset_name or cfg.get("dataset", {}).get("name") or cfg.get("dataset", {}).get("txt_file", "")
    backbone_name = backbone_name or cfg.get("model", {}).get("architecture", "")
    target_fpir = float(cfg.get("scoring", {}).get("target_fpir", 0.01))
    additional_fpirs = [float(x) for x in cfg.get("scoring", {}).get("additional_fpirs", []) or []]
    static_gallery_sizes = list(cfg.get("static_gallery_sizes", []) or [])
    final_gallery_size = int(static_gallery_sizes[-1]) if static_gallery_sizes else 0

    summary: Dict[str, Any] = {
        "dataset_name": str(dataset_name),
        "backbone_name": str(backbone_name),
        "final_gallery_size": final_gallery_size,
        "target_fpir": target_fpir,
        "additional_fpirs": additional_fpirs,
    }

    # Identity-count fields (best-effort: read from sanity_check / B-final block)
    sanity = full.get("sanity_check", {}) or {}
    summary["sanity_pass"] = bool(sanity.get("all_pass", sanity.get("sanity_pass", False)))

    b_final_key, b_final = _get_b_final(full, final_gallery_size)
    if b_final:
        summary["external_identity_count"] = int(
            b_final.get("external_identity_count", b_final.get("n_external", 0)) or 0
        )
        summary["external_dev_probe_count"] = int(
            b_final.get("external_dev_probe_count", b_final.get("n_external_dev_probes", 0)) or 0
        )
        summary["external_test_probe_count"] = int(
            b_final.get("external_test_probe_count", b_final.get("n_external_test_probes", 0)) or 0
        )

    # ---- A: closed-set rank-1 ----
    a_block = full.get("A", {}) or {}
    a_final = _get_final_step(a_block) if isinstance(a_block, dict) else {}
    summary["A_final_rank1"] = _scalar(_first_present(a_final, _RANK1_KEYS))

    # ---- B-final: TPIR @ target FPIR ----
    b_tpir = _scalar(_first_present(b_final, _TPIR_KEYS))
    summary["B_final_tpir_at_target_fpir"] = b_tpir
    summary["B_final_tpir_at_1pct_fpir"] = b_tpir
    summary["B_final_achieved_external_fpir"] = _scalar(_first_present(b_final, _FPIR_KEYS))

    # ---- C variants: final-step metrics ----
    for key in ("C_raw_fixed", "C_raw_recalib", "C_snorm_fixed", "C_snorm_recalib"):
        block = full.get(key, {}) or {}
        final_step = _get_final_step(block)
        tpir = _scalar(_first_present(final_step, _TPIR_KEYS))
        summary[f"{key}_final_tpir_at_target_fpir"] = tpir
        summary[f"{key}_final_tpir_at_1pct_fpir"] = tpir
        out_key_fpir = f"{key}_final_achieved_external_fpir"
        summary[out_key_fpir] = _scalar(_first_present(final_step, _FPIR_KEYS))

    # Threshold trajectory (raw-fixed initial vs raw-recalib final)
    rf_block = full.get("C_raw_fixed", {}) or {}
    rr_block = full.get("C_raw_recalib", {}) or {}
    rf_first = _get_first_step(rf_block)
    rr_final = _get_final_step(rr_block)
    summary["threshold_fixed_initial"] = _scalar(_first_present(rf_first, _THRESHOLD_KEYS))
    summary["threshold_recalib_final"] = _scalar(_first_present(rr_final, _THRESHOLD_KEYS))

    # ---- additional FPIRs gate ----
    if additional_fpirs:
        gate = _try_compute_additional_fpirs(
            output_dir=output_dir,
            additional_fpirs=additional_fpirs,
            full=full,
        )
        summary.update(gate)
    else:
        summary["additional_fpirs_status"] = "not_requested"

    out_path = output_dir / "summary.json"
    _dump_summary(out_path, summary)
    return out_path


def _get_first_step(condition_block: Dict[str, Any]) -> Dict[str, Any]:
    for container_key in ("per_step", "steps", "step_results"):
        steps = condition_block.get(container_key)
        items = _ordered_step_items(steps)
        if items:
            return items[0][1]
    return condition_block


def _try_compute_additional_fpirs(
    *,
    output_dir: Path,
    additional_fpirs: Sequence[float],
    full: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute additional-FPIR metrics from dev-calibrated thresholds."""
    cfg = full.get("config", {}) or {}
    static_gallery_sizes = list(cfg.get("static_gallery_sizes", []) or [])
    final_gallery_size = int(static_gallery_sizes[-1]) if static_gallery_sizes else 0

    expected: List[Tuple[str, Dict[str, Any]]] = []
    _, b_final = _get_b_final(full, final_gallery_size)
    if b_final:
        expected.append(("B_final", b_final))

    for key in ("C_raw_fixed", "C_raw_recalib", "C_snorm_fixed", "C_snorm_recalib"):
        block = full.get(key)
        if not isinstance(block, dict) or not block:
            continue
        _, final_step = _get_final_step_entry(block)
        score_blob = _load_condition_score_blob(
            output_dir,
            key,
            final_step,
            condition_block=block,
        )
        expected.append((f"{key}_final", score_blob or {}))

    if not expected or not any(_first_present(blob, _DEV_IMPOSTOR_KEYS) is not None for _, blob in expected):
        return {
            "additional_fpirs_status": "deferred",
            "additional_fpirs_deferred_reason": (
                "No B/C result exposes external_dev impostor max-scores. "
                "Computing additional FPIR thresholds from external_test would "
                "constitute calibration leakage and is forbidden."
            ),
        }

    extras: Dict[str, Any] = {}
    partial_reasons: Dict[str, str] = {}
    computed_conditions = 0

    for prefix, blob in expected:
        condition_ok = True
        for fpir in additional_fpirs:
            try:
                tau, tpir, achieved_fpir = _compute_metrics_at_fpir(blob, fpir)
            except Exception as exc:
                condition_ok = False
                partial_reasons[prefix] = str(exc)
                break
            extras[f"{prefix}_tau_at_{fpir:g}_fpir"] = tau
            extras[f"{prefix}_tpir_at_{fpir:g}_fpir"] = tpir
            extras[f"{prefix}_achieved_external_fpir_at_{fpir:g}_fpir"] = achieved_fpir
        if condition_ok:
            computed_conditions += 1

    extras["additional_fpirs_status"] = (
        "computed" if computed_conditions == len(expected) else "partial"
    )
    if partial_reasons:
        extras["additional_fpirs_partial_reasons"] = partial_reasons
    return extras
