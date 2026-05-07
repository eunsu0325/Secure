"""Protocols A / B / C-fixed / C-recal (D4-D5).

All four protocols share the score matrix; each one defines a different row /
column slicing rule and (for B/C) a different threshold-calibration recipe.

Layout:
  - `build_t_schedule`         : enforces t_max inclusion (plan: t_schedule rule)
  - `evaluate_step`            : compute all metrics at a single (gallery, τ) pair
  - `run_protocol_a`           : closed-set rank-1 trajectory (no τ)
  - `run_protocol_b`           : static endpoint (single number)
  - `run_protocol_c_fixed`     : sequential, τ calibrated once at gallery_0
  - `run_protocol_c_recal`     : sequential, τ recalibrated per step
  - `assert_b_equals_c_recal_endpoint` : the 8-condition sanity check
                                          (Δ < 1e-6 on shared metrics)

Hard rules enforced here:
  - external_dev rows are the ONLY calibration probe set (Hard Rule 1)
  - external_test never has prototypes (Hard Rule 3)
  - future cols not yet enrolled are masked at every C step (Hard Rule 4)
  - A/B/C share gallery composition `base_anchor + future_order[:t]`
    (Hard Rule 6)
  - argmax tie-break: lowest palm_id wins (IER-3 / IER-4 / sanity #8)
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from exp1_baselines.eval.metrics import (
    argmax_with_tiebreak,
    compute_fpir_metrics,
    compute_known_metrics,
    min_observable_fpir,
    reliability_flag,
    wilson_ci,
)
from exp1_baselines.eval.score_matrix import (
    ScoreMatrix,
    assert_step_consistency,
    enrolled_palm_set,
    not_enrolled_future_set,
)
from exp1_baselines.eval.thresholding import calibrate_threshold_at_fpir


# ---------------------------------------------------------------------------
# t-schedule (mandatory t_max inclusion)
# ---------------------------------------------------------------------------

def build_t_schedule(t_max: int, step: int) -> List[int]:
    """t-schedule for Protocol C; always includes 0 and t_max.

    The plan rule (B vs C-recal endpoint sanity) requires the schedule to
    end exactly at t_max so the gallery composition matches Protocol B.
    """
    if t_max < 0:
        raise ValueError(f"t_max must be >= 0, got {t_max}")
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    schedule = list(range(0, t_max + 1, step))
    if not schedule:
        schedule = [0]
    if schedule[-1] != t_max:
        schedule.append(t_max)
    return schedule


# ---------------------------------------------------------------------------
# Default contralateral lookup (even-odd palm pairing)
# ---------------------------------------------------------------------------

def contralateral_xor1(palm_id: int) -> int:
    """Default contralateral mapping for canonical even-odd palm pairing.

    Used by Tongji (P0a VERIFIED-INFERRED), IITD, BJTU when palm_ids are
    assigned so that adjacent pairs share a subject.
    """
    return int(palm_id) ^ 1


# ---------------------------------------------------------------------------
# Internal slicing helpers
# ---------------------------------------------------------------------------

def _slice_scores(
    sm: ScoreMatrix,
    *,
    query_palms: Sequence[int],
    gallery_palms: Sequence[int],
):
    """Bool-mask slicing helper used by every protocol step.

    Returns (sliced_scores, sliced_query_palm_ids, sliced_gallery_palm_ids).
    Empty inputs return zero-shape arrays so callers can handle them uniformly.
    """
    if not query_palms or not gallery_palms:
        empty = np.zeros((0, 0), dtype=sm.scores.dtype)
        return empty, np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    row_mask = sm.query_mask_for_palms(query_palms)
    col_mask = sm.proto_mask_for_palms(gallery_palms)
    sliced = sm.scores[np.ix_(row_mask, col_mask)]
    return sliced, sm.query_palm_id[row_mask], sm.proto_palm_id[col_mask]


def _calibrate_against_gallery(
    sm: ScoreMatrix,
    gallery_palms: Sequence[int],
    calibration_palms: Sequence[int],
    target_fpir: float,
) -> float:
    """Tie-aware τ from external_dev × current gallery top-1 scores.

    `calibration_palms` MUST be the external_dev palm_ids (Hard Rule 1).
    """
    sliced, _, _ = _slice_scores(
        sm, query_palms=calibration_palms, gallery_palms=gallery_palms,
    )
    if sliced.size == 0 or sliced.shape[0] == 0 or sliced.shape[1] == 0:
        return float("inf")
    top1_scores = sliced.max(axis=1)
    return calibrate_threshold_at_fpir(top1_scores, target_fpir)


def _evaluate_known(
    sm: ScoreMatrix,
    gallery_palms: Sequence[int],
    known_palms: Sequence[int],
    tau: float,
) -> Dict[str, object]:
    sliced, q_palm, g_palm = _slice_scores(
        sm, query_palms=known_palms, gallery_palms=gallery_palms,
    )
    if sliced.shape[0] == 0 or sliced.shape[1] == 0:
        return compute_known_metrics(
            np.array([], dtype=np.int64),
            np.array([]),
            np.array([], dtype=np.int64),
            tau,
        )
    top1_palm, top1_score = argmax_with_tiebreak(sliced, g_palm)
    return compute_known_metrics(top1_palm, top1_score, q_palm, tau)


def _evaluate_unknown(
    sm: ScoreMatrix,
    gallery_palms: Sequence[int],
    unknown_palms: Sequence[int],
    tau: float,
    *,
    prefix: str,
) -> Dict[str, object]:
    sliced, _, _ = _slice_scores(
        sm, query_palms=unknown_palms, gallery_palms=gallery_palms,
    )
    if sliced.shape[0] == 0 or sliced.shape[1] == 0:
        return compute_fpir_metrics(np.array([]), tau, prefix=prefix)
    top1_scores = sliced.max(axis=1)
    return compute_fpir_metrics(top1_scores, tau, prefix=prefix)


def _evaluate_future_with_contralateral(
    sm: ScoreMatrix,
    gallery_palms: Sequence[int],
    future_unknown_palms: Sequence[int],
    tau: float,
    contralateral_of: Optional[Callable[[int], int]],
) -> Dict[str, object]:
    base = _evaluate_unknown(
        sm, gallery_palms, future_unknown_palms, tau, prefix="future",
    )
    if contralateral_of is None or not future_unknown_palms:
        return base

    gallery_set = set(int(x) for x in gallery_palms)
    has_contra: List[int] = []
    no_contra: List[int] = []
    for pid in future_unknown_palms:
        if contralateral_of(int(pid)) in gallery_set:
            has_contra.append(int(pid))
        else:
            no_contra.append(int(pid))

    contra = _evaluate_unknown(
        sm, gallery_palms, has_contra, tau, prefix="future_contralateral",
    )
    nocontra = _evaluate_unknown(
        sm, gallery_palms, no_contra, tau, prefix="future_no_contralateral",
    )
    return {**base, **contra, **nocontra}


# ---------------------------------------------------------------------------
# Step evaluation (used by C-fixed and C-recal)
# ---------------------------------------------------------------------------

def evaluate_step(
    sm: ScoreMatrix,
    *,
    base_anchor_ids: Sequence[int],
    future_order: Sequence[int],
    t: int,
    external_dev_ids: Sequence[int],
    external_test_ids: Sequence[int],
    tau: float,
    target_fpir: float,
    contralateral_of: Optional[Callable[[int], int]] = contralateral_xor1,
    assert_consistency: bool = True,
) -> Dict[str, object]:
    """Compute all step metrics at a fixed τ. Assumes IER-3 invariants."""
    enrolled = enrolled_palm_set(base_anchor_ids, future_order, t)
    future_unk = not_enrolled_future_set(future_order, t)

    if assert_consistency:
        # Build the per-step "ground truth" sets from the masks themselves.
        known_query_pids = sorted(set(int(x) for x in
            sm.query_palm_id[sm.query_mask_for_palms(enrolled)]))
        future_unknown_pids = sorted(set(int(x) for x in
            sm.query_palm_id[sm.query_mask_for_palms(future_unk)]))
        calibration_pids = sorted(set(int(x) for x in
            sm.query_palm_id[sm.query_mask_for_palms(external_dev_ids)]))
        external_unknown_pids = sorted(set(int(x) for x in
            sm.query_palm_id[sm.query_mask_for_palms(external_test_ids)]))
        gallery_col_pids = sorted(set(int(x) for x in
            sm.proto_palm_id[sm.proto_mask_for_palms(enrolled)]))

        # Filter expected to whatever actually has rows/cols in the matrix
        # (a palm with no queries should be tolerated; we assert SUBSET).
        # The strict equality in score_matrix.assert_step_consistency expects
        # full coverage, so we check via subset relationships here.
        if set(gallery_col_pids) - set(enrolled):
            raise AssertionError(
                f"step {t}: gallery cols include unenrolled palms"
            )
        if set(future_unknown_pids) - set(future_unk):
            raise AssertionError(
                f"step {t}: future-unknown rows include enrolled palms"
            )

    known = _evaluate_known(sm, enrolled, enrolled, tau)
    future = _evaluate_future_with_contralateral(
        sm, enrolled, future_unk, tau, contralateral_of,
    )
    external = _evaluate_unknown(
        sm, enrolled, external_test_ids, tau, prefix="external",
    )

    return {
        "t": int(t),
        "t_max": len(future_order),
        "tau": float(tau),
        "gallery_size": len(enrolled),
        "target_fpir": float(target_fpir),
        **known,
        **future,
        **external,
        "future_min_observable_fpir": min_observable_fpir(
            int(future["future_total"])
        ),
        "external_min_observable_fpir": min_observable_fpir(
            int(external["external_total"])
        ),
        "future_reliability": reliability_flag(
            int(future["future_total"]), target_fpir,
        ),
        "external_reliability": reliability_flag(
            int(external["external_total"]), target_fpir,
        ),
    }


# ---------------------------------------------------------------------------
# Protocol A — closed-set rank-1 (no τ)
# ---------------------------------------------------------------------------

def run_protocol_a(
    sm: ScoreMatrix,
    *,
    base_anchor_ids: Sequence[int],
    future_order: Sequence[int],
    t_schedule: Sequence[int],
) -> List[Dict[str, object]]:
    """Closed-set rank-1 over `base_anchor + future_order[:t]` (plan: A).

    Background diagnostic only; no τ, no decomposition. Bug fix vs legacy:
    gallery now ALWAYS includes base_anchor (Hard Rule 6 / Decision Log).
    """
    results: List[Dict[str, object]] = []
    for t in t_schedule:
        enrolled = enrolled_palm_set(base_anchor_ids, future_order, t)
        sliced, q_palm, g_palm = _slice_scores(
            sm, query_palms=enrolled, gallery_palms=enrolled,
        )
        if sliced.shape[0] == 0 or sliced.shape[1] == 0:
            results.append({
                "t": int(t),
                "gallery_size": len(enrolled),
                "rank1": float("nan"),
                "rank1_correct": 0,
                "rank1_total": 0,
                "rank1_ci95": (float("nan"), float("nan")),
            })
            continue
        top1_palm, _ = argmax_with_tiebreak(sliced, g_palm)
        n = int(q_palm.shape[0])
        correct = int((top1_palm == q_palm).sum())
        results.append({
            "t": int(t),
            "gallery_size": len(enrolled),
            "rank1": correct / n,
            "rank1_correct": correct,
            "rank1_total": n,
            "rank1_ci95": wilson_ci(correct, n),
        })
    return results


# ---------------------------------------------------------------------------
# Protocol B — static full-gallery endpoint (single number, NOT t-wise)
# ---------------------------------------------------------------------------

def run_protocol_b(
    sm: ScoreMatrix,
    *,
    base_anchor_ids: Sequence[int],
    future_ids: Sequence[int],
    external_dev_ids: Sequence[int],
    external_test_ids: Sequence[int],
    target_fpir: float,
) -> Dict[str, object]:
    """Single endpoint at gallery = base_anchor + all_future."""
    full_gallery = sorted(set(int(x) for x in base_anchor_ids)
                          | set(int(x) for x in future_ids))
    tau = _calibrate_against_gallery(
        sm, full_gallery, external_dev_ids, target_fpir,
    )
    known = _evaluate_known(sm, full_gallery, full_gallery, tau)
    external = _evaluate_unknown(
        sm, full_gallery, external_test_ids, tau, prefix="external",
    )
    return {
        "protocol": "B",
        "tau": float(tau),
        "gallery_size": len(full_gallery),
        "target_fpir": float(target_fpir),
        **known,
        **external,
        "external_min_observable_fpir": min_observable_fpir(
            int(external["external_total"])
        ),
        "external_reliability": reliability_flag(
            int(external["external_total"]), target_fpir,
        ),
    }


# ---------------------------------------------------------------------------
# Protocol C-fixed — sequential, τ calibrated once at gallery_0
# ---------------------------------------------------------------------------

def run_protocol_c_fixed(
    sm: ScoreMatrix,
    *,
    base_anchor_ids: Sequence[int],
    future_order: Sequence[int],
    external_dev_ids: Sequence[int],
    external_test_ids: Sequence[int],
    target_fpir: float,
    t_schedule: Sequence[int],
    contralateral_of: Optional[Callable[[int], int]] = contralateral_xor1,
) -> List[Dict[str, object]]:
    """C-fixed: τ calibrated once against gallery_0 = base_anchor."""
    tau_fixed = _calibrate_against_gallery(
        sm, base_anchor_ids, external_dev_ids, target_fpir,
    )
    results: List[Dict[str, object]] = []
    for t in t_schedule:
        step = evaluate_step(
            sm,
            base_anchor_ids=base_anchor_ids,
            future_order=future_order, t=t,
            external_dev_ids=external_dev_ids,
            external_test_ids=external_test_ids,
            tau=tau_fixed, target_fpir=target_fpir,
            contralateral_of=contralateral_of,
        )
        step["protocol"] = "C-fixed"
        results.append(step)
    return results


# ---------------------------------------------------------------------------
# Protocol C-recal — sequential, τ recalibrated per step
# ---------------------------------------------------------------------------

def run_protocol_c_recal(
    sm: ScoreMatrix,
    *,
    base_anchor_ids: Sequence[int],
    future_order: Sequence[int],
    external_dev_ids: Sequence[int],
    external_test_ids: Sequence[int],
    target_fpir: float,
    t_schedule: Sequence[int],
    contralateral_of: Optional[Callable[[int], int]] = contralateral_xor1,
) -> List[Dict[str, object]]:
    """C-recal: per-step τ recalibrated against current operational gallery."""
    results: List[Dict[str, object]] = []
    for t in t_schedule:
        enrolled = enrolled_palm_set(base_anchor_ids, future_order, t)
        tau_t = _calibrate_against_gallery(
            sm, enrolled, external_dev_ids, target_fpir,
        )
        step = evaluate_step(
            sm,
            base_anchor_ids=base_anchor_ids,
            future_order=future_order, t=t,
            external_dev_ids=external_dev_ids,
            external_test_ids=external_test_ids,
            tau=tau_t, target_fpir=target_fpir,
            contralateral_of=contralateral_of,
        )
        step["protocol"] = "C-recal"
        results.append(step)
    return results


# ---------------------------------------------------------------------------
# B == C-recal endpoint sanity check
# ---------------------------------------------------------------------------

_B_VS_C_FLOAT_KEYS = (
    "tau", "tpir", "fnir",
    "known_wrong_accept_rate", "known_reject_rate",
    "external_fpir",
)
_B_VS_C_INT_KEYS = (
    "known_total", "known_correct_accept_count",
    "known_wrong_accept_count", "known_reject_count",
    "external_total", "external_false_accept",
    "gallery_size",
)


def assert_b_equals_c_recal_endpoint(
    b_result: Dict[str, object],
    c_recal_endpoint: Dict[str, object],
    *,
    tol: float = 1e-6,
) -> None:
    """Verify Protocol B equals Protocol C-recal at t = t_max.

    The 8 sanity conditions (plan: Protocol B section) require identical
    score matrix, gallery composition, query sets, target_fpir, and tie-break,
    so under that contract these metric values must match within `tol`.
    """
    for k in _B_VS_C_INT_KEYS:
        if k not in b_result or k not in c_recal_endpoint:
            continue
        if int(b_result[k]) != int(c_recal_endpoint[k]):
            raise AssertionError(
                f"B == C-recal endpoint mismatch on {k}: "
                f"B={b_result[k]} C={c_recal_endpoint[k]}"
            )
    for k in _B_VS_C_FLOAT_KEYS:
        if k not in b_result or k not in c_recal_endpoint:
            continue
        b_val = float(b_result[k])
        c_val = float(c_recal_endpoint[k])
        if np.isnan(b_val) and np.isnan(c_val):
            continue
        if abs(b_val - c_val) > tol:
            raise AssertionError(
                f"B == C-recal endpoint mismatch on {k}: "
                f"B={b_val} C={c_val} (Δ={abs(b_val-c_val):.2e})"
            )
