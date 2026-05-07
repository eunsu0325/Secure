"""Protocol A/B/C evaluation + fixed-threshold calibration for exp1_baselines.

This script implements:
  - Phase 5: fixed tau calibration from threshold_val_ids only.
  - Phase 6: Protocol A (closed-set), B (static open-set), C (sequential open-set).

Two hard rules (must never break):
  Rule 1: fixed tau NEVER uses external_test_ids or future_ids.
  Rule 2: known true accept is ALWAYS (top1_id == gt_id) AND (top1_score >= tau).

Usage::

    python exp1_baselines/eval_protocols.py \\
        --embeddings exp1_baselines/embeddings/mfn_tongji_112_embeddings.npz \\
        --manifest_dir exp1_baselines/manifests \\
        --model_tag mfn \\
        --output_dir exp1_baselines/results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Embedding loader
# ============================================================

class EmbeddingStore:
    """In-memory store of embeddings keyed by (label, session, sample_id)."""

    def __init__(self, npz_path: Path) -> None:
        data = np.load(npz_path, allow_pickle=True)
        self.embeddings = data["embeddings"].astype(np.float32)
        self.palm_id = data["palm_id"].astype(np.int64)
        self.session = np.asarray(data["session"]).astype(str)
        self.sample_id = data["sample_id"].astype(np.int64)
        self.path = np.asarray(data["path"]).astype(str)
        self.split_name = np.asarray(data["split_name"]).astype(str)

    def select(
        self, *, palm_ids: List[int], session: Optional[str] = None
    ) -> np.ndarray:
        """Return [N, D] embeddings for given palm_ids and optional session filter,
        sorted by (palm_id, sample_id) to be deterministic."""
        palm_set = set(int(x) for x in palm_ids)
        mask = np.array([int(p) in palm_set for p in self.palm_id], dtype=bool)
        if session is not None:
            mask = mask & (self.session == session)
        idx = np.where(mask)[0]
        order = sorted(idx, key=lambda i: (int(self.palm_id[i]), int(self.sample_id[i])))
        order_arr = np.array(order, dtype=np.int64)
        return self.embeddings[order_arr]

    def select_with_labels(
        self, *, palm_ids: List[int], session: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ([N,D], [N]) embeddings + their palm_id labels."""
        palm_set = set(int(x) for x in palm_ids)
        mask = np.array([int(p) in palm_set for p in self.palm_id], dtype=bool)
        if session is not None:
            mask = mask & (self.session == session)
        idx = np.where(mask)[0]
        order = sorted(idx, key=lambda i: (int(self.palm_id[i]), int(self.sample_id[i])))
        order_arr = np.array(order, dtype=np.int64)
        return self.embeddings[order_arr], self.palm_id[order_arr]


# ============================================================
# Prototype construction
# ============================================================

def build_prototypes(
    store: EmbeddingStore, palm_ids: List[int], session: str = "session1"
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean-pool session1 embeddings per identity, L2-normalize.

    Returns:
        protos: [P, D] L2-normalized prototypes
        proto_labels: [P] palm_id for each prototype
    """
    palm_ids = sorted(int(x) for x in palm_ids)
    feats, labels = store.select_with_labels(palm_ids=palm_ids, session=session)
    proto_list: List[np.ndarray] = []
    proto_labels: List[int] = []
    for pid in palm_ids:
        sel = labels == pid
        if not sel.any():
            continue
        mean = feats[sel].mean(axis=0)
        norm = np.linalg.norm(mean) + 1e-12
        proto_list.append(mean / norm)
        proto_labels.append(int(pid))
    protos = np.stack(proto_list, axis=0).astype(np.float32)
    return protos, np.array(proto_labels, dtype=np.int64)


# ============================================================
# Scoring
# ============================================================

def cosine_scores(queries: np.ndarray, protos: np.ndarray) -> np.ndarray:
    """[N, P] cosine score matrix. Both inputs assumed L2-normalized."""
    return queries @ protos.T


def top1(scores: np.ndarray, proto_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (top1_score [N], top1_id [N])."""
    if scores.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    idx = scores.argmax(axis=1)
    top1_score = scores[np.arange(scores.shape[0]), idx].astype(np.float32)
    top1_id = proto_labels[idx].astype(np.int64)
    return top1_score, top1_id


# ============================================================
# Threshold calibration (Phase 5)
# ============================================================

def calibrate_tau_at_far(
    unknown_scores: np.ndarray, target_far: float = 0.01
) -> Tuple[float, float]:
    """Smallest tau satisfying empirical FAR <= target_far on unknown scores.

    FAR(tau) = mean(unknown_scores >= tau). Returns (tau, realized_far).

    Strategy: pick tau = scores_sorted[k] where
        k = ceil((1 - target_far) * n)
    so that #scores >= tau is at most n - k <= floor(target_far * n).

    Then verify realized FAR; if it exceeds target_far due to ties at tau,
    nudge tau up to nextafter(tau, +inf).
    """
    scores = np.sort(np.asarray(unknown_scores, dtype=np.float64))
    n = int(scores.size)
    if n == 0:
        return float("inf"), 0.0
    k = int(np.ceil((1.0 - target_far) * n))
    k = min(max(k, 0), n - 1)
    tau = float(scores[k])
    realized = float((scores >= tau).mean())
    if realized > target_far + 1e-12:
        tau_up = float(np.nextafter(tau, np.inf))
        realized_up = float((scores >= tau_up).mean())
        if realized_up <= target_far + 1e-12:
            tau = tau_up
            realized = realized_up
    return tau, realized


# ============================================================
# Curve-based metrics
# ============================================================

def auroc_eer(known_scores: np.ndarray, unknown_scores: np.ndarray) -> Tuple[float, float]:
    """Compute AUROC and EER from known (positive) vs unknown (negative) scores."""
    if known_scores.size == 0 or unknown_scores.size == 0:
        return float("nan"), float("nan")
    y_true = np.concatenate([
        np.ones_like(known_scores, dtype=np.int32),
        np.zeros_like(unknown_scores, dtype=np.int32),
    ])
    scores = np.concatenate([known_scores, unknown_scores]).astype(np.float64)

    order = np.argsort(-scores, kind="stable")
    y_sorted = y_true[order]
    scores_sorted = scores[order]

    n_pos = float(known_scores.size)
    n_neg = float(unknown_scores.size)

    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = tps / max(n_pos, 1.0)
    fpr = fps / max(n_neg, 1.0)
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    auroc = float(np.trapz(tpr, fpr))

    fnr = 1.0 - tpr
    diff = fpr - fnr
    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_change) > 0:
        i = int(sign_change[0])
        eer = float((fpr[i] + fnr[i + 1]) / 2.0)
    else:
        idx = int(np.argmin(np.abs(diff)))
        eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return auroc, eer


def threshold_for_far(unknown_scores: np.ndarray, target_far: float) -> float:
    """Smallest threshold satisfying empirical FAR <= target_far."""
    if unknown_scores.size == 0:
        return float("inf")
    tau, _ = calibrate_tau_at_far(unknown_scores, target_far=target_far)
    return tau


def tar_at_far(known_scores: np.ndarray, unknown_scores: np.ndarray,
               target_far: float = 0.01) -> Tuple[float, float]:
    """Verification-style TAR@FAR. Returns (tar, threshold)."""
    if known_scores.size == 0 or unknown_scores.size == 0:
        return float("nan"), float("nan")
    thr = threshold_for_far(unknown_scores, target_far)
    tar = float((known_scores >= thr).mean())
    return tar, thr


def dir_at_fpir(
    known_top1_scores: np.ndarray,
    known_top1_correct: np.ndarray,
    unknown_top1_scores: np.ndarray,
    target_fpir: float = 0.01,
) -> Tuple[float, float]:
    """Open-set identification DIR@FPIR.

    A known query is a true accept iff (top1 correct AND top1_score >= thr).
    FPIR is fraction of unknown queries with max score >= thr.
    Returns (dir, threshold).
    """
    if known_top1_scores.size == 0 or unknown_top1_scores.size == 0:
        return float("nan"), float("nan")
    thr = threshold_for_far(unknown_top1_scores, target_fpir)
    accepted = (known_top1_scores >= thr) & known_top1_correct.astype(bool)
    dir_value = float(accepted.mean())
    return dir_value, thr


# ============================================================
# Per-known decomposition with fixed tau
# ============================================================

def known_decomposition(
    known_top1_scores: np.ndarray,
    known_top1_correct: np.ndarray,
    tau: float,
) -> Dict[str, float]:
    n = int(known_top1_scores.size)
    if n == 0:
        return {
            "n": 0,
            "rank1": float("nan"),
            "identification_acceptance_at_tau": float("nan"),
            "rejection_rate_at_tau": float("nan"),
            "misidentification_acceptance_at_tau": float("nan"),
            "total_error_at_tau": float("nan"),
            "mean_top1_score": float("nan"),
        }
    correct = known_top1_correct.astype(bool)
    above = known_top1_scores >= tau
    correct_accept = (correct & above).mean()
    rejected = (~above).mean()
    wrong_accept = (above & ~correct).mean()
    return {
        "n": n,
        "rank1": float(correct.mean()),
        "identification_acceptance_at_tau": float(correct_accept),
        "rejection_rate_at_tau": float(rejected),
        "misidentification_acceptance_at_tau": float(wrong_accept),
        "total_error_at_tau": float(1.0 - correct_accept),
        "mean_top1_score": float(known_top1_scores.mean()),
    }


def unknown_decomposition(
    unknown_top1_scores: np.ndarray, tau: float
) -> Dict[str, float]:
    n = int(unknown_top1_scores.size)
    if n == 0:
        return {
            "n": 0,
            "false_accept_rate_at_tau": None,
            "rejection_rate_at_tau": None,
            "mean_top1_score": None,
        }
    above = unknown_top1_scores >= tau
    return {
        "n": n,
        "false_accept_rate_at_tau": float(above.mean()),
        "rejection_rate_at_tau": float((~above).mean()),
        "mean_top1_score": float(unknown_top1_scores.mean()),
    }


# ============================================================
# Genuine score for verification-style TAR@FAR
# ============================================================

def genuine_scores_correct_class(
    queries: np.ndarray, query_labels: np.ndarray,
    protos: np.ndarray, proto_labels: np.ndarray,
) -> np.ndarray:
    """Cosine score against the prototype of the query's ground-truth class.

    Skips queries whose label is not in proto_labels (i.e., unenrolled).
    """
    label_to_idx = {int(l): i for i, l in enumerate(proto_labels.tolist())}
    out: List[float] = []
    for i, lbl in enumerate(query_labels.tolist()):
        if int(lbl) not in label_to_idx:
            continue
        pidx = label_to_idx[int(lbl)]
        score = float(queries[i] @ protos[pidx])
        out.append(score)
    return np.asarray(out, dtype=np.float32)


# ============================================================
# Phase 5: tau calibration
# ============================================================

def phase5_calibrate_tau(
    store: EmbeddingStore,
    threshold_gallery_ids: List[int],
    threshold_unknown_ids: List[int],
    target_far: float = 0.01,
) -> Dict[str, Any]:
    """Calibrate fixed tau from threshold_val_ids only."""
    protos, proto_labels = build_prototypes(store, threshold_gallery_ids, "session1")
    unknown_q, _ = store.select_with_labels(
        palm_ids=threshold_unknown_ids, session="session2"
    )
    unknown_scores = cosine_scores(unknown_q, protos)
    unknown_top1 = unknown_scores.max(axis=1) if unknown_scores.size > 0 else np.array([])
    tau, realized_far = calibrate_tau_at_far(unknown_top1, target_far=target_far)
    return {
        "tau": float(tau),
        "target_far": float(target_far),
        "realized_far": float(realized_far),
        "n_threshold_gallery": int(len(threshold_gallery_ids)),
        "n_threshold_unknown_probes": int(unknown_top1.size),
    }


# ============================================================
# Protocol A: closed-set known-only
# ============================================================

def run_protocol_a(
    store: EmbeddingStore,
    enrollment_order: List[int],
    t_schedule: List[int],
) -> Dict:
    steps: List[Dict] = []
    for t in t_schedule:
        gallery_ids = enrollment_order[:t]
        protos, proto_labels = build_prototypes(store, gallery_ids, "session1")
        queries, q_labels = store.select_with_labels(
            palm_ids=gallery_ids, session="session2"
        )
        scores = cosine_scores(queries, protos)
        top1_score, top1_id = top1(scores, proto_labels)
        correct = (top1_id == q_labels)
        steps.append({
            "t": int(t),
            "gallery_size": int(len(gallery_ids)),
            "n_probes": int(top1_score.size),
            "closed_set_rank1": float(correct.mean()) if top1_score.size else float("nan"),
            "mean_top1_score": float(top1_score.mean()) if top1_score.size else float("nan"),
        })
    return {
        "protocol": "A_closed_set_known_only",
        "schedule": list(t_schedule),
        "steps": steps,
    }


# ============================================================
# Protocol B: static open-set snapshots
# ============================================================

def run_protocol_b(
    store: EmbeddingStore,
    enrollment_order: List[int],
    external_test_ids: List[int],
    t_schedule: List[int],
    tau: float,
) -> Dict:
    steps: List[Dict] = []
    ext_q, ext_q_labels = store.select_with_labels(
        palm_ids=external_test_ids, session="session2"
    )
    for t in t_schedule:
        gallery_ids = enrollment_order[:t]
        protos, proto_labels = build_prototypes(store, gallery_ids, "session1")
        # known
        known_q, known_q_labels = store.select_with_labels(
            palm_ids=gallery_ids, session="session2"
        )
        known_scores = cosine_scores(known_q, protos)
        known_top1_score, known_top1_id = top1(known_scores, proto_labels)
        known_correct = (known_top1_id == known_q_labels)
        # external
        ext_scores = cosine_scores(ext_q, protos) if ext_q.size > 0 else np.zeros((0, protos.shape[0]))
        ext_top1_score = ext_scores.max(axis=1) if ext_scores.size > 0 else np.zeros((0,))

        # genuine scores for verification TAR@FAR
        genuine = genuine_scores_correct_class(
            known_q, known_q_labels, protos, proto_labels,
        )

        # curve-based
        auroc_e, eer_e = auroc_eer(known_top1_score, ext_top1_score)
        dir_e, thr_dir_e = dir_at_fpir(known_top1_score, known_correct.astype(np.int32),
                                       ext_top1_score, target_fpir=0.01)
        tar_e, thr_tar_e = tar_at_far(genuine, ext_top1_score, target_far=0.01)

        # fixed-tau
        kdec = known_decomposition(known_top1_score, known_correct.astype(np.int32), tau)
        edec = unknown_decomposition(ext_top1_score, tau)

        steps.append({
            "t": int(t),
            "gallery_size": int(len(gallery_ids)),
            "tau": float(tau),
            "n_known_probes": kdec["n"],
            "n_external_probes": edec["n"],
            "known_rank1": kdec["rank1"],
            "known_identification_acceptance_at_tau": kdec["identification_acceptance_at_tau"],
            "known_rejection_rate_at_tau": kdec["rejection_rate_at_tau"],
            "known_misidentification_acceptance_at_tau": kdec["misidentification_acceptance_at_tau"],
            "known_total_error_at_tau": kdec["total_error_at_tau"],
            "external_false_accept_rate_at_tau": edec["false_accept_rate_at_tau"],
            "external_rejection_rate_at_tau": edec["rejection_rate_at_tau"],
            "mean_known_top1_score": kdec["mean_top1_score"],
            "mean_external_top1_score": edec["mean_top1_score"],
            # curve-based
            "AUROC_known_vs_external": auroc_e,
            "EER_known_vs_external": eer_e,
            "DIR_at_FPIR1pct_known_vs_external": dir_e,
            "DIR_at_FPIR1pct_known_vs_external_threshold": thr_dir_e,
            "TAR_at_FAR1pct_known_vs_external": tar_e,
            "TAR_at_FAR1pct_known_vs_external_threshold": thr_tar_e,
        })
    return {
        "protocol": "B_static_open_set",
        "schedule": list(t_schedule),
        "tau": float(tau),
        "steps": steps,
    }


# ============================================================
# Protocol C: sequential enrollment open-set
# ============================================================

def run_protocol_c(
    store: EmbeddingStore,
    enrollment_order: List[int],
    external_test_ids: List[int],
    t_schedule: List[int],
    tau: float,
) -> Dict:
    steps: List[Dict] = []
    score_distributions: Dict[str, Dict[str, List[float]]] = {}
    ext_q, ext_q_labels = store.select_with_labels(
        palm_ids=external_test_ids, session="session2"
    )
    for t in t_schedule:
        enrolled = enrollment_order[:t]
        remaining_future = enrollment_order[t:]
        protos, proto_labels = build_prototypes(store, enrolled, "session1")

        # known
        known_q, known_q_labels = store.select_with_labels(
            palm_ids=enrolled, session="session2"
        )
        known_scores = cosine_scores(known_q, protos)
        known_top1_score, known_top1_id = top1(known_scores, proto_labels)
        known_correct = (known_top1_id == known_q_labels)
        genuine = genuine_scores_correct_class(
            known_q, known_q_labels, protos, proto_labels,
        )

        # remaining future (may be empty at t=150)
        if len(remaining_future) > 0:
            fut_q, _ = store.select_with_labels(
                palm_ids=remaining_future, session="session2"
            )
            fut_scores = cosine_scores(fut_q, protos)
            fut_top1_score = fut_scores.max(axis=1) if fut_scores.size > 0 else np.zeros((0,))
        else:
            fut_top1_score = np.zeros((0,), dtype=np.float32)

        # external
        if ext_q.size > 0:
            ext_scores = cosine_scores(ext_q, protos)
            ext_top1_score = ext_scores.max(axis=1)
        else:
            ext_top1_score = np.zeros((0,), dtype=np.float32)

        # curve-based vs future / external
        if fut_top1_score.size > 0:
            auroc_f, eer_f = auroc_eer(known_top1_score, fut_top1_score)
            dir_f, _ = dir_at_fpir(known_top1_score, known_correct.astype(np.int32),
                                   fut_top1_score, target_fpir=0.01)
            tar_f, _ = tar_at_far(genuine, fut_top1_score, target_far=0.01)
        else:
            auroc_f = eer_f = dir_f = tar_f = None

        auroc_e, eer_e = auroc_eer(known_top1_score, ext_top1_score)
        dir_e, _ = dir_at_fpir(known_top1_score, known_correct.astype(np.int32),
                               ext_top1_score, target_fpir=0.01)
        tar_e, _ = tar_at_far(genuine, ext_top1_score, target_far=0.01)

        kdec = known_decomposition(known_top1_score, known_correct.astype(np.int32), tau)
        fdec = unknown_decomposition(fut_top1_score, tau)
        edec = unknown_decomposition(ext_top1_score, tau)

        step_dict = {
            "t": int(t),
            "gallery_size": int(len(enrolled)),
            "n_remaining_future": int(len(remaining_future)),
            "tau": float(tau),
            "n_known_probes": kdec["n"],
            "n_future_probes": fdec["n"],
            "n_external_probes": edec["n"],

            "known_rank1": kdec["rank1"],
            "known_identification_acceptance_at_tau": kdec["identification_acceptance_at_tau"],
            "known_rejection_rate_at_tau": kdec["rejection_rate_at_tau"],
            "known_misidentification_acceptance_at_tau": kdec["misidentification_acceptance_at_tau"],
            "known_total_error_at_tau": kdec["total_error_at_tau"],

            "future_false_accept_rate_at_tau": fdec["false_accept_rate_at_tau"],
            "future_rejection_rate_at_tau": fdec["rejection_rate_at_tau"],

            "external_false_accept_rate_at_tau": edec["false_accept_rate_at_tau"],
            "external_rejection_rate_at_tau": edec["rejection_rate_at_tau"],

            "mean_known_top1_score": kdec["mean_top1_score"],
            "mean_future_top1_score": fdec["mean_top1_score"],
            "mean_external_top1_score": edec["mean_top1_score"],

            # curve-based (None at t=150 for future)
            "AUROC_known_vs_future": auroc_f,
            "EER_known_vs_future": eer_f,
            "DIR_at_FPIR1pct_known_vs_future": dir_f,
            "TAR_at_FAR1pct_known_vs_future": tar_f,

            "AUROC_known_vs_external": auroc_e,
            "EER_known_vs_external": eer_e,
            "DIR_at_FPIR1pct_known_vs_external": dir_e,
            "TAR_at_FAR1pct_known_vs_external": tar_e,
        }
        steps.append(step_dict)

        # save score distributions for visualization (keep size small)
        score_distributions[f"t_{int(t)}"] = {
            "known_top1_scores": known_top1_score.tolist(),
            "future_top1_scores": fut_top1_score.tolist(),
            "external_top1_scores": ext_top1_score.tolist(),
        }

    return {
        "protocol": "C_sequential_open_set",
        "schedule": list(t_schedule),
        "tau": float(tau),
        "steps": steps,
        "score_distributions": score_distributions,
    }


# ============================================================
# Main
# ============================================================

def _json_safe(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return _json_safe(o.tolist())
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Protocol A/B/C evaluation")
    parser.add_argument("--embeddings", required=True, type=str)
    parser.add_argument("--manifest_dir", required=True, type=str)
    parser.add_argument("--model_tag", required=True, type=str,
                        help="e.g. mfn or ir50 — used for output filenames")
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--target_far", default=0.01, type=float)
    parser.add_argument(
        "--t_schedule", default="10,20,30,40,50,60,70,80,90,100,110,120,130,140,150",
        type=str, help="comma-separated enrollment sizes"
    )
    args = parser.parse_args(argv)

    npz_path = Path(args.embeddings)
    if not npz_path.is_absolute():
        npz_path = (PROJECT_ROOT / npz_path).resolve()
    manifest_dir = Path(args.manifest_dir)
    if not manifest_dir.is_absolute():
        manifest_dir = (PROJECT_ROOT / manifest_dir).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_dir / "identity_split.json", "r", encoding="utf-8") as f:
        identity_split = json.load(f)

    threshold_gallery_ids = [int(x) for x in identity_split["threshold_gallery_ids"]]
    threshold_unknown_ids = [int(x) for x in identity_split["threshold_unknown_ids"]]
    enrollment_order = [int(x) for x in identity_split["enrollment_order"]]
    external_test_ids = [int(x) for x in identity_split["external_test_ids"]]

    # Hard-rule sanity checks
    base_set = set(int(x) for x in identity_split["base_ids"])
    fut_set = set(int(x) for x in identity_split["future_ids"])
    ext_set = set(int(x) for x in identity_split["external_test_ids"])
    tval_set = set(int(x) for x in identity_split["threshold_val_ids"])
    assert not (tval_set & fut_set), "threshold_val ∩ future should be empty"
    assert not (tval_set & ext_set), "threshold_val ∩ external_test should be empty"
    assert not (base_set & tval_set), "base ∩ threshold_val should be empty"
    assert not (base_set & fut_set), "base ∩ future should be empty"
    assert not (base_set & ext_set), "base ∩ external_test should be empty"

    t_schedule = [int(x) for x in args.t_schedule.split(",") if x.strip()]
    print(f"[schedule] t = {t_schedule}")

    store = EmbeddingStore(npz_path)
    print(f"[store] embeddings: {store.embeddings.shape}")

    # Phase 5: calibrate tau
    cal = phase5_calibrate_tau(
        store, threshold_gallery_ids, threshold_unknown_ids,
        target_far=args.target_far,
    )
    tau = float(cal["tau"])
    print(
        f"[tau] tau={tau:.5f} target_far={cal['target_far']} "
        f"realized_far={cal['realized_far']:.5f} "
        f"(n_unknown_probes={cal['n_threshold_unknown_probes']})"
    )

    # Protocol A
    print("[protocol A] running...")
    res_a = run_protocol_a(store, enrollment_order, t_schedule)
    res_a["calibration"] = cal

    # Protocol B
    print("[protocol B] running...")
    res_b = run_protocol_b(store, enrollment_order, external_test_ids, t_schedule, tau)
    res_b["calibration"] = cal

    # Protocol C
    print("[protocol C] running...")
    res_c = run_protocol_c(store, enrollment_order, external_test_ids, t_schedule, tau)
    res_c["calibration"] = cal

    tag = args.model_tag
    for proto_letter, payload in [("a", res_a), ("b", res_b), ("c", res_c)]:
        path = output_dir / f"{tag}_protocol_{proto_letter}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False)
        print(f"[save] {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
