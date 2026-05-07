"""Shared metric primitives for Protocols A/B/C.

Provides:
  - `argmax_with_tiebreak`     : deterministic per-row top-1 (ties → lowest palm_id)
  - `wilson_ci`                : Wilson 95% CI for binomial proportions
  - `compute_known_metrics`    : TPIR / FNIR + decomposition (correct accept,
                                 wrong accept, reject) per the plan's
                                 "Known query decomposition" rule
  - `compute_fpir_metrics`     : false-accept rate for unknown probes
  - `min_observable_fpir`      : 1 / n  (operational floor)
  - `reliability_flag`         : ✓ / ⚠ / ⚠⚠ per the n × target_fpir rule
  - `is_stable_step`           : C_mean_stable selection criterion (n=100 floor +
                                 n×fpir≥5 + t<t_max)
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


# Z-score for the standard 95% Wilson interval; precomputed.
_Z_95 = 1.959963984540054


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score CI for a binomial proportion k / n.

    Returns (low, high). For n == 0 returns (0.0, 1.0) by convention.
    Only the alpha=0.05 (95%) z-score is precomputed; pass other alphas
    to use a freshly computed z (slower path is rarely exercised here).
    """
    if n == 0:
        return (0.0, 1.0)
    if alpha == 0.05:
        z = _Z_95
    else:
        from math import sqrt as _sqrt  # local import: rarely used branch
        from scipy.stats import norm  # type: ignore
        z = float(norm.ppf(1.0 - alpha / 2.0))
    p = k / n
    z2_n = z * z / n
    denom = 1.0 + z2_n
    center = (p + z2_n / 2.0) / denom
    halfwidth = (z * np.sqrt(p * (1 - p) / n + z2_n / (4 * n))) / denom
    low = max(0.0, center - halfwidth)
    high = min(1.0, center + halfwidth)
    return (float(low), float(high))


def argmax_with_tiebreak(
    scores: np.ndarray, palm_ids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-row argmax over `scores`; on ties, the smallest palm_id wins.

    Args:
        scores: [N_query, N_proto] cosine score matrix slice.
        palm_ids: [N_proto] int array of column-aligned palm_ids.

    Returns:
        top1_palm_ids: [N_query] int — palm_id of the chosen prototype per row.
        top1_scores:   [N_query] float — score at the chosen column.

    Tie-break is deterministic and order-independent: we sort columns by
    palm_id ascending, then call `argmax`, which returns the first occurrence
    of the maximum (i.e. the smallest palm_id among tied maxima).
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2-D, got shape {scores.shape}")
    if scores.shape[1] != palm_ids.shape[0]:
        raise ValueError(
            f"scores cols {scores.shape[1]} != palm_ids len {palm_ids.shape[0]}"
        )
    if scores.shape[1] == 0:
        # Empty gallery → return sentinel: -1 palm_id, -inf score.
        n = scores.shape[0]
        return (
            np.full(n, -1, dtype=np.int64),
            np.full(n, -np.inf, dtype=scores.dtype),
        )
    order = np.argsort(palm_ids, kind="stable")
    sorted_scores = scores[:, order]
    sorted_palms = palm_ids[order]
    cols_in_sorted = sorted_scores.argmax(axis=1)
    top1_palm = sorted_palms[cols_in_sorted].astype(np.int64)
    top1_score = sorted_scores[np.arange(scores.shape[0]), cols_in_sorted]
    return top1_palm, top1_score


def compute_known_metrics(
    top1_palm: np.ndarray,
    top1_score: np.ndarray,
    gt_palm: np.ndarray,
    tau: float,
) -> Dict[str, object]:
    """TPIR/FNIR and 3-way decomposition for known queries (plan: Metrics).

    Per-query categorization:
      correct_accept = (top1_palm == gt) AND (top1_score >= tau)
      wrong_accept   = (top1_palm != gt) AND (top1_score >= tau)
      reject         = (top1_score < tau)

    Aggregates and 95% Wilson CIs are returned; FNIR == 1 - TPIR.
    """
    n = int(top1_palm.shape[0])
    if n == 0:
        return {
            "known_total": 0,
            "known_correct_accept_count": 0,
            "known_wrong_accept_count": 0,
            "known_reject_count": 0,
            "tpir": float("nan"),
            "tpir_ci95": (float("nan"), float("nan")),
            "known_wrong_accept_rate": float("nan"),
            "known_reject_rate": float("nan"),
            "known_failure_rate": float("nan"),
            "fnir": float("nan"),
        }

    accept = top1_score >= tau
    correct = accept & (top1_palm == gt_palm)
    wrong = accept & (top1_palm != gt_palm)
    reject = ~accept

    correct_n = int(correct.sum())
    wrong_n = int(wrong.sum())
    reject_n = int(reject.sum())
    if correct_n + wrong_n + reject_n != n:
        raise AssertionError(
            f"decomposition sum {correct_n+wrong_n+reject_n} != total {n}"
        )

    tpir = correct_n / n
    return {
        "known_total": n,
        "known_correct_accept_count": correct_n,
        "known_wrong_accept_count": wrong_n,
        "known_reject_count": reject_n,
        "tpir": tpir,
        "tpir_ci95": wilson_ci(correct_n, n),
        "known_wrong_accept_rate": wrong_n / n,
        "known_reject_rate": reject_n / n,
        "known_failure_rate": 1.0 - tpir,
        "fnir": 1.0 - tpir,  # NIST FRTE alias
    }


def compute_fpir_metrics(
    top1_score: np.ndarray, tau: float, *, prefix: str
) -> Dict[str, object]:
    """False-accept rate for unknown probes (future or external).

    `prefix` ∈ {"future", "external"} so the keys are namespaced
    (`future_fpir`, `external_fpir`, etc.) for direct merge into a step dict.
    """
    n = int(top1_score.shape[0])
    if n == 0:
        return {
            f"{prefix}_total": 0,
            f"{prefix}_false_accept": 0,
            f"{prefix}_fpir": float("nan"),
            f"{prefix}_fpir_ci95": (float("nan"), float("nan")),
        }
    fa = int((top1_score >= tau).sum())
    fpir = fa / n
    return {
        f"{prefix}_total": n,
        f"{prefix}_false_accept": fa,
        f"{prefix}_fpir": fpir,
        f"{prefix}_fpir_ci95": wilson_ci(fa, n),
    }


def min_observable_fpir(n: int) -> float:
    """Smallest non-zero FPIR observable with `n` unknown probes (= 1/n)."""
    if n <= 0:
        return float("nan")
    return 1.0 / n


def reliability_flag(n: int, target_fpir: float) -> str:
    """Sample-size reliability label per plan: ✓ / ⚠ / ⚠⚠."""
    expected = n * target_fpir
    if expected >= 5:
        return "reliable"
    if expected >= 1:
        return "wide_ci"
    return "noise_dominated"


def is_stable_step(step: Dict[str, object], target_fpir: float) -> bool:
    """C_mean_stable selection criterion (plan: Stable step criterion).

    A step qualifies for averaging only if all of:
      - known_total >= 100 (Wilson CI half-width on TPIR small enough)
      - future_total * target_fpir >= 5
      - external_total * target_fpir >= 5
      - t < t_max (Future-FPIR is undefined at full enrollment)
    """
    if int(step.get("known_total", 0)) < 100:
        return False
    if float(step.get("future_total", 0)) * target_fpir < 5:
        return False
    if float(step.get("external_total", 0)) * target_fpir < 5:
        return False
    if int(step.get("t", -1)) >= int(step.get("t_max", 0)):
        return False
    return True


__all__ = [
    "argmax_with_tiebreak",
    "wilson_ci",
    "compute_known_metrics",
    "compute_fpir_metrics",
    "min_observable_fpir",
    "reliability_flag",
    "is_stable_step",
]
