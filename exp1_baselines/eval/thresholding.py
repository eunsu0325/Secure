"""Tie-aware threshold calibration for FPIR-targeted operating points.

The function below is the canonical implementation from the plan's
"Threshold Calibration (tie-aware)" section — it returns an OBSERVED-score
threshold that empirically guarantees `realized_fpir <= target_fpir`, with
ties handled conservatively via `np.nextafter` only when the raw quantile
threshold would exceed the target due to repeated values.

Inputs are the per-query top-1 scores against the operational gallery (NOT
all pairwise scores). FPIR convention: fraction of unknown queries whose
top-1 score exceeds τ.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def calibrate_threshold_at_fpir(
    scores: Iterable[float], target_fpir: float
) -> float:
    """Empirical threshold guaranteeing realized FPIR ≤ target_fpir.

    The threshold is chosen from observed score values when possible; ties at
    the candidate threshold are handled conservatively by nudging just above
    the tied score only when necessary (when the raw threshold's realized
    FPIR exceeds target due to ties).

    NOT the mathematical "smallest threshold" in continuous threshold space —
    returns an observed-score-based threshold that guarantees the empirical
    FPIR constraint.

    Args:
        scores: per-query top-1 scores from unknown calibration probes (the
            "scores" against the operational gallery). For Protocol B these
            come from external_dev × full_gallery; for C-recal at step t,
            from external_dev × (base_anchor + future_order[:t]).
        target_fpir: target false-positive identification rate (e.g. 0.01).

    Returns:
        τ such that `(scores >= τ).mean() <= target_fpir`. If `scores` is
        empty, returns `+inf`. If no false accept is allowed (target_fpir = 0
        or `floor(target_fpir * n) == 0`), returns the smallest float strictly
        greater than `max(scores)`.
    """
    arr = np.sort(np.asarray(list(scores), dtype=np.float64))
    n = arr.size
    if n == 0:
        return float("inf")
    max_accept = int(np.floor(target_fpir * n))
    if max_accept <= 0:
        return float(np.nextafter(arr[-1], np.inf))

    raw_tau = float(arr[n - max_accept])
    realized_raw = float((arr >= raw_tau).mean())
    if realized_raw <= target_fpir + 1e-12:
        return raw_tau

    tau = float(np.nextafter(raw_tau, np.inf))
    realized = float((arr >= tau).mean())
    if realized > target_fpir + 1e-12:
        raise AssertionError(
            f"calibrate_threshold_at_fpir: even after nudge tau={tau} "
            f"yields realized={realized} > target={target_fpir}"
        )
    return tau
