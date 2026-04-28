"""Calibration helpers shared by Exp1 protocol and summaries."""

import numpy as np


def threshold_from_dev_scores(scores, fpir):
    """Conservative threshold from external-dev impostor max-scores."""
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("empty dev impostor scores")
    return float(np.quantile(arr, 1.0 - float(fpir), method="higher"))
