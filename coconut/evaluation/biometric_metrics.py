"""
Biometric Evaluation Metrics
=============================
EER, TAR@FAR, and other metrics for biometric authentication systems
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from scipy.interpolate import interp1d
from scipy.optimize import brentq


def calculate_eer(genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> float:
    """
    Calculate Equal Error Rate (EER).

    Args:
        genuine_scores: Similarity scores for genuine attempts
        impostor_scores: Similarity scores for impostor attempts

    Returns:
        EER value (0-1)
    """
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        return 0.5  # Return worst case if no data

    # Convert to numpy arrays
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    # Calculate FAR and FRR for different thresholds
    thresholds = np.unique(np.concatenate([genuine_scores, impostor_scores]))
    thresholds = np.sort(thresholds)

    far_list = []
    frr_list = []

    for threshold in thresholds:
        # FAR: impostor scores >= threshold (incorrectly accepted)
        far = np.mean(impostor_scores >= threshold)

        # FRR: genuine scores < threshold (incorrectly rejected)
        frr = np.mean(genuine_scores < threshold)

        far_list.append(far)
        frr_list.append(frr)

    far_array = np.array(far_list)
    frr_array = np.array(frr_list)

    # Find EER - where FAR and FRR curves intersect
    try:
        # Use interpolation to find exact intersection
        if len(thresholds) > 1:
            # Create interpolation functions
            far_interp = interp1d(thresholds, far_array, kind='linear',
                                 bounds_error=False, fill_value='extrapolate')
            frr_interp = interp1d(thresholds, frr_array, kind='linear',
                                 bounds_error=False, fill_value='extrapolate')

            # Define difference function
            def diff_far_frr(t):
                return far_interp(t) - frr_interp(t)

            # Find root (where FAR = FRR)
            threshold_min, threshold_max = thresholds.min(), thresholds.max()

            # Check if there's a sign change
            if diff_far_frr(threshold_min) * diff_far_frr(threshold_max) < 0:
                eer_threshold = brentq(diff_far_frr, threshold_min, threshold_max)
                eer = float(far_interp(eer_threshold))
            else:
                # If no exact intersection, find minimum distance
                idx = np.argmin(np.abs(far_array - frr_array))
                eer = (far_array[idx] + frr_array[idx]) / 2
        else:
            eer = 0.5

    except Exception:
        # Fallback: find point where FAR and FRR are closest
        idx = np.argmin(np.abs(far_array - frr_array))
        eer = (far_array[idx] + frr_array[idx]) / 2

    return float(eer)


def calculate_tar_at_far(genuine_scores: np.ndarray, impostor_scores: np.ndarray,
                         target_far: float = 0.01) -> float:
    """
    Calculate True Accept Rate at a specific False Accept Rate.

    Args:
        genuine_scores: Similarity scores for genuine attempts
        impostor_scores: Similarity scores for impostor attempts
        target_far: Target FAR (e.g., 0.01 for 1%, 0.001 for 0.1%)

    Returns:
        TAR at the specified FAR
    """
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        return 0.0

    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    # Sort impostor scores to find threshold for target FAR
    sorted_impostor = np.sort(impostor_scores)[::-1]  # Descending order

    # Find threshold that gives us the target FAR
    n_impostors = len(impostor_scores)
    index = int(n_impostors * target_far)
    index = min(index, n_impostors - 1)

    if index >= 0 and index < n_impostors:
        threshold = sorted_impostor[index]
    else:
        threshold = sorted_impostor[0] if n_impostors > 0 else 0

    # Calculate TAR at this threshold
    tar = np.mean(genuine_scores >= threshold)

    return float(tar)


def calculate_similarity_margin(genuine_scores: np.ndarray,
                               impostor_scores: np.ndarray) -> float:
    """
    Separation between genuine and impostor score distributions, reported as
    **d-prime** (detection-theory sensitivity index):

        d' = (mu_genuine - mu_impostor) / sqrt(0.5 * (var_g + var_i))

    Why d-prime (and not the raw mean difference): the scores fed here can be
    S-norm per-class z-scores (use_snorm=True), whose raw mean-difference is
    scale-dependent and blows up for near-degenerate classes (sigma->0) —
    e.g. an early-cohort 'margin' BWT of -1320 was such an artifact, not a real
    1320-unit drop. d-prime is **scale-invariant** (same on raw cosine or
    z-scores) and the pooled-std denominator + floor keep it bounded and
    outlier-robust. Higher = better genuine/impostor separation; lower over
    time = the separation erosion that disallocation/AAVB targets (non-saturating,
    unlike TAR@FAR which caps at 1.0).
    """
    g = np.asarray(genuine_scores, dtype=float)
    i = np.asarray(impostor_scores, dtype=float)
    if g.size == 0 or i.size == 0:
        return 0.0

    num = float(g.mean() - i.mean())
    # pooled std; impostor set is large so var_i>0 keeps this well-conditioned.
    pooled = float(np.sqrt(0.5 * (g.var() + i.var())))
    return num / (pooled + 1e-6)


def compute_roc_curve(genuine_scores: np.ndarray, impostor_scores: np.ndarray,
                     n_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve (FAR vs TAR).

    Args:
        genuine_scores: Similarity scores for genuine attempts
        impostor_scores: Similarity scores for impostor attempts
        n_points: Number of points for the curve

    Returns:
        (far_array, tar_array, thresholds)
    """
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        return np.array([0, 1]), np.array([0, 1]), np.array([0, 1])

    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    # Generate thresholds
    min_score = min(genuine_scores.min(), impostor_scores.min())
    max_score = max(genuine_scores.max(), impostor_scores.max())
    thresholds = np.linspace(min_score, max_score, n_points)

    far_list = []
    tar_list = []

    for threshold in thresholds:
        # FAR: impostor scores >= threshold
        far = np.mean(impostor_scores >= threshold)

        # TAR: genuine scores >= threshold
        tar = np.mean(genuine_scores >= threshold)

        far_list.append(far)
        tar_list.append(tar)

    return np.array(far_list), np.array(tar_list), thresholds


def calculate_all_metrics(genuine_scores: np.ndarray,
                         impostor_scores: np.ndarray) -> Dict[str, float]:
    """
    Calculate all biometric metrics at once.

    Args:
        genuine_scores: Similarity scores for genuine attempts
        impostor_scores: Similarity scores for impostor attempts

    Returns:
        Dictionary with all metrics
    """
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    metrics = {
        'eer': calculate_eer(genuine_scores, impostor_scores),
        'tar_001': calculate_tar_at_far(genuine_scores, impostor_scores, 0.01),
        'tar_005': calculate_tar_at_far(genuine_scores, impostor_scores, 0.05),
        'tar_010': calculate_tar_at_far(genuine_scores, impostor_scores, 0.10),
        'tar_0001': calculate_tar_at_far(genuine_scores, impostor_scores, 0.001),
        'tar_00001': calculate_tar_at_far(genuine_scores, impostor_scores, 0.0001),
        'margin': calculate_similarity_margin(genuine_scores, impostor_scores),
        'genuine_mean': float(np.mean(genuine_scores)) if len(genuine_scores) > 0 else 0.0,
        'genuine_std': float(np.std(genuine_scores)) if len(genuine_scores) > 0 else 0.0,
        'impostor_mean': float(np.mean(impostor_scores)) if len(impostor_scores) > 0 else 0.0,
        'impostor_std': float(np.std(impostor_scores)) if len(impostor_scores) > 0 else 0.0,
    }

    # Add 1-EER for convenience
    metrics['1-eer'] = 1.0 - metrics['eer']

    # Add FAR at EER threshold
    metrics['far_at_eer'] = metrics['eer']
    metrics['frr_at_eer'] = metrics['eer']

    return metrics


def find_threshold_for_far(impostor_scores: np.ndarray, target_far: float) -> float:
    """
    Find the threshold that achieves a specific FAR.

    Args:
        impostor_scores: Similarity scores for impostor attempts
        target_far: Target FAR

    Returns:
        Threshold value
    """
    if len(impostor_scores) == 0:
        return 0.5

    impostor_scores = np.array(impostor_scores)
    sorted_scores = np.sort(impostor_scores)[::-1]  # Descending

    n_impostors = len(impostor_scores)
    index = int(n_impostors * target_far)
    index = min(max(0, index), n_impostors - 1)

    return float(sorted_scores[index])


def evaluate_at_threshold(genuine_scores: np.ndarray, impostor_scores: np.ndarray,
                         threshold: float) -> Dict[str, float]:
    """
    Evaluate performance at a specific threshold.

    Args:
        genuine_scores: Similarity scores for genuine attempts
        impostor_scores: Similarity scores for impostor attempts
        threshold: Decision threshold

    Returns:
        Dictionary with TAR, FAR, FRR
    """
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    tar = np.mean(genuine_scores >= threshold) if len(genuine_scores) > 0 else 0.0
    far = np.mean(impostor_scores >= threshold) if len(impostor_scores) > 0 else 0.0
    frr = 1.0 - tar

    return {
        'threshold': threshold,
        'tar': float(tar),
        'far': float(far),
        'frr': float(frr)
    }