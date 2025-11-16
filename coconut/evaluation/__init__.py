"""
COCONUT Evaluation Module
========================
Continual Learning evaluation with proper forgetting measurement
"""

from .forgetting_tracker import ForgettingTracker
from .biometric_metrics import (
    calculate_eer,
    calculate_tar_at_far,
    calculate_similarity_margin,
    calculate_all_metrics,
    compute_roc_curve
)
from .visualization import (
    plot_roc_curve,
    plot_det_curve,
    generate_far_tar_table,
    plot_score_distributions,
    save_evaluation_curves
)

__all__ = [
    'ForgettingTracker',
    'calculate_eer',
    'calculate_tar_at_far',
    'calculate_similarity_margin',
    'calculate_all_metrics',
    'compute_roc_curve',
    'plot_roc_curve',
    'plot_det_curve',
    'generate_far_tar_table',
    'plot_score_distributions',
    'save_evaluation_curves'
]