"""Open-set recognition utilities for COCONUT"""

from .utils import (
    set_seed,
    split_user_data,
    load_paths_labels_from_txt,
    load_paths_labels_excluding,
    balance_impostor_scores,
    predict_batch,
    _open_with_channels  # Internal utility
)

from .score_extraction import (
    extract_features,  # Feature extraction
    extract_scores_genuine,
    extract_scores_impostor_between,
    extract_scores_impostor_unknown,
    extract_scores_impostor_negref,
    extract_scores_for_user  # Per-user evaluation
)

from .tta_operations import (
    get_light_augmentation,
    predict_batch_tta,
    extract_scores_genuine_tta,
    extract_scores_impostor_between_tta,
    extract_scores_impostor_negref_tta
)

__all__ = [
    # Utils
    'set_seed',
    'split_user_data',
    'load_paths_labels_from_txt',
    'load_paths_labels_excluding',
    'balance_impostor_scores',
    'predict_batch',
    '_open_with_channels',
    # Score extraction
    'extract_features',
    'extract_scores_genuine',
    'extract_scores_impostor_between',
    'extract_scores_impostor_unknown',
    'extract_scores_impostor_negref',
    'extract_scores_for_user',
    # TTA operations
    'get_light_augmentation',
    'predict_batch_tta',
    'extract_scores_genuine_tta',
    'extract_scores_impostor_between_tta',
    'extract_scores_impostor_negref_tta'
]