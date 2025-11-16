"""Basic utility functions for open-set recognition"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple, Set, Optional
import os
import random
from tqdm import tqdm


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # For complete reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _open_with_channels(path: str, channels: int) -> Image.Image:
    """Open image and convert to appropriate channels"""
    with Image.open(path) as img:
        if channels == 1:
            return img.convert('L')
        else:
            return img.convert('RGB')


def split_user_data(paths: List[str], labels: List[int],
                   dev_ratio: float = 0.2, min_dev: int = 2) -> Tuple:
    """Split user data into train/dev sets"""
    idx = np.arange(len(paths))
    np.random.shuffle(idx)
    n_dev = max(min_dev, int(len(paths) * dev_ratio))

    dev_idx = idx[:n_dev]
    train_idx = idx[n_dev:]

    train_paths = [paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    dev_paths = [paths[i] for i in dev_idx]
    dev_labels = [labels[i] for i in dev_idx]

    return train_paths, train_labels, dev_paths, dev_labels


def load_paths_labels_from_txt(txt_file: str) -> Tuple[List[str], List[int]]:
    """Load paths and labels from text file"""
    paths, labels = [], []
    if not os.path.exists(txt_file):
        return paths, labels

    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) == 2:
                paths.append(parts[0])
                labels.append(int(parts[1]))
    return paths, labels


def load_paths_labels_excluding(txt_file: str, exclude_labels: Set[int]) -> Tuple[List[str], List[int]]:
    """Load paths and labels excluding specific labels"""
    all_paths, all_labels = load_paths_labels_from_txt(txt_file)
    paths, labels = [], []

    for p, l in zip(all_paths, all_labels):
        if l not in exclude_labels:
            paths.append(p)
            labels.append(l)

    return paths, labels


def balance_impostor_scores(
    s_between: np.ndarray,
    s_unknown: Optional[np.ndarray] = None,
    s_negref: Optional[np.ndarray] = None,
    ratio_config: Optional[dict] = None,
    total: Optional[int] = None,
    verbose: bool = False
) -> np.ndarray:
    """
    Balance impostor scores from different sources

    Args:
        s_between: Between-class impostor scores
        s_unknown: Unknown class impostor scores
        s_negref: Negative reference impostor scores
        ratio_config: Dictionary with weights for each type
        total: Target total number of scores
        verbose: Print statistics

    Returns:
        Balanced impostor scores
    """
    available = []
    weights = []

    # Collect available scores
    if len(s_between) > 0:
        available.append(s_between)
        weight_key = 'between'
        weights.append(ratio_config.get(weight_key, 1.0) if ratio_config else 1.0)

    if s_unknown is not None and len(s_unknown) > 0:
        available.append(s_unknown)
        weight_key = 'unknown'
        weights.append(ratio_config.get(weight_key, 0.0) if ratio_config else 0.0)

    if s_negref is not None and len(s_negref) > 0:
        available.append(s_negref)
        weight_key = 'negref'
        weights.append(ratio_config.get(weight_key, 1.0) if ratio_config else 1.0)

    if not available:
        return np.array([])

    # Normalize weights
    weights = np.array(weights)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(weights)) / len(weights)

    # Determine counts
    if total is None:
        total = sum(len(a) for a in available)

    counts = [int(total * w) for w in weights]

    # Adjust for rounding
    diff = total - sum(counts)
    if diff > 0:
        counts[0] += diff

    # Sample from each source
    balanced = []
    for scores, count in zip(available, counts):
        if count > 0:
            if count <= len(scores):
                sampled = np.random.choice(scores, count, replace=False)
            else:
                sampled = scores  # Use all available
            balanced.append(sampled)

    if balanced:
        result = np.concatenate(balanced)
        np.random.shuffle(result)

        if verbose:
            print(f"Balanced impostor scores: total={len(result)}")
            for i, (scores, count) in enumerate(zip(available, counts)):
                print(f"  Source {i}: {count}/{len(scores)} samples")

        return result

    return np.array([])


@torch.no_grad()
def predict_batch(model, ncm, paths: List[str], transform, device,
                 batch_size: int = 32, channels: int = 1) -> np.ndarray:
    """
    Batch prediction using NCM classifier

    Args:
        model: Feature extraction model
        ncm: NCM classifier
        paths: List of image paths
        transform: Image transformation
        device: Torch device
        batch_size: Batch size for processing
        channels: Number of image channels

    Returns:
        Predicted class IDs
    """
    if not paths:
        return np.array([])

    model.eval()
    preds = []

    for i in range(0, len(paths), batch_size):
        batch = []
        for p in paths[i:i+batch_size]:
            img = _open_with_channels(p, channels)
            img = transform(img)
            batch.append(img)

        x = torch.stack(batch, dim=0).to(device)
        feat = model.getFeatureCode(x)
        # Normalization handled by NCM
        pred = ncm.predict_openset(feat)
        preds.extend(pred.cpu().numpy())

    return np.array(preds)