"""Score extraction functions for open-set recognition"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Set, Optional
from collections import defaultdict
import random
import os
from .utils import _open_with_channels, load_paths_labels_excluding, load_paths_labels_from_txt


@torch.no_grad()
def extract_features(model, paths: List[str], transform, device,
                    batch_size: int = 64, channels: int = 1) -> np.ndarray:
    """
    Extract features from image paths (with L2 normalization)

    Returns:
        np.ndarray: L2 normalized feature vectors (N, D)
    """
    if not paths:
        return np.array([])

    feats = []
    model.eval()

    for i in range(0, len(paths), batch_size):
        batch = []
        for p in paths[i:i+batch_size]:
            try:
                img = _open_with_channels(p, channels)
                img = transform(img)
                batch.append(img)
            except Exception as e:
                print(f"WARNING: Failed to load {p}: {e}")
                continue

        if not batch:
            continue

        x = torch.stack(batch, dim=0).to(device)
        f = model.getFeatureCode(x)
        # Normalization handled by NCM
        feats.append(f.cpu())

    return torch.cat(feats, dim=0).numpy() if feats else np.array([])


@torch.no_grad()
def extract_scores_genuine(model, ncm, dev_paths: List[str], dev_labels: List[int],
                          transform, device, channels: int = 1) -> np.ndarray:
    """Extract genuine scores (MAX mode)"""
    if not dev_paths:
        return np.array([])

    model.eval()
    scores = []

    batch_size = 64
    for i in range(0, len(dev_paths), batch_size):
        batch_paths = dev_paths[i:i+batch_size]

        feats = extract_features(model, batch_paths, transform, device,
                                batch_size=32, channels=channels)
        if len(feats) == 0:
            continue

        feats_tensor = torch.from_numpy(feats).to(device)
        ncm_scores = ncm.forward(feats_tensor)

        if ncm_scores.numel() > 0:
            max_scores = ncm_scores.max(dim=1).values
            scores.extend(max_scores.cpu().numpy())

    return np.array(scores)


@torch.no_grad()
def extract_scores_impostor_between(model, ncm, dev_paths: List[str], dev_labels: List[int],
                                   transform, device, max_pairs: int = 2000,
                                   channels: int = 1) -> np.ndarray:
    """Extract between-class impostor scores (excluding own class)"""
    if not dev_paths:
        return np.array([])

    by_class = defaultdict(list)
    for p, y in zip(dev_paths, dev_labels):
        by_class[int(y)].append(p)

    model.eval()
    scores = []

    for cls_id, cls_paths in by_class.items():
        if len(scores) >= max_pairs:
            break

        sample_paths = cls_paths[:min(10, len(cls_paths))]
        if len(sample_paths) > 5:
            # Use numpy for deterministic sampling with fixed local seed
            rng = np.random.RandomState(42)
            indices = rng.choice(len(sample_paths), 5, replace=False)
            sample_paths = [sample_paths[i] for i in indices]

        for path in sample_paths:
            if len(scores) >= max_pairs:
                break

            feat = extract_features(model, [path], transform, device, channels=channels)
            if len(feat) == 0:
                continue

            feat_tensor = torch.from_numpy(feat).to(device)
            ncm_scores = ncm.forward(feat_tensor)

            if ncm_scores.numel() > 0 and cls_id < ncm_scores.shape[1]:
                ncm_scores[0, cls_id] = -float('inf')
                max_score = ncm_scores.max().item()
                scores.append(max_score)

    return np.array(scores)


@torch.no_grad()
def extract_scores_impostor_unknown(model, ncm, txt_file: str, registered_users: Set[int],
                                   transform, device, max_eval: int = 3000,
                                   channels: int = 1) -> np.ndarray:
    """Extract impostor scores from unregistered users"""
    paths, labels = load_paths_labels_excluding(txt_file, registered_users)

    if not paths:
        return np.array([])

    if len(paths) > max_eval:
        idx = np.random.choice(len(paths), max_eval, replace=False)
        paths = [paths[i] for i in idx]

    feats = extract_features(model, paths, transform, device, channels=channels)

    if len(feats) == 0:
        return np.array([])

    if not hasattr(ncm, 'class_means_dict') or not ncm.class_means_dict:
        return np.array([])

    feats_tensor = torch.from_numpy(feats).to(device)
    ncm_scores = ncm.forward(feats_tensor)

    if ncm_scores.numel() > 0:
        max_scores = ncm_scores.max(dim=1).values
        return max_scores.cpu().numpy()

    return np.array([])


@torch.no_grad()
def extract_scores_impostor_negref(model, ncm, negref_file: str, transform, device,
                                  max_eval: int = 5000, channels: int = 1) -> np.ndarray:
    """Extract impostor scores from negative reference data"""
    if not negref_file or not os.path.exists(negref_file):
        return np.array([])

    paths, labels = load_paths_labels_from_txt(negref_file)

    if not paths:
        return np.array([])

    if len(paths) > max_eval:
        idx = np.random.choice(len(paths), max_eval, replace=False)
        paths = [paths[i] for i in idx]

    feats = extract_features(model, paths, transform, device, channels=channels)

    if len(feats) == 0:
        return np.array([])

    if not hasattr(ncm, 'class_means_dict') or not ncm.class_means_dict:
        return np.array([])

    feats_tensor = torch.from_numpy(feats).to(device)
    ncm_scores = ncm.forward(feats_tensor)

    if ncm_scores.numel() > 0:
        max_scores = ncm_scores.max(dim=1).values
        return max_scores.cpu().numpy()

    return np.array([])


@torch.no_grad()
def extract_scores_for_user(model, ncm, user_id: int, test_paths: List[str], test_labels: List[int],
                           all_registered_users: Set[int], transform, device,
                           channels: int = 1) -> tuple:
    """
    Extract genuine and impostor scores for a specific user.

    Args:
        model: Neural network model
        ncm: NCM classifier
        user_id: User to evaluate
        test_paths: Test sample paths
        test_labels: Test sample labels
        all_registered_users: Set of all registered user IDs
        transform: Image transform
        device: Computation device
        channels: Number of image channels

    Returns:
        (genuine_scores, impostor_scores) as numpy arrays
    """
    if not test_paths:
        return np.array([]), np.array([])

    model.eval()
    genuine_scores = []
    impostor_scores = []

    # Filter test samples for this user
    user_test_paths = []
    for path, label in zip(test_paths, test_labels):
        if int(label) == user_id:
            user_test_paths.append(path)

    if not user_test_paths:
        return np.array([]), np.array([])

    # Extract features for user's test samples
    feats = extract_features(model, user_test_paths, transform, device,
                           batch_size=32, channels=channels)

    if len(feats) == 0:
        return np.array([]), np.array([])

    feats_tensor = torch.from_numpy(feats).to(device)

    # Calculate scores
    for feat in feats_tensor:
        feat = feat.unsqueeze(0)  # Add batch dimension

        # Genuine: similarity to correct user
        if hasattr(ncm, 'similarity'):
            sim_genuine = ncm.similarity(feat, user_id)
            if sim_genuine is not None:
                genuine_scores.append(float(sim_genuine))

        # Impostor: similarities to other users
        for other_id in all_registered_users:
            if other_id != user_id:
                if hasattr(ncm, 'similarity'):
                    sim_impostor = ncm.similarity(feat, other_id)
                    if sim_impostor is not None:
                        impostor_scores.append(float(sim_impostor))

    return np.array(genuine_scores), np.array(impostor_scores)