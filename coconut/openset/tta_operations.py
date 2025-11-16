"""Test-Time Augmentation (TTA) operations for open-set recognition"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple, Union, Dict, Optional
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from collections import Counter

from .utils import _open_with_channels, set_seed, load_paths_labels_from_txt
from .score_extraction import extract_features


def get_light_augmentation(strength: float = 0.5, img_size: int = 128) -> T.Compose:
    """Create light augmentation pipeline for TTA

    Args:
        strength: Augmentation strength (0 to 1)
        img_size: Target image size

    Returns:
        Composed augmentation pipeline
    """
    transforms = []

    if strength > 0:
        transforms.append(T.RandomRotation(
            degrees=15 * strength
        ))

    if strength > 0.2:
        transforms.append(T.RandomPerspective(
            distortion_scale=0.15 * strength,
            p=0.7
        ))

    if strength > 0.3:
        transforms.append(T.RandomResizedCrop(
            size=img_size,
            scale=(1.0 - 0.15*strength, 1.0),
            ratio=(0.9, 1.1),
            interpolation=InterpolationMode.BICUBIC
        ))

    if strength > 0.5:
        transforms.append(T.RandomChoice([
            T.ColorJitter(brightness=0.02, contrast=0.03),
            T.RandomAffine(
                degrees=0,
                translate=(0.05 * strength, 0.05 * strength),
                scale=(0.95, 1.05)
            ),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
        ]))

    return T.Compose(transforms) if transforms else T.Lambda(lambda x: x)


@torch.no_grad()
def predict_batch_tta(
    model,
    ncm,
    paths: List[str],
    base_transform,
    device,
    n_views: int = 3,
    include_original: bool = True,
    agree_k: int = 0,
    aug_strength: float = 0.5,
    batch_size: int = 32,
    return_details: bool = False,
    img_size: int = 128,
    channels: int = 1,
    verbose: bool = False,
    seed: int = 42
) -> Union[np.ndarray, Tuple[np.ndarray, List[Dict]]]:
    """
    Test-Time Augmentation based multi-view consensus prediction

    Args:
        model: Feature extraction model
        ncm: NCM classifier
        paths: List of image paths
        base_transform: Base image transformation
        device: Torch device
        n_views: Number of augmented views
        include_original: Include original image as first view
        agree_k: Minimum agreement count for acceptance
        aug_strength: Augmentation strength
        batch_size: Batch size for processing
        return_details: Return detailed results
        img_size: Image size
        channels: Number of channels
        verbose: Print progress
        seed: Random seed for reproducibility

    Returns:
        Predictions array, optionally with details
    """
    # Set seed for reproducibility
    set_seed(seed)

    if not paths:
        return (np.array([]), []) if return_details else np.array([])

    # If single view, fall back to regular prediction
    if n_views <= 1:
        from .utils import predict_batch
        preds = predict_batch(model, ncm, paths, base_transform, device,
                            batch_size, channels)
        if return_details:
            details = [{
                'view_scores': [0.0],
                'view_preds': [int(p) if p != -1 else -1],
                'agree_count': 1 if p != -1 else 0,
                'agree_k': 1,
                'final_decision': (p != -1, int(p) if p != -1 else -1),
                'reject_reason': None if p != -1 else 'gate'
            } for p in preds]
            return preds, details
        return preds

    eff_batch_size = max(1, batch_size // max(1, n_views // 2))

    if agree_k <= 0:
        agree_k = (n_views + 1) // 2
    else:
        agree_k = min(max(agree_k, 1), n_views)

    light_aug = get_light_augmentation(aug_strength, img_size)

    model.eval()
    all_predictions = []
    all_details = []

    if verbose:
        all_accept_ratios = []
        all_disagreement_rates = []

    for batch_start in range(0, len(paths), eff_batch_size):
        batch_paths = paths[batch_start:batch_start + eff_batch_size]
        batch_views = []

        for path in batch_paths:
            try:
                img = _open_with_channels(path, channels)
            except Exception as e:
                print(f"WARNING: Failed to load {path}: {e}")
                all_predictions.append(-1)
                if return_details:
                    all_details.append({
                        'view_scores': [],
                        'view_preds': [],
                        'agree_count': 0,
                        'agree_k': agree_k,
                        'final_decision': (False, -1),
                        'reject_reason': 'load_error'
                    })
                continue

            views = []

            if include_original:
                views.append(base_transform(img))
                for _ in range(n_views - 1):
                    aug_img = light_aug(img) if aug_strength > 0 else img
                    views.append(base_transform(aug_img))
            else:
                for _ in range(n_views):
                    aug_img = light_aug(img) if aug_strength > 0 else img
                    views.append(base_transform(aug_img))

            batch_views.append(torch.stack(views))

        if not batch_views:
            continue

        batch_tensor = torch.stack(batch_views)
        B = batch_tensor.shape[0]
        batch_flat = batch_tensor.view(B * n_views, *batch_tensor.shape[2:]).to(device)

        features = model.getFeatureCode(batch_flat)
        features = features.view(B, n_views, -1)

        for i in range(B):
            view_scores = []
            view_preds = []
            view_accepts = []

            for v in range(n_views):
                ncm_scores = ncm.forward(features[i:i+1, v])

                if ncm.predict_openset is not None:
                    pred = ncm.predict_openset(features[i:i+1, v])
                    pred_id = int(pred[0].item())

                    if ncm_scores.numel() > 0:
                        max_score = float(ncm_scores.max().item())
                    else:
                        max_score = 0.0
                else:
                    if ncm_scores.numel() > 0:
                        max_score = float(ncm_scores.max().item())
                        pred_id = int(ncm_scores.argmax().item())
                    else:
                        max_score = 0.0
                        pred_id = -1

                view_scores.append(max_score)
                view_preds.append(pred_id)
                view_accepts.append(pred_id != -1)

            accept_count = sum(view_accepts)
            accept_ratio = accept_count / n_views

            if verbose:
                all_accept_ratios.append(accept_ratio)

            if accept_count >= agree_k:
                valid_preds = [p for p in view_preds if p != -1]

                if valid_preds:
                    vote_counts = Counter(valid_preds)
                    consensus_pred = vote_counts.most_common(1)[0][0]

                    disagreement_rate = 1.0 - (vote_counts[consensus_pred] / len(valid_preds))
                    if verbose:
                        all_disagreement_rates.append(disagreement_rate)

                    all_predictions.append(consensus_pred)

                    if return_details:
                        all_details.append({
                            'view_scores': view_scores,
                            'view_preds': view_preds,
                            'view_accepts': view_accepts,
                            'agree_count': accept_count,
                            'agree_k': agree_k,
                            'accept_ratio': accept_ratio,
                            'final_decision': (True, consensus_pred),
                            'reject_reason': None,
                            'disagreement_rate': disagreement_rate,
                            'score_stats': {
                                'median': np.median(view_scores),
                                'std': np.std(view_scores)
                            }
                        })
                else:
                    all_predictions.append(-1)
                    if return_details:
                        all_details.append({
                            'view_scores': view_scores,
                            'view_preds': view_preds,
                            'view_accepts': view_accepts,
                            'agree_count': accept_count,
                            'agree_k': agree_k,
                            'accept_ratio': accept_ratio,
                            'final_decision': (False, -1),
                            'reject_reason': 'no_valid_preds'
                        })
            else:
                all_predictions.append(-1)

                disagreement_rate = 1.0 if accept_count > 0 else 0.0
                if verbose and accept_count > 0:
                    all_disagreement_rates.append(disagreement_rate)

                if return_details:
                    all_details.append({
                        'view_scores': view_scores,
                        'view_preds': view_preds,
                        'view_accepts': view_accepts,
                        'agree_count': accept_count,
                        'agree_k': agree_k,
                        'accept_ratio': accept_ratio,
                        'final_decision': (False, -1),
                        'reject_reason': 'insufficient_agreement',
                        'disagreement_rate': disagreement_rate,
                        'score_stats': {
                            'median': np.median(view_scores),
                            'std': np.std(view_scores)
                        }
                    })

    if verbose and all_accept_ratios:
        print(f"TTA Statistics:")
        print(f"  Avg accept ratio: {np.mean(all_accept_ratios):.2%}")
        print(f"  Avg disagreement: {np.mean(all_disagreement_rates):.2%}")
        print(f"  Score spread: {np.mean([d['score_stats']['std'] for d in all_details if 'score_stats' in d]):.4f}")

    predictions = np.array(all_predictions)

    if return_details:
        return predictions, all_details
    else:
        return predictions


@torch.no_grad()
def extract_scores_genuine_tta(
    model, ncm, paths: List[str], labels: List[int],
    base_transform, device,
    n_views: int = 3,
    n_repeats: int = 1,
    aug_strength: float = 0.5,
    include_original: bool = True,
    view_aggregation: str = 'median',
    repeat_aggregation: str = 'median',
    img_size: int = 128,
    channels: int = 1,
    verbose: bool = False,
    seed: int = 42
) -> np.ndarray:
    """
    Extract genuine scores with TTA

    Multiple views and repeats for robust scoring
    """
    if not paths:
        return np.array([])

    set_seed(seed)

    model.eval()
    light_aug = get_light_augmentation(aug_strength, img_size)

    all_views = []
    sample_info = []

    for idx, path in enumerate(tqdm(paths, desc="Preparing TTA views", disable=not verbose)):
        try:
            img = _open_with_channels(path, channels)
        except Exception as e:
            print(f"WARNING: Failed to load {path}: {e}")
            continue

        label = labels[idx]

        for r_idx in range(n_repeats):
            for v_idx in range(n_views):
                if v_idx == 0 and include_original:
                    view = base_transform(img)
                else:
                    aug_img = light_aug(img) if aug_strength > 0 else img
                    view = base_transform(aug_img)

                all_views.append(view)
                sample_info.append((idx, r_idx, v_idx, label))

    if not all_views:
        return np.array([])

    # Process in batches
    try:
        batch_tensor = torch.stack(all_views).to(device)
        features = model.getFeatureCode(batch_tensor)
    except torch.cuda.OutOfMemoryError:
        print("WARNING: OOM detected, splitting batch...")
        features = []
        batch_size = 50
        for i in range(0, len(all_views), batch_size):
            batch = torch.stack(all_views[i:i+batch_size]).to(device)
            feat_batch = model.getFeatureCode(batch)
            features.append(feat_batch)
        features = torch.cat(features, dim=0)

    current_sample_idx = -1
    repeat_scores = []
    final_scores = []

    for i, (s_idx, r_idx, v_idx, label) in enumerate(sample_info):
        if s_idx != current_sample_idx:
            if current_sample_idx >= 0 and repeat_scores:
                if n_repeats > 1 and repeat_aggregation == 'median':
                    final_score = np.median(repeat_scores)
                elif n_repeats > 1:
                    final_score = np.mean(repeat_scores)
                else:
                    final_score = repeat_scores[0]
                final_scores.append(final_score)
                repeat_scores = []

            current_sample_idx = s_idx

        if r_idx == len(repeat_scores):
            view_scores = []

        ncm_scores = ncm.forward(features[i:i+1])

        if ncm_scores.numel() > 0:
            max_score = ncm_scores.max().item()
            view_scores.append(max_score)

        if v_idx == n_views - 1:
            if view_scores:
                if n_views > 1 and view_aggregation == 'median':
                    repeat_score = np.median(view_scores)
                else:
                    repeat_score = np.mean(view_scores)
                repeat_scores.append(repeat_score)

    # Handle last sample
    if repeat_scores:
        if n_repeats > 1 and repeat_aggregation == 'median':
            final_score = np.median(repeat_scores)
        elif n_repeats > 1:
            final_score = np.mean(repeat_scores)
        else:
            final_score = repeat_scores[0]
        final_scores.append(final_score)

    return np.array(final_scores)


@torch.no_grad()
def extract_scores_impostor_between_tta(
    model, ncm, paths: List[str], labels: List[int],
    base_transform, device,
    max_pairs: int = 2000,
    n_views: int = 3,
    n_repeats: int = 1,
    aug_strength: float = 0.5,
    include_original: bool = True,
    view_aggregation: str = 'median',
    repeat_aggregation: str = 'median',
    img_size: int = 128,
    channels: int = 1,
    verbose: bool = False,
    seed: int = 42
) -> np.ndarray:
    """
    Extract between-class impostor scores with TTA
    """
    if not paths:
        return np.array([])

    set_seed(seed)

    from collections import defaultdict
    by_class = defaultdict(list)
    for p, y in zip(paths, labels):
        by_class[int(y)].append((p, y))

    model.eval()
    light_aug = get_light_augmentation(aug_strength, img_size)

    all_views = []
    sample_info = []
    sample_idx = 0

    for cls_id, cls_items in by_class.items():
        if sample_idx >= max_pairs:
            break

        sample_items = cls_items[:min(10, len(cls_items))]
        if len(sample_items) > 5:
            import random
            sample_items = random.sample(sample_items, 5)

        for path, _ in sample_items:
            if sample_idx >= max_pairs:
                break

            try:
                img = _open_with_channels(path, channels)
            except Exception as e:
                print(f"WARNING: Failed to load {path}: {e}")
                continue

            for r_idx in range(n_repeats):
                for v_idx in range(n_views):
                    if v_idx == 0 and include_original:
                        view = base_transform(img)
                    else:
                        aug_img = light_aug(img) if aug_strength > 0 else img
                        view = base_transform(aug_img)

                    all_views.append(view)
                    sample_info.append((sample_idx, r_idx, v_idx, cls_id))

            sample_idx += 1

    if not all_views:
        return np.array([])

    # Process features
    try:
        batch_tensor = torch.stack(all_views).to(device)
        features = model.getFeatureCode(batch_tensor)
    except torch.cuda.OutOfMemoryError:
        print("WARNING: OOM detected, splitting batch...")
        features = []
        batch_size = 50
        for i in range(0, len(all_views), batch_size):
            batch = torch.stack(all_views[i:i+batch_size]).to(device)
            feat_batch = model.getFeatureCode(batch)
            features.append(feat_batch)
        features = torch.cat(features, dim=0)

    current_sample_idx = -1
    repeat_scores = []
    final_scores = []

    for i, (s_idx, r_idx, v_idx, cls_id) in enumerate(sample_info):
        if s_idx != current_sample_idx:
            if current_sample_idx >= 0 and repeat_scores:
                if n_repeats > 1 and repeat_aggregation == 'median':
                    final_score = np.median(repeat_scores)
                elif n_repeats > 1:
                    final_score = np.mean(repeat_scores)
                else:
                    final_score = repeat_scores[0]
                final_scores.append(final_score)
                repeat_scores = []

            current_sample_idx = s_idx

        if r_idx == len(repeat_scores):
            view_scores = []

        ncm_scores = ncm.forward(features[i:i+1])

        if ncm_scores.numel() > 0 and cls_id < ncm_scores.shape[1]:
            ncm_scores[0, cls_id] = -float('inf')
            max_score = ncm_scores.max().item()
            view_scores.append(max_score)

        if v_idx == n_views - 1:
            if view_scores:
                if n_views > 1 and view_aggregation == 'median':
                    repeat_score = np.median(view_scores)
                else:
                    repeat_score = np.mean(view_scores)
                repeat_scores.append(repeat_score)

    # Handle last sample
    if repeat_scores:
        if n_repeats > 1 and repeat_aggregation == 'median':
            final_score = np.median(repeat_scores)
        elif n_repeats > 1:
            final_score = np.mean(repeat_scores)
        else:
            final_score = repeat_scores[0]
        final_scores.append(final_score)

    return np.array(final_scores)


@torch.no_grad()
def extract_scores_impostor_negref_tta(
    model, ncm, negref_file: str,
    base_transform, device,
    max_eval: int = 5000,
    n_views: int = 3,
    n_repeats: int = 1,
    aug_strength: float = 0.5,
    include_original: bool = True,
    view_aggregation: str = 'median',
    repeat_aggregation: str = 'median',
    force_memory_save: bool = False,
    img_size: int = 128,
    channels: int = 1,
    verbose: bool = False,
    seed: int = 42
) -> np.ndarray:
    """
    Extract impostor scores from negative reference with TTA
    """
    if not negref_file or not os.path.exists(negref_file):
        return np.array([])

    paths, labels = load_paths_labels_from_txt(negref_file)

    if not paths:
        return np.array([])

    if len(paths) > max_eval:
        idx = np.random.choice(len(paths), max_eval, replace=False)
        paths = [paths[i] for i in idx]

    set_seed(seed)

    model.eval()
    final_scores = []
    base_seed = seed

    effective_repeats = n_repeats
    if force_memory_save and n_repeats > 1:
        effective_repeats = 1
        if verbose:
            print(f"NegRef: Memory save mode - forcing n_repeats=1 (was {n_repeats})")
    else:
        if verbose and n_repeats > 1:
            print(f"NegRef: Using {n_repeats} repeat(s) as configured")
            print(f"  WARNING: Note: High repeat counts may use significant memory")

    all_views = []
    sample_info = []

    light_aug = get_light_augmentation(aug_strength, img_size)

    for idx, path in enumerate(tqdm(paths[:max_eval],
                                    desc="Processing negref",
                                    disable=not verbose)):
        try:
            img = _open_with_channels(path, channels)
        except Exception as e:
            print(f"WARNING: Failed to load {path}: {e}")
            continue

        for r_idx in range(effective_repeats):
            if effective_repeats > 1:
                set_seed(base_seed + idx * 1000 + r_idx)

            for v_idx in range(n_views):
                if v_idx == 0 and include_original:
                    view = base_transform(img)
                else:
                    aug_img = light_aug(img) if aug_strength > 0 else img
                    view = base_transform(aug_img)

                all_views.append(view)
                sample_info.append((idx, r_idx, v_idx))

    if not all_views:
        return np.array([])

    # Process in batches with OOM handling
    batch_tensor = torch.stack(all_views)
    features_list = []

    try:
        batch_tensor = batch_tensor.to(device)
        features = model.getFeatureCode(batch_tensor)
        features_list.append(features.cpu())
    except torch.cuda.OutOfMemoryError:
        print(f"WARNING: OOM at batch size {len(batch_tensor)}, splitting...")
        for i in range(0, len(batch_tensor), 32):
            mini_batch = batch_tensor[i:i+32].to(device)
            feat = model.getFeatureCode(mini_batch)
            features_list.append(feat.cpu())
            del feat, mini_batch
            torch.cuda.empty_cache()

        del batch_tensor

    features = torch.cat(features_list, dim=0)

    current_sample_idx = -1
    repeat_scores = []
    view_scores = []

    for i, (s_idx, r_idx, v_idx) in enumerate(sample_info):
        if s_idx != current_sample_idx:
            if current_sample_idx >= 0 and repeat_scores:
                if effective_repeats > 1 and repeat_aggregation == 'median':
                    final_score = np.median(repeat_scores)
                elif effective_repeats > 1:
                    final_score = np.mean(repeat_scores)
                else:
                    final_score = repeat_scores[0]
                final_scores.append(final_score)
                repeat_scores = []

            current_sample_idx = s_idx

        feat_tensor = features[i:i+1].to(device)
        ncm_scores = ncm.forward(feat_tensor)

        if ncm_scores.numel() > 0:
            max_score = ncm_scores.max().item()
            view_scores.append(max_score)

        if v_idx == n_views - 1:
            if view_scores:
                if n_views > 1 and view_aggregation == 'median':
                    repeat_score = np.median(view_scores)
                else:
                    repeat_score = np.mean(view_scores)
                repeat_scores.append(repeat_score)
                view_scores = []

    # Handle last sample
    if repeat_scores:
        if effective_repeats > 1 and repeat_aggregation == 'median':
            final_score = np.median(repeat_scores)
        elif effective_repeats > 1:
            final_score = np.mean(repeat_scores)
        else:
            final_score = repeat_scores[0]

        final_scores.append(final_score)

    del features
    torch.cuda.empty_cache()

    if verbose:
        print(f"NegRef processed: {len(final_scores)} samples")
        if effective_repeats > 1:
            print(f"  with {effective_repeats} repeats per sample")

    return np.array(final_scores[:max_eval])