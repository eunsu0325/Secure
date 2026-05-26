"""
Phase -1.0b: PCA Explained Variance Ratio (EVR) diagnostic.

Measures the explained variance ratio for two PCA fitting strategies:
  - enroll-only: PCA fit on enroll_file features (~450 samples)
  - enroll+dev: PCA fit on enroll_file + unknown_dev_file (~900 samples)

For each strategy, reports EVR at dim=128/256/512 (or up to rank limit),
plus pooled statistics (participation_ratio, effective_rank).

Critical: enroll-only PCA has rank limit ≤ N-1, so EVR@512 is undefined
when N≈450 (trailing 63 singular values are zero). The script reports
EVR@min(512, N-1) and flags the rank limit explicitly.

This is a read-only diagnostic — no checkpoint modification, no training.

Usage (Colab):
    python scripts/phase_minus_1/pca_evr_diagnostic.py \
        --config configs/proj_512_legacy_50u.yaml \
        --output /content/drive/MyDrive/phase_minus_1/pca_evr.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from coconut.models import ccnet, PretrainedLoader  # noqa: E402
from coconut.openset.score_extraction import extract_features  # noqa: E402
from coconut.openset.utils import load_paths_labels_from_txt  # noqa: E402
from coconut.data.transforms import get_scr_transforms  # noqa: E402
from config.config_parser import ConfigParser  # noqa: E402


def collect_features(
    model: torch.nn.Module,
    paths_file: str,
    transform,
    device: torch.device,
    channels: int,
    max_n: int | None = None,
) -> Tuple[np.ndarray, List[int]]:
    """Extract raw 2048-D CCNet features for the given file."""
    paths, labels = load_paths_labels_from_txt(paths_file)
    if max_n is not None and len(paths) > max_n:
        paths = paths[:max_n]
        labels = labels[:max_n]
    feats = extract_features(
        model, paths, transform, device,
        batch_size=64, channels=channels,
    )
    if feats is None:
        raise RuntimeError(f"Feature extraction returned None for {paths_file}")
    return np.asarray(feats), labels


def compute_pca_stats(features: np.ndarray, dims_of_interest=(128, 256, 512)) -> Dict:
    """Compute PCA on features and return EVR + spectrum statistics.

    Args:
        features: (N, D) raw feature matrix.
        dims_of_interest: dims at which to report cumulative EVR.

    Returns:
        Dict with keys:
            n_samples, feature_dim, rank_limit
            singular_values_full (list), eigenvalues_full (list)
            evr_at_dim: dict {dim: cumulative_evr or None if dim > rank_limit}
            participation_ratio, effective_rank
    """
    n_samples, feature_dim = features.shape
    rank_limit = min(n_samples - 1, feature_dim)

    centered = features - features.mean(axis=0, keepdims=True)
    # Truncated SVD (full_matrices=False yields min(N, D) singular values)
    _, S, _ = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = S ** 2  # variance per component
    total_var = eigenvalues.sum()

    evr_at_dim: Dict[int, float | None] = {}
    for d in dims_of_interest:
        if d > rank_limit:
            evr_at_dim[d] = None  # undefined — trailing dims are degenerate
        else:
            evr_at_dim[d] = float(eigenvalues[:d].sum() / total_var)

    # Participation ratio: effective number of components
    pr = float((eigenvalues.sum()) ** 2 / (eigenvalues ** 2).sum())

    # Effective rank via entropy of normalized spectrum
    p = eigenvalues / total_var
    p = p[p > 1e-15]
    entropy = -(p * np.log(p)).sum()
    eff_rank = float(np.exp(entropy))

    return {
        'n_samples': int(n_samples),
        'feature_dim': int(feature_dim),
        'rank_limit': int(rank_limit),
        'singular_values_top10': S[:10].tolist(),
        'eigenvalues_top10': eigenvalues[:10].tolist(),
        'evr_at_dim': evr_at_dim,
        'evr_full_cumulative': np.cumsum(eigenvalues / total_var).tolist(),
        'participation_ratio': pr,
        'effective_rank': eff_rank,
        'total_variance': float(total_var),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to YAML config')
    parser.add_argument('--output', required=True, help='Output JSON path')
    parser.add_argument('--max-enroll', type=int, default=None,
                        help='Cap enroll samples (debug only)')
    parser.add_argument('--max-dev', type=int, default=None,
                        help='Cap unknown_dev samples (debug only)')
    args = parser.parse_args()

    config = ConfigParser(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[1.0b] device: {device}")

    # Load pretrained CCNet (no projection — we want raw 2048-D features)
    model = ccnet(
        num_classes=getattr(config.model, 'num_classes', 600),
        weight=config.model.competition_weight,
    ).to(device)
    if config.model.use_pretrained:
        loader = PretrainedLoader(model)
        loader.load(str(config.model.pretrained_path), strict=False)
        print(f"[1.0b] loaded pretrained: {config.model.pretrained_path}")
    model.eval()

    transform = get_scr_transforms(
        train=False,
        imside=config.dataset.height,
        channels=config.dataset.channels,
    )

    # 1) enroll-only PCA
    print(f"[1.0b] extracting enroll features: {config.dataset.enroll_file}")
    enroll_feats, _ = collect_features(
        model, str(config.dataset.enroll_file), transform, device,
        channels=config.dataset.channels, max_n=args.max_enroll,
    )
    print(f"  enroll shape: {enroll_feats.shape}")

    # 2) enroll + unknown_dev PCA
    dev_feats = None
    if config.dataset.unknown_dev_file:
        print(f"[1.0b] extracting unknown_dev features: {config.dataset.unknown_dev_file}")
        dev_feats, _ = collect_features(
            model, str(config.dataset.unknown_dev_file), transform, device,
            channels=config.dataset.channels, max_n=args.max_dev,
        )
        print(f"  unknown_dev shape: {dev_feats.shape}")

    results = {
        'config': args.config,
        'enroll_only': compute_pca_stats(enroll_feats),
    }
    if dev_feats is not None:
        combined = np.concatenate([enroll_feats, dev_feats], axis=0)
        print(f"  combined shape: {combined.shape}")
        results['enroll_plus_dev'] = compute_pca_stats(combined)

    # Summary print
    print("\n=== EVR Summary ===")
    for label, stats in results.items():
        if label == 'config':
            continue
        print(f"\n[{label}]  N={stats['n_samples']}, rank_limit={stats['rank_limit']}")
        for d, evr in stats['evr_at_dim'].items():
            if evr is None:
                print(f"  EVR@{d}: undefined (d > rank_limit={stats['rank_limit']})")
            else:
                print(f"  EVR@{d}: {evr:.4f}")
        print(f"  participation_ratio: {stats['participation_ratio']:.2f}")
        print(f"  effective_rank:      {stats['effective_rank']:.2f}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n[1.0b] saved: {args.output}")


if __name__ == '__main__':
    main()
