"""
Phase -1.0c: PCA leakage impact diagnostic.

Compares two PCA-initialized projections of the proj_512 winner:
  A) PCA fit on enroll_file only  (~450 samples, rank ≤ 449)
  B) PCA fit on enroll+unknown_dev (~900 samples, current production)

For a fixed evaluation set (`enroll_file` probes + a held-out probe split),
report whether the two projections are functionally equivalent:

Metrics (paired per probe):
  - cosine_alignment(proj_A(h), proj_B(h)) — mean, p5, p50
  - rank_top1_agreement on genuine NCM classification
  - score distribution comparison
  - threshold and realized FPIR delta (if checkpoint includes calibration)

Decision (GPT v7):
  top1_agreement ≥ 0.95 AND |TAR Δ| < 0.5pp AND |τ Δ| < 0.02
  AND |realized FPIR Δ| < 0.5pp AND |overlap_gap Δ| < 0.02
  → enroll-only safe as main → no re-measurement needed

Otherwise → enroll-only main requires re-running 35u sweep + 50u verification.

This script does NOT modify the production checkpoint.
It builds two projection heads in-memory, applies each to extracted features,
and compares downstream behavior.

Usage:
    python scripts/phase_minus_1/pca_leakage_diagnostic.py \
        --config configs/proj_512_legacy_50u.yaml \
        --proj-dim 512 \
        --output /content/drive/MyDrive/phase_minus_1/pca_leakage.json
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

from coconut.models import ccnet, PretrainedLoader, ProjectionHead  # noqa: E402
from coconut.openset.score_extraction import extract_features  # noqa: E402
from coconut.openset.utils import load_paths_labels_from_txt  # noqa: E402
from coconut.data.transforms import get_scr_transforms  # noqa: E402
from config.config_parser import ConfigParser  # noqa: E402


def fit_pca_projection(
    features: np.ndarray,
    out_dim: int,
    in_dim: int = 2048,
) -> ProjectionHead:
    """Build a ProjectionHead and initialize via PCA on given features."""
    head = ProjectionHead(in_dim=in_dim, out_dim=out_dim)
    feats_t = torch.from_numpy(features).float()
    info = head.init_with_pca(feats_t)
    print(f"  PCA init: usable_components={info['usable_components']}, "
          f"retained_variance={info['retained_variance_ratio']:.4f}")
    return head


@torch.no_grad()
def project(head: ProjectionHead, features: np.ndarray) -> np.ndarray:
    """Apply head to features, return projected (N, out_dim) array."""
    x = torch.from_numpy(features).float()
    y = head(x).numpy()
    return y


def per_class_means_l2(features: np.ndarray, labels: List[int]) -> Tuple[np.ndarray, List[int]]:
    """Compute L2-normalized per-class means (NCM-style)."""
    by_class: Dict[int, list] = {}
    for f, y in zip(features, labels):
        by_class.setdefault(int(y), []).append(f)
    classes = sorted(by_class.keys())
    means = []
    for c in classes:
        m = np.mean(by_class[c], axis=0)
        m = m / (np.linalg.norm(m) + 1e-12)
        means.append(m)
    return np.stack(means, axis=0), classes


def compare_projections(
    feats_eval: np.ndarray,
    labels_eval: List[int],
    head_A: ProjectionHead,
    head_B: ProjectionHead,
    feats_enroll: np.ndarray,
    labels_enroll: List[int],
) -> Dict:
    """Compute alignment + top-1 agreement on eval set under both projections."""
    # Project all features
    proj_A_eval = project(head_A, feats_eval)
    proj_B_eval = project(head_B, feats_eval)
    proj_A_enroll = project(head_A, feats_enroll)
    proj_B_enroll = project(head_B, feats_enroll)

    # L2-normalize for cosine
    nA_eval = proj_A_eval / (np.linalg.norm(proj_A_eval, axis=1, keepdims=True) + 1e-12)
    nB_eval = proj_B_eval / (np.linalg.norm(proj_B_eval, axis=1, keepdims=True) + 1e-12)
    nA_enroll = proj_A_enroll / (np.linalg.norm(proj_A_enroll, axis=1, keepdims=True) + 1e-12)
    nB_enroll = proj_B_enroll / (np.linalg.norm(proj_B_enroll, axis=1, keepdims=True) + 1e-12)

    # Cosine alignment per probe
    cos_align = np.sum(nA_eval * nB_eval, axis=1)

    # NCM prototypes from enrollment under each projection
    means_A, classes_A = per_class_means_l2(proj_A_enroll, labels_enroll)
    means_B, classes_B = per_class_means_l2(proj_B_enroll, labels_enroll)

    # Verify class ordering matches
    assert classes_A == classes_B, "class orders differ between PCA fits"

    # Cosine scores eval × classes
    scores_A = nA_eval @ means_A.T  # (N_eval, C)
    scores_B = nB_eval @ means_B.T

    pred_A = scores_A.argmax(axis=1)
    pred_B = scores_B.argmax(axis=1)
    top1_agreement = float(np.mean(pred_A == pred_B))

    # Top-1 score per probe (closed-set "genuine" if pred == label)
    label_idx_map = {c: i for i, c in enumerate(classes_A)}
    eval_class_idx = np.array([label_idx_map.get(int(y), -1) for y in labels_eval])
    in_gallery_mask = eval_class_idx >= 0

    top1_scores_A = scores_A.max(axis=1)
    top1_scores_B = scores_B.max(axis=1)

    # Per-class genuine scores
    if in_gallery_mask.any():
        gen_scores_A = scores_A[np.arange(len(eval_class_idx)), eval_class_idx]
        gen_scores_B = scores_B[np.arange(len(eval_class_idx)), eval_class_idx]
        gen_mask = in_gallery_mask
        # Closed-set rank-1 correct
        rank1_A = (pred_A == eval_class_idx) & gen_mask
        rank1_B = (pred_B == eval_class_idx) & gen_mask
    else:
        gen_scores_A = np.array([])
        gen_scores_B = np.array([])
        rank1_A = np.array([], dtype=bool)
        rank1_B = np.array([], dtype=bool)

    return {
        'cosine_alignment_mean': float(cos_align.mean()),
        'cosine_alignment_p5': float(np.percentile(cos_align, 5)),
        'cosine_alignment_p50': float(np.median(cos_align)),
        'top1_agreement': top1_agreement,
        'rank1_correct_A': float(rank1_A.sum() / gen_mask.sum()) if gen_mask.any() else None,
        'rank1_correct_B': float(rank1_B.sum() / gen_mask.sum()) if gen_mask.any() else None,
        'top1_score_mean_A': float(top1_scores_A.mean()),
        'top1_score_mean_B': float(top1_scores_B.mean()),
        'top1_score_delta_mean': float((top1_scores_A - top1_scores_B).mean()),
        'top1_score_delta_p5': float(np.percentile(top1_scores_A - top1_scores_B, 5)),
        'top1_score_delta_p95': float(np.percentile(top1_scores_A - top1_scores_B, 95)),
        'genuine_score_delta_mean': float(
            (gen_scores_A - gen_scores_B).mean()) if gen_scores_A.size > 0 else None,
        'n_eval_total': int(len(labels_eval)),
        'n_eval_in_gallery': int(gen_mask.sum()) if gen_mask.size else 0,
        'n_classes': int(len(classes_A)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--proj-dim', type=int, default=512)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-eval', type=int, default=None,
                        help='Cap eval samples (debug only)')
    args = parser.parse_args()

    config = ConfigParser(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[1.0c] device: {device}")

    # Load pretrained CCNet (raw 2048-D features)
    model = ccnet(
        num_classes=getattr(config.model, 'num_classes', 600),
        weight=config.model.competition_weight,
    ).to(device)
    if config.model.use_pretrained:
        loader = PretrainedLoader(model)
        loader.load(str(config.model.pretrained_path), strict=False)
    model.eval()

    transform = get_scr_transforms(
        train=False,
        imside=config.dataset.height,
        channels=config.dataset.channels,
    )

    # Extract features
    print(f"[1.0c] extracting enroll features...")
    enroll_paths, enroll_labels = load_paths_labels_from_txt(str(config.dataset.enroll_file))
    enroll_feats = extract_features(
        model, enroll_paths, transform, device,
        batch_size=64, channels=config.dataset.channels,
    )
    enroll_feats = np.asarray(enroll_feats)
    print(f"  enroll: {enroll_feats.shape}, classes={len(set(enroll_labels))}")

    print(f"[1.0c] extracting unknown_dev features...")
    dev_paths, _ = load_paths_labels_from_txt(str(config.dataset.unknown_dev_file))
    dev_feats = extract_features(
        model, dev_paths, transform, device,
        batch_size=64, channels=config.dataset.channels,
    )
    dev_feats = np.asarray(dev_feats)
    print(f"  unknown_dev: {dev_feats.shape}")

    # Build two PCA-init projection heads
    print(f"[1.0c] PCA-A (enroll-only)...")
    head_A = fit_pca_projection(enroll_feats, out_dim=args.proj_dim)
    print(f"[1.0c] PCA-B (enroll + unknown_dev)...")
    combined = np.concatenate([enroll_feats, dev_feats], axis=0)
    head_B = fit_pca_projection(combined, out_dim=args.proj_dim)

    # Eval set: use eval_probe_file (held-out per-user probe)
    print(f"[1.0c] extracting eval probe features: {config.dataset.eval_probe_file}")
    eval_paths, eval_labels = load_paths_labels_from_txt(str(config.dataset.eval_probe_file))
    if args.max_eval is not None and len(eval_paths) > args.max_eval:
        eval_paths = eval_paths[:args.max_eval]
        eval_labels = eval_labels[:args.max_eval]
    eval_feats = extract_features(
        model, eval_paths, transform, device,
        batch_size=64, channels=config.dataset.channels,
    )
    eval_feats = np.asarray(eval_feats)
    print(f"  eval: {eval_feats.shape}")

    # Compare projections
    print(f"[1.0c] computing projection comparison...")
    metrics = compare_projections(
        eval_feats, eval_labels,
        head_A, head_B,
        enroll_feats, enroll_labels,
    )

    # Print summary
    print("\n=== PCA Leakage Comparison ===")
    for k, v in metrics.items():
        if v is None:
            print(f"  {k}: N/A")
        elif isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Verdict (GPT v7 thresholds)
    print("\n=== Verdict ===")
    top1_ok = metrics['top1_agreement'] >= 0.95
    cos_ok = metrics['cosine_alignment_p5'] >= 0.90  # 90th percentile
    score_delta_ok = abs(metrics['top1_score_delta_mean']) < 0.02
    print(f"  top1_agreement >= 0.95? {top1_ok} ({metrics['top1_agreement']:.4f})")
    print(f"  cosine_alignment_p5 >= 0.90? {cos_ok} ({metrics['cosine_alignment_p5']:.4f})")
    print(f"  |top1_score_delta_mean| < 0.02? {score_delta_ok} ({metrics['top1_score_delta_mean']:.4f})")
    if top1_ok and score_delta_ok:
        print("  → enroll-only PCA likely SAFE for main result")
    elif top1_ok:
        print("  → marginal: top-1 stable but score-scale shifted")
    else:
        print("  → enroll-only diverges: re-measurement (35u/50u) may be needed")
    print("\n  NOTE: full verdict also requires τ Δ + realized FPIR Δ + overlap_gap Δ,")
    print("        which need a forward pass through the threshold calibrator (not in this script).")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output = {
        'config': args.config,
        'proj_dim': args.proj_dim,
        'metrics': metrics,
        'verdict': {
            'top1_agreement_ok': top1_ok,
            'cosine_alignment_ok': cos_ok,
            'score_delta_ok': score_delta_ok,
            'threshold_delta_checked': False,  # not in this script
        }
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\n[1.0c] saved: {args.output}")


if __name__ == '__main__':
    main()
