"""
Phase -1.2: Proxy-NCM alignment diagnostic (C-GPAC-2 verification).

Measures how well-aligned the ProxyAnchor trainable proxies are with the
NCM class-mean prototypes used at inference time. If they diverge, then
GPAC training signal on proxies does not directly translate to inference
improvement.

Metrics (GPT v7):
  - proxy_ncm_alignment_mean = mean_u cos(p_u, mu_u^NCM)
  - proxy_ncm_alignment_p5
  - proxy_ncm_top1_agreement = P(argmax cos(h, p) == argmax cos(h, mu_NCM))
                              on genuine probe samples
  - boundary_disagreement_rate = P(disagree | margin_p1_p2 < threshold)

Verdict (full GPAC requires both):
  top1_agreement >= 0.90 AND mean >= 0.85
    -> proxy-based GPAC viable
  top1_agreement 0.80-0.90
    -> auxiliary GPAC only
  top1_agreement < 0.80
    -> proxy GPAC must be discarded, switch to NCM-Gaussian

Usage:
    python scripts/phase_minus_1/proxy_ncm_alignment.py \
        --config configs/proj_512_legacy_50u.yaml \
        --checkpoint /content/drive/MyDrive/coconut_proj512_50u/coconut_results/checkpoint_exp_50.pth \
        --output /content/drive/MyDrive/phase_minus_1/proxy_ncm_alignment.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from coconut.models import ccnet, ProjectionHead  # noqa: E402
from coconut.openset.score_extraction import extract_features  # noqa: E402
from coconut.openset.utils import load_paths_labels_from_txt  # noqa: E402
from coconut.data.transforms import get_scr_transforms  # noqa: E402
from config.config_parser import ConfigParser  # noqa: E402


def load_checkpoint_components(ckpt_path: str, device: torch.device):
    """Load the trained CCNet + ProjectionHead + proxies + NCM means."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # CCNet backbone
    print(f"  loading CCNet (state_dict keys: {len(ckpt['model_state_dict'])})")
    model = ccnet(weight=0.8)  # weight overridden by state dict load
    # Use strict=False to tolerate any tiny mismatches
    missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if missing:
        print(f"  CCNet missing keys: {len(missing)} (showing first 3: {missing[:3]})")
    if unexpected:
        print(f"  CCNet unexpected keys: {len(unexpected)}")
    model = model.to(device).eval()

    # ProjectionHead
    meta = ckpt['projection_meta']
    print(f"  loading projection: {meta['in_dim']} -> {meta['out_dim']}")
    projection = ProjectionHead(in_dim=meta['in_dim'], out_dim=meta['out_dim'])
    projection.load_state_dict(ckpt['projection_state_dict'])
    projection = projection.to(device).eval()

    # Proxies (raw, not L2-normalized)
    pa = ckpt['proxy_anchor_data']
    proxies = pa['proxies'].to(device)
    class_to_idx = pa['class_to_idx']
    num_classes = pa['num_classes']
    print(f"  proxies: shape={tuple(proxies.shape)}, num_classes={num_classes}")

    # NCM class means (already L2-normalized in the buffer)
    ncm_sd = ckpt['ncm_state_dict']
    class_means = ncm_sd['class_means'].to(device)  # (max_class+1, D)
    print(f"  NCM class_means: shape={tuple(class_means.shape)}")

    return model, projection, proxies, class_to_idx, num_classes, class_means


def project_through_pipeline(
    raw_features: np.ndarray,
    projection: ProjectionHead,
    device: torch.device,
) -> torch.Tensor:
    """Apply projection to raw 2048-D features. Returns L2-normalized 512-D tensor."""
    x = torch.from_numpy(raw_features).float().to(device)
    with torch.no_grad():
        h = projection(x)  # (N, D_out)
        h = F.normalize(h, p=2, dim=1)
    return h


def compute_alignment_metrics(
    proxies: torch.Tensor,            # (C, D), raw
    class_to_idx: Dict[int, int],     # user_id -> proxy_idx
    class_means: torch.Tensor,        # (max_class+1, D), L2-normalized
    eval_features: torch.Tensor,      # (N, D), L2-normalized after projection
    eval_labels: List[int],
) -> Dict:
    """Core comparison: how well do ProxyAnchor proxies align with NCM prototypes?"""
    # 1) Per-user alignment cos(p_u, mu_u^NCM)
    # Both must be L2-normalized for cosine
    proxies_n = F.normalize(proxies, p=2, dim=1)  # ProxyAnchor normalizes in forward

    alignments = {}
    for user_id, proxy_idx in class_to_idx.items():
        if user_id >= class_means.shape[0]:
            continue
        mu_u = class_means[user_id]  # already L2-normalized
        if torch.allclose(mu_u, torch.zeros_like(mu_u), atol=1e-6):
            continue  # unused NCM slot
        p_u = proxies_n[proxy_idx]
        alignments[user_id] = float((p_u * mu_u).sum().item())

    align_vals = np.array(list(alignments.values()))
    align_stats = {
        'n_users': len(alignments),
        'mean': float(align_vals.mean()),
        'p5': float(np.percentile(align_vals, 5)),
        'p25': float(np.percentile(align_vals, 25)),
        'median': float(np.median(align_vals)),
        'p75': float(np.percentile(align_vals, 75)),
        'p95': float(np.percentile(align_vals, 95)),
        'min': float(align_vals.min()),
        'max': float(align_vals.max()),
    }

    # 2) Top-1 agreement on eval probes
    # We need both proxy-based and NCM-based predictions in the SAME user_id namespace.
    # Build ordered (user_id, idx) maps.
    proxy_user_ids = sorted(class_to_idx.keys())
    proxy_indices = [class_to_idx[u] for u in proxy_user_ids]
    proxy_mat = proxies_n[proxy_indices]  # (C, D) reordered by user_id ascending

    ncm_user_ids = [
        i for i in range(class_means.shape[0])
        if not torch.allclose(class_means[i], torch.zeros_like(class_means[i]), atol=1e-6)
    ]
    ncm_mat = class_means[ncm_user_ids]  # (C', D)

    # Use intersection of registered users
    common_users = sorted(set(proxy_user_ids) & set(ncm_user_ids))
    proxy_idx_map = {u: i for i, u in enumerate(proxy_user_ids)}
    ncm_idx_map = {u: i for i, u in enumerate(ncm_user_ids)}

    common_proxy_rows = torch.stack([proxy_mat[proxy_idx_map[u]] for u in common_users])
    common_ncm_rows = torch.stack([ncm_mat[ncm_idx_map[u]] for u in common_users])

    # Cosine scores: (N, C_common)
    scores_proxy = eval_features @ common_proxy_rows.T
    scores_ncm = eval_features @ common_ncm_rows.T

    pred_proxy = scores_proxy.argmax(dim=1).cpu().numpy()
    pred_ncm = scores_ncm.argmax(dim=1).cpu().numpy()

    # Common users vector for label mapping
    common_users_arr = np.array(common_users)
    label_to_common_idx = {u: i for i, u in enumerate(common_users)}

    # Eval labels in common-user index space
    eval_in_gallery = np.array([
        label_to_common_idx[y] if y in label_to_common_idx else -1
        for y in eval_labels
    ])
    in_gallery_mask = eval_in_gallery >= 0

    top1_agreement = float((pred_proxy == pred_ncm).mean())
    top1_agreement_gallery = (
        float((pred_proxy[in_gallery_mask] == pred_ncm[in_gallery_mask]).mean())
        if in_gallery_mask.any() else None
    )

    # 3) Boundary disagreement: low-margin probes (NCM-based margin, since inference uses NCM)
    scores_ncm_sorted, _ = torch.sort(scores_ncm, dim=1, descending=True)
    margin_p1_p2 = (scores_ncm_sorted[:, 0] - scores_ncm_sorted[:, 1]).cpu().numpy()
    median_margin = float(np.median(margin_p1_p2))
    low_margin_thresh = 0.5 * median_margin
    low_margin_mask = margin_p1_p2 < low_margin_thresh

    if low_margin_mask.any():
        boundary_disagreement = float(
            (pred_proxy[low_margin_mask] != pred_ncm[low_margin_mask]).mean()
        )
        boundary_count = int(low_margin_mask.sum())
    else:
        boundary_disagreement = None
        boundary_count = 0

    # 4) Per-user correctness comparison (closed-set rank-1)
    if in_gallery_mask.any():
        gallery_idx_arr = eval_in_gallery[in_gallery_mask]
        rank1_proxy = float(
            (pred_proxy[in_gallery_mask] == gallery_idx_arr).mean()
        )
        rank1_ncm = float(
            (pred_ncm[in_gallery_mask] == gallery_idx_arr).mean()
        )
    else:
        rank1_proxy = rank1_ncm = None

    return {
        'alignment_stats': align_stats,
        'per_user_alignment': {int(k): float(v) for k, v in alignments.items()},
        'top1_agreement_all': top1_agreement,
        'top1_agreement_gallery': top1_agreement_gallery,
        'boundary_disagreement_rate': boundary_disagreement,
        'boundary_count': boundary_count,
        'median_margin': median_margin,
        'low_margin_threshold': low_margin_thresh,
        'rank1_correct_proxy': rank1_proxy,
        'rank1_correct_ncm': rank1_ncm,
        'n_eval_total': int(len(eval_labels)),
        'n_eval_in_gallery': int(in_gallery_mask.sum()),
        'n_common_users': len(common_users),
    }


def render_verdict(metrics: Dict) -> Dict:
    """Apply GPT v7 verdict rules."""
    mean_align = metrics['alignment_stats']['mean']
    p5_align = metrics['alignment_stats']['p5']
    top1_agree = metrics['top1_agreement_gallery'] or metrics['top1_agreement_all']
    boundary = metrics['boundary_disagreement_rate']

    if top1_agree >= 0.90 and mean_align >= 0.85:
        action = "FULL_GPAC_VIABLE"
        rationale = "top1_agreement >= 0.90 AND mean alignment >= 0.85"
    elif top1_agree >= 0.80 and mean_align >= 0.70:
        action = "AUXILIARY_GPAC_ONLY"
        rationale = "top1_agreement 0.80-0.90 — proxy partially aligned"
    elif top1_agree < 0.80:
        action = "PROXY_GPAC_DISCARD"
        rationale = "top1_agreement < 0.80 — proxy and NCM diverge significantly; switch to NCM-Gaussian"
    else:
        action = "MARGINAL_USE_CONSERVATIVE_DEFAULT"
        rationale = "borderline values; default to proj_512_legacy"

    return {
        'action': action,
        'rationale': rationale,
        'thresholds_applied': {
            'top1_agreement >= 0.90': top1_agree >= 0.90,
            'mean_alignment >= 0.85': mean_align >= 0.85,
            'p5_alignment >= 0.65': p5_align >= 0.65,
            'boundary_disagreement < 0.20': (
                boundary is not None and boundary < 0.20
            ),
        },
        'measured': {
            'top1_agreement': top1_agree,
            'mean_alignment': mean_align,
            'p5_alignment': p5_align,
            'boundary_disagreement': boundary,
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-eval', type=int, default=None,
                        help='Cap eval probes (debug)')
    args = parser.parse_args()

    config = ConfigParser(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[1.2] device: {device}")
    print(f"[1.2] checkpoint: {args.checkpoint}")

    # Load all components
    model, projection, proxies, class_to_idx, num_classes, class_means = \
        load_checkpoint_components(args.checkpoint, device)

    print(f"[1.2] loaded: {num_classes} proxies, "
          f"{(class_means.norm(dim=1) > 1e-6).sum().item()} non-zero NCM means")

    # Extract eval probe features (raw 2048-D, then project)
    transform = get_scr_transforms(
        train=False,
        imside=config.dataset.height,
        channels=config.dataset.channels,
    )
    print(f"[1.2] extracting eval probe features...")
    eval_paths, eval_labels = load_paths_labels_from_txt(str(config.dataset.eval_probe_file))
    if args.max_eval is not None and len(eval_paths) > args.max_eval:
        eval_paths = eval_paths[:args.max_eval]
        eval_labels = eval_labels[:args.max_eval]
    raw_feats = extract_features(
        model, eval_paths, transform, device,
        batch_size=64, channels=config.dataset.channels,
    )
    raw_feats = np.asarray(raw_feats)
    print(f"  raw: {raw_feats.shape}")

    # Project to 512-D and L2-normalize
    eval_feats_proj = project_through_pipeline(raw_feats, projection, device)
    print(f"  projected+normalized: {tuple(eval_feats_proj.shape)}")

    # Compute alignment metrics
    metrics = compute_alignment_metrics(
        proxies, class_to_idx, class_means, eval_feats_proj, eval_labels,
    )

    # Verdict
    verdict = render_verdict(metrics)

    # Print summary
    print("\n=== Proxy-NCM Alignment Summary ===")
    a = metrics['alignment_stats']
    print(f"  per-user cos(p_u, mu_u^NCM):")
    print(f"    mean={a['mean']:.4f}  p5={a['p5']:.4f}  median={a['median']:.4f}  p95={a['p95']:.4f}")
    print(f"    min={a['min']:.4f}  max={a['max']:.4f}  n_users={a['n_users']}")
    print(f"  top1_agreement (all probes):     {metrics['top1_agreement_all']:.4f}")
    if metrics['top1_agreement_gallery'] is not None:
        print(f"  top1_agreement (gallery only):    {metrics['top1_agreement_gallery']:.4f}")
    if metrics['boundary_disagreement_rate'] is not None:
        print(f"  boundary_disagreement_rate:      {metrics['boundary_disagreement_rate']:.4f}")
        print(f"    (margin < {metrics['low_margin_threshold']:.4f}, n={metrics['boundary_count']})")
    print(f"  rank1_correct_proxy: {metrics['rank1_correct_proxy']:.4f}")
    print(f"  rank1_correct_ncm:   {metrics['rank1_correct_ncm']:.4f}")

    print(f"\n=== Verdict ===")
    print(f"  ACTION: {verdict['action']}")
    print(f"  rationale: {verdict['rationale']}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({
            'config': args.config,
            'checkpoint': args.checkpoint,
            'metrics': metrics,
            'verdict': verdict,
        }, f, indent=2, default=float)
    print(f"\n[1.2] saved: {args.output}")


if __name__ == '__main__':
    main()
