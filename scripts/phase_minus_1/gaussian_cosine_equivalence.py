"""
Phase -1.3: Gaussian-cosine equivalence diagnostic (C-GPAC-1 verification).

Tests whether a Gaussian (Mahalanobis) distance from a probe to per-user
proxies/prototypes produces different RANKING than plain cosine distance.

If Gaussian distance is rank-equivalent to cosine, then GPAC training
loss based on Gaussian distance adds no new information — it is just
ProxyAnchor cosine in a different scale.

Critical insight (GPT v5): the equivalence may hold *globally* (high
mean_rho) but break at boundary samples (low_margin) — and boundary is
exactly where GPAC would help. So we report both global and low-margin
subset rho.

Mathematical setup:
  On unit sphere (||h||=||p||=1):
    ||h - p||^2 = 2(1 - cos(h, p))
  If sigma^2 is shared across users/dims:
    D_u(h) = (1/D) sum_d (h_d - p_{u,d})^2 / sigma^2
           = (2/(D*sigma^2)) * (1 - cos(h, p_u))
  -> perfect rank equivalence with cosine.

If sigma^2 varies meaningfully (per-user or per-dim), the equivalence
breaks and GPAC has information to add.

Variance models tested:
  (a) shared scalar  sigma^2 = pooled variance
  (b) shared per-dim sigma^2_d
  (c) per-user shrinkage diagonal: rho * shared + (1-rho) * user-specific

Metrics:
  - mean_rho_all       = mean over probes Spearman(D_G, D_cos)
  - mean_rho_low_margin
  - mean_rho_silent     (placeholder; needs probe-level rank info)
  - user_var_CV  (per-user mean variance variability)
  - dim_var_CV
  - pooled residual eigenspectrum (anisotropy)

Verdict:
  Strong duplicate:   mean_rho >= 0.985 AND var CV very low  -> GPAC loss USELESS
  Useful signal:      mean_rho_low_margin < 0.90              -> GPAC viable
  Marginal:           between                                  -> auxiliary only

Usage:
    python scripts/phase_minus_1/gaussian_cosine_equivalence.py \
        --config configs/proj_512_legacy_50u.yaml \
        --checkpoint /content/drive/MyDrive/coconut_proj512_50u/coconut_results/checkpoint_exp_50.pth \
        --output /content/drive/MyDrive/phase_minus_1/gaussian_cosine_equivalence.json
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
    """Load CCNet, ProjectionHead, proxies, NCM means."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = ccnet(weight=0.8)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(device).eval()

    meta = ckpt['projection_meta']
    projection = ProjectionHead(in_dim=meta['in_dim'], out_dim=meta['out_dim'])
    projection.load_state_dict(ckpt['projection_state_dict'])
    projection = projection.to(device).eval()

    pa = ckpt['proxy_anchor_data']
    proxies = pa['proxies'].to(device)
    class_to_idx = pa['class_to_idx']

    ncm_sd = ckpt['ncm_state_dict']
    class_means = ncm_sd['class_means'].to(device)

    return model, projection, proxies, class_to_idx, class_means


@torch.no_grad()
def project_eval(raw_feats: np.ndarray, projection: ProjectionHead, device: torch.device):
    """Apply projection, return (raw_proj, normalized_proj) as torch tensors on device."""
    x = torch.from_numpy(raw_feats).float().to(device)
    h_raw = projection(x)
    h_norm = F.normalize(h_raw, p=2, dim=1)
    return h_raw, h_norm


def estimate_variance_models(
    enroll_proj: torch.Tensor,    # (N_enroll, D) projected (not normalized)
    enroll_labels: List[int],
    proxies_normalized: torch.Tensor,  # (C, D)
    class_to_idx: Dict[int, int],
):
    """Fit three sigma^2 variance models from enrollment features.

    Returns:
      sigma_shared_scalar: scalar tensor
      sigma_shared_per_dim: (D,) tensor
      per_user_sigma: dict {user_id: (D,) tensor}, shrinkage-adjusted
      anisotropy_stats: dict with eigenspectrum stats
    """
    D = enroll_proj.shape[1]

    # Per-user means (computed in projected space; will compare with proxies separately)
    by_class: Dict[int, list] = {}
    for h, y in zip(enroll_proj, enroll_labels):
        by_class.setdefault(int(y), []).append(h)

    # Residuals e = h - mu_u, pooled across all users
    user_residuals = {}
    pooled_residuals = []
    user_var_norms = []  # ||sigma^2_u||_1 per user (for CV)

    for user_id, samples in by_class.items():
        s = torch.stack(samples, dim=0)  # (n_u, D)
        mu_u = s.mean(dim=0, keepdim=True)
        eps = s - mu_u  # (n_u, D)
        user_residuals[user_id] = eps
        pooled_residuals.append(eps)
        if eps.shape[0] >= 2:
            var_u = eps.var(dim=0, unbiased=True)  # (D,)
            user_var_norms.append(float(var_u.mean().item()))

    pooled = torch.cat(pooled_residuals, dim=0)  # (N_total, D)

    # (a) shared scalar
    sigma_shared_scalar = float(pooled.var(dim=0, unbiased=True).mean().item())

    # (b) shared per-dim
    sigma_shared_per_dim = pooled.var(dim=0, unbiased=True)  # (D,)

    # (c) per-user shrinkage: rho = n_0 / (n_0 + n_u), with n_0 = 20
    n_0 = 20
    var_floor = 1e-6
    per_user_sigma = {}
    for user_id, eps in user_residuals.items():
        n_u = eps.shape[0]
        rho = n_0 / (n_0 + n_u)
        if n_u >= 2:
            var_u = eps.var(dim=0, unbiased=True)  # (D,)
        else:
            var_u = sigma_shared_per_dim
        # shrunk
        var_shrunk = (1 - rho) * var_u + rho * sigma_shared_per_dim
        per_user_sigma[user_id] = var_shrunk.clamp_min(var_floor)

    # Variance CV
    var_arr = np.array(user_var_norms)
    user_var_CV = float(var_arr.std() / (var_arr.mean() + 1e-12)) if len(var_arr) > 0 else None

    dim_var_arr = sigma_shared_per_dim.cpu().numpy()
    dim_var_CV = float(dim_var_arr.std() / (dim_var_arr.mean() + 1e-12))

    # Anisotropy from pooled covariance eigenspectrum
    # NOTE: rank is limited by sample size, treat with caution
    pooled_centered = pooled - pooled.mean(dim=0, keepdim=True)
    cov = (pooled_centered.T @ pooled_centered) / max(pooled.shape[0] - 1, 1)
    try:
        eigvals = torch.linalg.eigvalsh(cov).cpu().numpy()
        eigvals = np.sort(eigvals)[::-1]  # descending
        eigvals_pos = eigvals[eigvals > 1e-10]
        anisotropy = float(eigvals_pos[0] / eigvals_pos.mean()) if len(eigvals_pos) > 0 else None
        condition = float(eigvals_pos[0] / eigvals_pos[-1]) if len(eigvals_pos) > 0 else None
        eff_rank = float(np.exp(
            -np.sum((eigvals_pos / eigvals_pos.sum()) * np.log(eigvals_pos / eigvals_pos.sum()))
        ))
    except Exception as e:
        print(f"  [warn] eigval failed: {e}")
        anisotropy = condition = eff_rank = None

    return {
        'sigma_shared_scalar': sigma_shared_scalar,
        'sigma_shared_per_dim': sigma_shared_per_dim,
        'per_user_sigma': per_user_sigma,
        'stats': {
            'user_var_CV': user_var_CV,
            'dim_var_CV': dim_var_CV,
            'mean_per_user_var': float(np.mean(user_var_norms)) if user_var_norms else None,
            'pooled_residual_anisotropy': anisotropy,
            'pooled_residual_eff_rank': eff_rank,
            'pooled_residual_condition': condition,
            'n_pooled_residuals': int(pooled.shape[0]),
            'eigvals_top10': eigvals[:10].tolist() if 'eigvals' in dir() else [],
        }
    }


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation between two 1-D arrays. NaN-safe."""
    if len(x) < 2:
        return float('nan')
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    # Pearson on ranks
    rx_c = rx - rx.mean()
    ry_c = ry - ry.mean()
    denom = np.sqrt((rx_c ** 2).sum() * (ry_c ** 2).sum())
    if denom < 1e-12:
        return float('nan')
    return float((rx_c * ry_c).sum() / denom)


def compute_rho_per_probe(
    eval_proj: torch.Tensor,       # (N, D) projected, not normalized
    eval_proj_normalized: torch.Tensor,  # (N, D) L2-normalized
    proxy_normalized: torch.Tensor,      # (C, D) L2-normalized
    proxy_raw: torch.Tensor,             # (C, D) raw, used as Gaussian means
    proxy_user_ids: List[int],
    per_user_sigma: Dict[int, torch.Tensor],
    sigma_shared_per_dim: torch.Tensor,
) -> Dict:
    """Per probe, compute Spearman correlation between Gaussian distance and cosine distance."""
    N = eval_proj.shape[0]
    C = len(proxy_user_ids)

    # Cosine distance = 1 - cos
    cos_sim = eval_proj_normalized @ proxy_normalized.T  # (N, C)
    cos_dist = 1.0 - cos_sim  # smaller = more similar

    # Gaussian distance (per-dim shrunk variance)
    # D_u(h) = (1/D) sum_d (h_d - mu_{u,d})^2 / sigma_{u,d}^2
    D_dim = eval_proj.shape[1]

    # Build (C, D) sigma matrix in user-order
    sigma_mat = torch.stack(
        [per_user_sigma[u] for u in proxy_user_ids], dim=0
    )  # (C, D)
    eps_var = 1e-8

    # diff: (N, C, D)
    # Be careful with memory: N*C*D = 661*148*512 ~ 50M floats = 200MB OK
    diff = eval_proj.unsqueeze(1) - proxy_raw.unsqueeze(0)  # (N, C, D)
    gauss_dist = ((diff ** 2) / (sigma_mat.unsqueeze(0) + eps_var)).mean(dim=2)  # (N, C)

    cos_dist_np = cos_dist.cpu().numpy()
    gauss_dist_np = gauss_dist.cpu().numpy()

    # Per probe Spearman rho
    rho_per_probe = np.array([
        spearman_rho(gauss_dist_np[i], cos_dist_np[i]) for i in range(N)
    ])

    # Argmin gives "predicted user index"
    pred_cos = cos_dist_np.argmin(axis=1)
    pred_gauss = gauss_dist_np.argmin(axis=1)
    top1_agreement_gauss_vs_cos = float((pred_cos == pred_gauss).mean())

    # Margin (cosine, since NCM/ProxyAnchor use cosine for ranking)
    sorted_cos_sim = np.sort(cos_sim.cpu().numpy(), axis=1)[:, ::-1]
    margin_p1_p2 = sorted_cos_sim[:, 0] - sorted_cos_sim[:, 1]
    median_margin = float(np.median(margin_p1_p2))
    low_margin_thresh = 0.5 * median_margin
    low_margin_mask = margin_p1_p2 < low_margin_thresh

    return {
        'mean_rho_all': float(np.nanmean(rho_per_probe)),
        'mean_rho_p5': float(np.nanpercentile(rho_per_probe, 5)),
        'mean_rho_p95': float(np.nanpercentile(rho_per_probe, 95)),
        'mean_rho_low_margin': (
            float(np.nanmean(rho_per_probe[low_margin_mask]))
            if low_margin_mask.any() else None
        ),
        'low_margin_count': int(low_margin_mask.sum()),
        'median_margin': median_margin,
        'low_margin_threshold': low_margin_thresh,
        'top1_agreement_gauss_vs_cos': top1_agreement_gauss_vs_cos,
        'n_probes': int(N),
    }


def render_verdict(rho_stats: Dict, variance_stats: Dict) -> Dict:
    """3-tier decision (GPT v5)."""
    rho_all = rho_stats['mean_rho_all']
    rho_lm = rho_stats['mean_rho_low_margin']
    user_var_CV = variance_stats.get('user_var_CV', 0)
    dim_var_CV = variance_stats.get('dim_var_CV', 0)

    if rho_all >= 0.985 and (user_var_CV or 0) < 0.05 and (dim_var_CV or 0) < 0.10:
        action = "STRONG_DUPLICATE_DISCARD_GPAC_LOSS"
        rationale = (
            f"rho_all={rho_all:.4f} >= 0.985, variance CVs very low. "
            "Gaussian distance is functionally identical to cosine. "
            "GPAC training loss would be ProxyAnchor in different scale."
        )
    elif rho_lm is not None and rho_lm < 0.90:
        action = "USEFUL_SIGNAL_GPAC_VIABLE"
        rationale = (
            f"rho_low_margin={rho_lm:.4f} < 0.90 — at boundary samples "
            "(exactly where GPAC would help), Gaussian distance disagrees "
            "with cosine. GPAC has information to add."
        )
    elif rho_all >= 0.97:
        action = "LIKELY_DUPLICATE_AUXILIARY_ONLY"
        rationale = (
            f"rho_all={rho_all:.4f} high; variance CVs moderate. "
            "GPAC loss redundant; scheduler/diagnostic uses only."
        )
    else:
        action = "MARGINAL_USE_CONSERVATIVE_DEFAULT"
        rationale = f"rho_all={rho_all:.4f}, borderline; default to proj_512_legacy."

    return {'action': action, 'rationale': rationale}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-enroll', type=int, default=None)
    parser.add_argument('--max-eval', type=int, default=None)
    args = parser.parse_args()

    config = ConfigParser(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[1.3] device: {device}")

    model, projection, proxies, class_to_idx, class_means = \
        load_checkpoint_components(args.checkpoint, device)

    transform = get_scr_transforms(
        train=False,
        imside=config.dataset.height,
        channels=config.dataset.channels,
    )

    # Enrollment features (for variance estimation)
    print(f"[1.3] extracting enrollment features for variance estimation...")
    enroll_paths, enroll_labels = load_paths_labels_from_txt(str(config.dataset.enroll_file))
    if args.max_enroll is not None and len(enroll_paths) > args.max_enroll:
        enroll_paths = enroll_paths[:args.max_enroll]
        enroll_labels = enroll_labels[:args.max_enroll]
    enroll_raw = extract_features(
        model, enroll_paths, transform, device,
        batch_size=64, channels=config.dataset.channels,
    )
    enroll_proj, _ = project_eval(np.asarray(enroll_raw), projection, device)
    print(f"  enroll projected: {tuple(enroll_proj.shape)}")

    # Filter enroll to only users in proxy gallery
    valid_mask = [int(y) in class_to_idx for y in enroll_labels]
    enroll_proj_filtered = enroll_proj[torch.tensor(valid_mask, dtype=torch.bool)]
    enroll_labels_filtered = [y for y, m in zip(enroll_labels, valid_mask) if m]
    print(f"  enroll in gallery: {enroll_proj_filtered.shape[0]} samples, "
          f"{len(set(enroll_labels_filtered))} users")

    # Build proxy structures in user-id ordered way
    proxy_user_ids = sorted(class_to_idx.keys())
    proxy_indices = [class_to_idx[u] for u in proxy_user_ids]
    proxies_raw = proxies[proxy_indices]  # (C, D)
    proxies_normalized = F.normalize(proxies_raw, p=2, dim=1)

    # Variance estimation
    print(f"[1.3] fitting variance models (n0=20 shrinkage)...")
    var_models = estimate_variance_models(
        enroll_proj_filtered, enroll_labels_filtered,
        proxies_normalized, class_to_idx,
    )

    # Eval features
    print(f"[1.3] extracting eval probe features...")
    eval_paths, eval_labels = load_paths_labels_from_txt(str(config.dataset.eval_probe_file))
    if args.max_eval is not None and len(eval_paths) > args.max_eval:
        eval_paths = eval_paths[:args.max_eval]
        eval_labels = eval_labels[:args.max_eval]
    eval_raw = extract_features(
        model, eval_paths, transform, device,
        batch_size=64, channels=config.dataset.channels,
    )
    eval_proj, eval_proj_normalized = project_eval(np.asarray(eval_raw), projection, device)
    print(f"  eval projected: {tuple(eval_proj.shape)}")

    # Make sure all proxy users have a variance
    for u in proxy_user_ids:
        if u not in var_models['per_user_sigma']:
            print(f"  [warn] user {u} has no enrollment samples; using shared variance")
            var_models['per_user_sigma'][u] = var_models['sigma_shared_per_dim']

    # Compute rho per probe
    print(f"[1.3] computing Spearman rho per probe...")
    rho_stats = compute_rho_per_probe(
        eval_proj, eval_proj_normalized,
        proxies_normalized, proxies_raw, proxy_user_ids,
        var_models['per_user_sigma'], var_models['sigma_shared_per_dim'],
    )

    verdict = render_verdict(rho_stats, var_models['stats'])

    # Print summary
    print("\n=== Variance Stats ===")
    s = var_models['stats']
    print(f"  user_var_CV: {s['user_var_CV']}")
    print(f"  dim_var_CV:  {s['dim_var_CV']}")
    print(f"  mean per-user variance: {s['mean_per_user_var']}")
    print(f"  pooled residual eff_rank: {s['pooled_residual_eff_rank']}")
    print(f"  pooled residual anisotropy: {s['pooled_residual_anisotropy']}")
    print(f"  pooled residual condition: {s['pooled_residual_condition']}")

    print("\n=== Spearman Rho Stats ===")
    print(f"  mean_rho_all:        {rho_stats['mean_rho_all']:.4f}")
    print(f"  mean_rho_p5:         {rho_stats['mean_rho_p5']:.4f}")
    print(f"  mean_rho_p95:        {rho_stats['mean_rho_p95']:.4f}")
    print(f"  mean_rho_low_margin: {rho_stats['mean_rho_low_margin']}")
    print(f"    (margin < {rho_stats['low_margin_threshold']:.4f}, n={rho_stats['low_margin_count']})")
    print(f"  top1_agreement_gauss_vs_cos: {rho_stats['top1_agreement_gauss_vs_cos']:.4f}")

    print(f"\n=== Verdict ===")
    print(f"  ACTION: {verdict['action']}")
    print(f"  rationale: {verdict['rationale']}")

    # Save (drop tensor objects)
    save_obj = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'variance_stats': var_models['stats'],
        'rho_stats': rho_stats,
        'verdict': verdict,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(save_obj, f, indent=2, default=float)
    print(f"\n[1.3] saved: {args.output}")


if __name__ == '__main__':
    main()
