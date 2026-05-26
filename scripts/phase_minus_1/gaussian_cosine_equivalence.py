"""
Phase -1.3 (v2 — REWRITE): Gaussian-cosine equivalence diagnostic (C-GPAC-1).

PRIOR-VERSION DESIGN ERRORS (acknowledged and fixed here):
  (E1) Used raw proxy as Gaussian center, but 1.2 measured cos(proxy, NCM_mean)
       = 0.27 — proxy and NCM mean are DIFFERENT objects on the sphere.
       Inference uses NCM mean → Gaussian center MUST also be NCM mean for
       fair cosine-vs-Gaussian comparison.
  (E2) Estimated variance in raw projected space (norm ≈ 15), while every
       comparison (NCM cosine, ProxyAnchor cosine, S-norm) operates on the
       unit sphere. Variance MUST be estimated in L2-normalized space.
  (E3) var_floor = 1e-8 was sphere-incompatible. On the unit sphere, the
       per-dim typical variance is ~1/D = 1/512 ≈ 2e-3. A floor below 1e-5
       lets tiny noise dimensions dominate Mahalanobis distance.
  (E4) shrinkage n_0 = 20 gave ρ = 20/29 ≈ 0.69 for 9-sample users. With
       only 9 samples per user and 512 dims, per-user variance is
       fundamentally under-determined. Raise to n_0 = 50 (ρ ≈ 0.85).

What this diagnostic answers:
  When we replace cosine(h, μ_NCM) with a per-class Gaussian (Mahalanobis)
  distance to μ_NCM, does the RANKING change?

  If ranking is identical (high mean Spearman ρ, perfect top-1 agreement),
  then GPAC's training loss is ProxyAnchor in a different scale — adds
  nothing. If ranking diverges, especially on low-margin (boundary)
  probes, then Gaussian distance carries information cosine does not.

Math on the unit sphere (||h|| = ||μ|| = 1):
    ||h - μ||² = 2(1 - cos(h, μ))
    With shared scalar σ²:
        D_u(h) = ||h - μ_u||² / σ² = (2/σ²)(1 - cos)  ← rank-equivalent
    With per-dim or per-user σ²:
        D_u(h) = (1/D) Σ_d (h_d - μ_{u,d})² / σ²_{u,d}  ← can diverge

Variance models tested (all in L2-normalized space):
  (a) shared scalar σ²
  (b) shared per-dim σ²_d
  (c) per-user diagonal σ²_{u,d} with shrinkage to (b)

Metrics:
  - mean_rho_all          : Spearman(D_G, D_cos) averaged over probes
  - mean_rho_low_margin   : same, restricted to low-margin probes
                            (where GPAC would matter most)
  - top1_agreement_gauss_vs_cos
  - user_var_CV, dim_var_CV
  - pooled residual eigenspectrum (anisotropy, eff_rank)

Decision (3-tier, GPT v5):
  Strong duplicate:  mean_rho_all ≥ 0.985 AND user_var_CV < 0.05 AND
                     dim_var_CV < 0.10 AND top1_agreement ≥ 0.99
                     → GPAC loss USELESS
  Useful signal:     mean_rho_low_margin < 0.90 OR top1_agreement < 0.95
                     → GPAC viable
  Marginal / likely duplicate: between

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


# ── Design constants (sphere-appropriate, sample-adapted) ──────────────────
# var_floor is now ADAPTIVE: floor = FLOOR_REL × σ²_shared. v2 used absolute
# 1e-3 which was 12.8× the actual σ²_shared (7.8e-5), forcing all per-dim
# variances onto the floor and degenerating Gaussian → scaled-cosine. v3
# sweeps multiple relative scales to expose the true floor-dependence curve.
FLOOR_REL_SWEEP = [0.01, 0.1, 1.0, 10.0]   # 1%, 10%, 100%, 1000% of σ²_shared
FLOOR_ABS_MIN = 1e-12                       # numerical underflow guard
SHRINKAGE_N0 = 50                           # ρ = n_0/(n_0+n_u) = 50/59 ≈ 0.85 at n_u=9


def load_checkpoint_components(ckpt_path: str, device: torch.device):
    """Load CCNet + ProjectionHead + NCM means (NCM means already L2-normalized)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = ccnet(weight=0.8)
    missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if missing:
        print(f"  CCNet missing: {len(missing)}")
    if unexpected:
        print(f"  CCNet unexpected: {len(unexpected)}")
    model = model.to(device).eval()

    meta = ckpt['projection_meta']
    projection = ProjectionHead(in_dim=meta['in_dim'], out_dim=meta['out_dim'])
    projection.load_state_dict(ckpt['projection_state_dict'])
    projection = projection.to(device).eval()

    # NCM class means: already L2-normalized (ncm.py line 144-148)
    ncm_sd = ckpt['ncm_state_dict']
    class_means = ncm_sd['class_means'].to(device)  # (max_class+1, D)
    norms = class_means.norm(dim=1)
    n_valid = int((norms > 1e-6).sum().item())
    print(f"  NCM class_means: shape={tuple(class_means.shape)}, valid={n_valid}")

    return model, projection, class_means


@torch.no_grad()
def project_and_normalize(
    raw_feats: np.ndarray,
    projection: ProjectionHead,
    device: torch.device,
) -> torch.Tensor:
    """Apply projection then L2-normalize. Returns (N, D) on device."""
    x = torch.from_numpy(raw_feats).float().to(device)
    h = projection(x)
    return F.normalize(h, p=2, dim=1)


def estimate_variance_models(
    enroll_normalized: torch.Tensor,    # (N_enroll, D) ALREADY L2-normalized
    enroll_labels: List[int],
    class_means_normalized: Dict[int, torch.Tensor],  # user_id -> (D,) NCM mean
):
    """Fit three σ² variance models in the L2-normalized space.

    Residuals are computed against the NCM mean (Fix E1), in normalized
    space (Fix E2). Shrinkage uses n_0 = SHRINKAGE_N0 (Fix E4), floor =
    VAR_FLOOR (Fix E3).
    """
    D = enroll_normalized.shape[1]

    # Bucket enrollment samples by user
    by_class: Dict[int, list] = {}
    for h, y in zip(enroll_normalized, enroll_labels):
        by_class.setdefault(int(y), []).append(h)

    # Residuals e = h - μ_NCM (μ is the fixed NCM mean, NOT per-batch mean)
    user_residuals: Dict[int, torch.Tensor] = {}
    pooled_residuals = []
    user_mean_var_list = []  # mean(σ²_u) per user, for CV
    user_sample_counts = []

    for user_id, samples in by_class.items():
        if user_id not in class_means_normalized:
            continue
        s = torch.stack(samples, dim=0)            # (n_u, D)
        mu_u = class_means_normalized[user_id]     # (D,)  — already normalized
        eps = s - mu_u.unsqueeze(0)                # (n_u, D)
        user_residuals[user_id] = eps
        pooled_residuals.append(eps)
        if eps.shape[0] >= 2:
            var_u = eps.var(dim=0, unbiased=True)  # (D,)
            user_mean_var_list.append(float(var_u.mean().item()))
        user_sample_counts.append(int(eps.shape[0]))

    if not pooled_residuals:
        raise RuntimeError("No enrollment users matched class_means.")

    pooled = torch.cat(pooled_residuals, dim=0)    # (N_total, D)

    # (a) shared scalar σ² — base for adaptive floor
    sigma_shared_scalar = float(pooled.var(dim=0, unbiased=True).mean().item())

    # (b) shared per-dim σ²_d — NO floor here; floor applied per-sweep later
    sigma_shared_per_dim = pooled.var(dim=0, unbiased=True)

    # (c) per-user shrinkage diagonal — NO floor here either
    per_user_sigma_raw: Dict[int, torch.Tensor] = {}
    for user_id, eps in user_residuals.items():
        n_u = eps.shape[0]
        rho = SHRINKAGE_N0 / (SHRINKAGE_N0 + n_u)  # shrinkage weight toward shared
        if n_u >= 2:
            var_u = eps.var(dim=0, unbiased=True)
        else:
            var_u = sigma_shared_per_dim
        var_shrunk = (1.0 - rho) * var_u + rho * sigma_shared_per_dim
        per_user_sigma_raw[user_id] = var_shrunk
    per_user_sigma = per_user_sigma_raw  # alias preserved for downstream code

    # Variance CV (per-user variability of mean variance)
    user_var_arr = np.array(user_mean_var_list)
    user_var_CV = (
        float(user_var_arr.std() / (user_var_arr.mean() + 1e-12))
        if len(user_var_arr) > 0 else None
    )

    # dim-wise CV
    dim_var_np = sigma_shared_per_dim.cpu().numpy()
    dim_var_CV = float(dim_var_np.std() / (dim_var_np.mean() + 1e-12))

    # Anisotropy via pooled residual eigenspectrum
    pooled_centered = pooled - pooled.mean(dim=0, keepdim=True)
    cov = (pooled_centered.T @ pooled_centered) / max(pooled.shape[0] - 1, 1)
    try:
        eigvals = torch.linalg.eigvalsh(cov).cpu().numpy()
        eigvals = np.sort(eigvals)[::-1]  # descending
        eigvals_pos = eigvals[eigvals > 1e-12]
        if len(eigvals_pos) > 0:
            anisotropy = float(eigvals_pos[0] / eigvals_pos.mean())
            condition = float(eigvals_pos[0] / eigvals_pos[-1])
            p = eigvals_pos / eigvals_pos.sum()
            eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-30))))
            top10 = eigvals[:10].tolist()
        else:
            anisotropy = condition = eff_rank = None
            top10 = []
    except Exception as e:
        print(f"  [warn] eigval failed: {e}")
        anisotropy = condition = eff_rank = None
        top10 = []

    return {
        'sigma_shared_scalar': sigma_shared_scalar,
        'sigma_shared_per_dim': sigma_shared_per_dim,
        'per_user_sigma': per_user_sigma,
        'stats': {
            'space': 'L2_normalized',
            'floor_strategy': 'adaptive_sweep',
            'floor_rel_sweep': FLOOR_REL_SWEEP,
            'shrinkage_n0': SHRINKAGE_N0,
            'n_users_with_var': len(user_residuals),
            'mean_samples_per_user': float(np.mean(user_sample_counts))
            if user_sample_counts else None,
            'min_samples_per_user': int(min(user_sample_counts))
            if user_sample_counts else None,
            'mean_per_user_var': float(np.mean(user_mean_var_list))
            if user_mean_var_list else None,
            'sigma_shared_scalar': sigma_shared_scalar,
            'user_var_CV': user_var_CV,
            'dim_var_CV': dim_var_CV,
            'pooled_residual_anisotropy': anisotropy,
            'pooled_residual_eff_rank': eff_rank,
            'pooled_residual_condition': condition,
            'pooled_eigvals_top10': top10,
            'n_pooled_residuals': int(pooled.shape[0]),
        }
    }


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """NaN-safe Spearman rank correlation."""
    if len(x) < 2:
        return float('nan')
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx_c = rx - rx.mean()
    ry_c = ry - ry.mean()
    denom = np.sqrt((rx_c ** 2).sum() * (ry_c ** 2).sum())
    if denom < 1e-12:
        return float('nan')
    return float((rx_c * ry_c).sum() / denom)


def compute_rho_per_probe(
    eval_normalized: torch.Tensor,           # (N, D) L2-normalized
    class_means_mat: torch.Tensor,           # (C, D) L2-normalized, gallery-ordered
    gallery_user_ids: List[int],             # user_id for each row of class_means_mat
    per_user_sigma: Dict[int, torch.Tensor],
    sigma_shared_per_dim: torch.Tensor,
    var_floor: float,                        # applied via clamp_min on sigma_mat
    precomputed: Dict = None,                # optional cached diff_sq, cos_sim
) -> Dict:
    """Per probe: Spearman(D_Gaussian, D_cosine) across the gallery.

    var_floor is applied as clamp_min on the sigma matrix. precomputed lets the
    caller pass cos_sim / diff_sq / margin info that don't depend on floor,
    avoiding redundant work across a sweep.
    """
    N, D = eval_normalized.shape
    C = class_means_mat.shape[0]

    if precomputed is None:
        # Cosine distance (unit-sphere): both are L2-normalized
        cos_sim = eval_normalized @ class_means_mat.T          # (N, C)
        diff = eval_normalized.unsqueeze(1) - class_means_mat.unsqueeze(0)
        diff_sq = diff ** 2                                    # (N, C, D)
    else:
        cos_sim = precomputed['cos_sim']
        diff_sq = precomputed['diff_sq']
    cos_dist = 1.0 - cos_sim

    # Per-user σ² matrix in gallery order, then floor
    sigma_rows = []
    for u in gallery_user_ids:
        if u in per_user_sigma:
            sigma_rows.append(per_user_sigma[u])
        else:
            sigma_rows.append(sigma_shared_per_dim)
    sigma_mat = torch.stack(sigma_rows, dim=0).clamp_min(var_floor)  # (C, D)

    # Floor activity diagnostic: what fraction of (user,dim) entries hit floor
    floor_active_frac = float(
        (sigma_mat <= var_floor * 1.0001).float().mean().item()
    )

    # D_u(h) = (1/D) Σ_d (h_d - μ_{u,d})² / σ²_{u,d}
    gauss_dist = (diff_sq / sigma_mat.unsqueeze(0)).mean(dim=2)       # (N, C)

    cos_dist_np = cos_dist.cpu().numpy()
    gauss_dist_np = gauss_dist.cpu().numpy()

    rho_per_probe = np.array([
        spearman_rho(gauss_dist_np[i], cos_dist_np[i]) for i in range(N)
    ])

    # Top-1 agreement (Gaussian-rank-1 == Cosine-rank-1?)
    pred_cos = cos_dist_np.argmin(axis=1)
    pred_gauss = gauss_dist_np.argmin(axis=1)
    top1_agreement = float((pred_cos == pred_gauss).mean())

    # Margin in cosine space (inference uses cosine for ranking)
    sorted_cos_sim = np.sort(cos_sim.cpu().numpy(), axis=1)[:, ::-1]
    margin_p1_p2 = sorted_cos_sim[:, 0] - sorted_cos_sim[:, 1]
    median_margin = float(np.median(margin_p1_p2))
    low_margin_thresh = 0.5 * median_margin
    low_margin_mask = margin_p1_p2 < low_margin_thresh

    if low_margin_mask.any():
        rho_low = rho_per_probe[low_margin_mask]
        mean_rho_low_margin = float(np.nanmean(rho_low))
        # Also: top1 agreement on low-margin subset
        top1_agree_low = float(
            (pred_cos[low_margin_mask] == pred_gauss[low_margin_mask]).mean()
        )
    else:
        mean_rho_low_margin = None
        top1_agree_low = None

    return {
        'var_floor': var_floor,
        'floor_active_frac': floor_active_frac,
        'mean_rho_all': float(np.nanmean(rho_per_probe)),
        'median_rho': float(np.nanmedian(rho_per_probe)),
        'std_rho': float(np.nanstd(rho_per_probe)),
        'mean_rho_p5': float(np.nanpercentile(rho_per_probe, 5)),
        'mean_rho_p95': float(np.nanpercentile(rho_per_probe, 95)),
        'mean_rho_low_margin': mean_rho_low_margin,
        'low_margin_count': int(low_margin_mask.sum()),
        'median_margin': median_margin,
        'low_margin_threshold': low_margin_thresh,
        'top1_agreement_gauss_vs_cos': top1_agreement,
        'top1_agreement_low_margin': top1_agree_low,
        'n_probes': int(N),
    }


def render_verdict_for_floor(rho_stats: Dict, variance_stats: Dict) -> Dict:
    """3-tier decision (GPT v5), applied to a single floor."""
    rho_all = rho_stats['mean_rho_all']
    rho_lm = rho_stats['mean_rho_low_margin']
    top1 = rho_stats['top1_agreement_gauss_vs_cos']
    user_var_CV = variance_stats.get('user_var_CV') or 0
    dim_var_CV = variance_stats.get('dim_var_CV') or 0

    is_strong_duplicate = (
        rho_all >= 0.985
        and user_var_CV < 0.05
        and top1 >= 0.99
    )
    has_boundary_signal = (rho_lm is not None and rho_lm < 0.90) or top1 < 0.95

    if is_strong_duplicate:
        action = "STRONG_DUPLICATE_DISCARD_GPAC_LOSS"
    elif has_boundary_signal:
        action = "USEFUL_SIGNAL_GPAC_VIABLE"
    elif rho_all >= 0.97:
        action = "LIKELY_DUPLICATE_AUXILIARY_ONLY"
    else:
        action = "MARGINAL_USE_CONSERVATIVE_DEFAULT"

    return {
        'action': action,
        'rho_all': rho_all,
        'rho_low_margin': rho_lm,
        'top1_agreement': top1,
        'top1_low_margin': rho_stats.get('top1_agreement_low_margin'),
    }


def render_sweep_verdict(sweep_results: List[Dict]) -> Dict:
    """Aggregate verdict across the floor sweep.

    Key question: as floor → 0 (true anisotropic Gaussian emerges), do ρ and
    top1_agreement remain near 1.0, or do they drop? If they drop, Gaussian
    carries information cosine lacks.
    """
    # Sort by floor ascending so we read "tighter → looser"
    sweep_sorted = sorted(sweep_results, key=lambda r: r['var_floor'])
    smallest_floor = sweep_sorted[0]
    largest_floor = sweep_sorted[-1]

    rho_min = min(r['rho_all'] for r in sweep_sorted)
    rho_max = max(r['rho_all'] for r in sweep_sorted)
    top1_min = min(r['top1_agreement'] for r in sweep_sorted)
    top1_max = max(r['top1_agreement'] for r in sweep_sorted)
    rho_lm_vals = [r['rho_low_margin'] for r in sweep_sorted
                   if r['rho_low_margin'] is not None]
    rho_lm_min = min(rho_lm_vals) if rho_lm_vals else None

    delta_rho = rho_max - rho_min
    delta_top1 = top1_max - top1_min

    # Strong duplicate across the entire floor range -> GPAC loss has no math basis
    if rho_min >= 0.985 and top1_min >= 0.99:
        action = "STRONG_DUPLICATE_DISCARD_GPAC_LOSS"
        rationale = (
            f"Across all floor scales {[r['var_floor'] for r in sweep_sorted]}, "
            f"rho stayed in [{rho_min:.4f}, {rho_max:.4f}] and top1 in "
            f"[{top1_min:.4f}, {top1_max:.4f}]. Gaussian = cosine even when "
            f"per-dim anisotropy is fully exposed. GPAC training loss adds nothing."
        )
    elif rho_lm_min is not None and rho_lm_min < 0.90:
        action = "USEFUL_SIGNAL_GPAC_VIABLE"
        rationale = (
            f"At some floor, rho_low_margin dropped to {rho_lm_min:.4f} < 0.90. "
            "Gaussian distance disagrees with cosine on boundary samples — "
            "exactly where GPAC would help."
        )
    elif delta_rho > 0.02 or delta_top1 > 0.02 or top1_min < 0.95:
        action = "FLOOR_DEPENDENT_SIGNAL_GPAC_AUXILIARY"
        rationale = (
            f"Δrho={delta_rho:.4f}, Δtop1={delta_top1:.4f} across floor sweep. "
            "Anisotropy carries some information but only at low floor; "
            "auxiliary GPAC usage justified, full loss replacement is not."
        )
    else:
        action = "LIKELY_DUPLICATE_AUXILIARY_ONLY"
        rationale = (
            f"rho∈[{rho_min:.4f}, {rho_max:.4f}], top1∈[{top1_min:.4f}, {top1_max:.4f}]; "
            "narrow window with no boundary signal. GPAC loss redundant."
        )

    return {
        'action': action,
        'rationale': rationale,
        'sweep_summary': {
            'rho_min': rho_min,
            'rho_max': rho_max,
            'top1_min': top1_min,
            'top1_max': top1_max,
            'rho_low_margin_min': rho_lm_min,
            'delta_rho': delta_rho,
            'delta_top1': delta_top1,
            'smallest_floor': smallest_floor,
            'largest_floor': largest_floor,
        }
    }


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
    print(f"[1.3v3] device: {device}")
    print(f"[1.3v3] design fixes: NCM-center, normalized-space, "
          f"floor_sweep={FLOOR_REL_SWEEP}, n_0={SHRINKAGE_N0}")

    model, projection, class_means = load_checkpoint_components(
        args.checkpoint, device
    )

    # Build gallery: user_ids whose NCM mean is non-zero
    norms = class_means.norm(dim=1)
    gallery_user_ids = sorted([
        int(i) for i in range(class_means.shape[0])
        if float(norms[i].item()) > 1e-6
    ])
    class_means_dict = {u: class_means[u] for u in gallery_user_ids}
    class_means_mat = torch.stack([class_means_dict[u] for u in gallery_user_ids], dim=0)
    print(f"[1.3v3] gallery users: {len(gallery_user_ids)}")

    transform = get_scr_transforms(
        train=False,
        imside=config.dataset.height,
        channels=config.dataset.channels,
    )

    # ── Enroll features for variance estimation ────────────────────────────
    print(f"[1.3v3] extracting enroll features (for variance estimation)...")
    enroll_paths, enroll_labels = load_paths_labels_from_txt(str(config.dataset.enroll_file))
    if args.max_enroll is not None and len(enroll_paths) > args.max_enroll:
        enroll_paths = enroll_paths[:args.max_enroll]
        enroll_labels = enroll_labels[:args.max_enroll]
    enroll_raw = extract_features(
        model, enroll_paths, transform, device,
        batch_size=64, channels=config.dataset.channels,
    )
    enroll_norm = project_and_normalize(np.asarray(enroll_raw), projection, device)
    print(f"  enroll normalized: {tuple(enroll_norm.shape)}, "
          f"check norm={enroll_norm.norm(dim=1).mean():.4f} (should be 1.0)")

    # Filter enroll to users present in gallery
    gallery_set = set(gallery_user_ids)
    valid_mask = torch.tensor(
        [int(y) in gallery_set for y in enroll_labels], dtype=torch.bool
    )
    enroll_norm_f = enroll_norm[valid_mask]
    enroll_labels_f = [y for y, m in zip(enroll_labels, valid_mask.tolist()) if m]
    print(f"  enroll in gallery: {enroll_norm_f.shape[0]} samples, "
          f"{len(set(enroll_labels_f))} users")

    # ── Estimate variance models (NCM-centered, normalized-space) ──────────
    print(f"[1.3v3] fitting variance models...")
    var_models = estimate_variance_models(
        enroll_norm_f, enroll_labels_f, class_means_dict,
    )

    # ── Eval features ──────────────────────────────────────────────────────
    print(f"[1.3v3] extracting eval probe features...")
    eval_paths, eval_labels = load_paths_labels_from_txt(str(config.dataset.eval_probe_file))
    if args.max_eval is not None and len(eval_paths) > args.max_eval:
        eval_paths = eval_paths[:args.max_eval]
        eval_labels = eval_labels[:args.max_eval]
    eval_raw = extract_features(
        model, eval_paths, transform, device,
        batch_size=64, channels=config.dataset.channels,
    )
    eval_norm = project_and_normalize(np.asarray(eval_raw), projection, device)
    print(f"  eval normalized: {tuple(eval_norm.shape)}, "
          f"check norm={eval_norm.norm(dim=1).mean():.4f}")

    # Make sure every gallery user has a variance (fallback to shared)
    for u in gallery_user_ids:
        if u not in var_models['per_user_sigma']:
            var_models['per_user_sigma'][u] = var_models['sigma_shared_per_dim']

    # ── Precompute floor-independent tensors ─────────────────────────────
    print(f"[1.3v3] precomputing cos_sim, diff_sq ...")
    with torch.no_grad():
        cos_sim_all = eval_norm @ class_means_mat.T
        diff_all = eval_norm.unsqueeze(1) - class_means_mat.unsqueeze(0)
        diff_sq_all = diff_all ** 2
        del diff_all
    precomputed = {'cos_sim': cos_sim_all, 'diff_sq': diff_sq_all}

    # ── Floor sweep ───────────────────────────────────────────────────────
    sigma_shared_scalar = var_models['sigma_shared_scalar']
    print(f"[1.3v3] sigma_shared_scalar = {sigma_shared_scalar:.6e}")

    sweep_results = []
    sweep_verdicts = []
    for floor_rel in FLOOR_REL_SWEEP:
        var_floor = max(floor_rel * sigma_shared_scalar, FLOOR_ABS_MIN)
        print(f"\n[1.3v3] computing rho at floor_rel={floor_rel} → var_floor={var_floor:.6e}")
        rho_stats = compute_rho_per_probe(
            eval_norm, class_means_mat, gallery_user_ids,
            var_models['per_user_sigma'], var_models['sigma_shared_per_dim'],
            var_floor=var_floor, precomputed=precomputed,
        )
        rho_stats['floor_rel'] = floor_rel
        v = render_verdict_for_floor(rho_stats, var_models['stats'])
        v['floor_rel'] = floor_rel
        v['var_floor'] = var_floor
        v['floor_active_frac'] = rho_stats['floor_active_frac']
        sweep_results.append(v)
        sweep_verdicts.append(rho_stats)
        print(f"  floor_active_frac={rho_stats['floor_active_frac']:.4f}  "
              f"rho_all={rho_stats['mean_rho_all']:.4f}  "
              f"rho_lm={rho_stats['mean_rho_low_margin']}  "
              f"top1={rho_stats['top1_agreement_gauss_vs_cos']:.4f}  "
              f"top1_lm={rho_stats['top1_agreement_low_margin']}")

    verdict = render_sweep_verdict(sweep_results)

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n=== Variance Stats (L2-normalized space, NO floor) ===")
    s = var_models['stats']
    print(f"  space:                       {s['space']}")
    print(f"  floor strategy:              {s['floor_strategy']}")
    print(f"  floor_rel_sweep:             {s['floor_rel_sweep']}")
    print(f"  shrinkage n_0:               {s['shrinkage_n0']}")
    print(f"  n_users:                     {s['n_users_with_var']}")
    print(f"  mean samples/user:           {s['mean_samples_per_user']}")
    print(f"  sigma_shared_scalar:         {s['sigma_shared_scalar']:.6e}")
    print(f"  mean per-user variance:      {s['mean_per_user_var']:.6e}")
    print(f"  user_var_CV:                 {s['user_var_CV']}")
    print(f"  dim_var_CV (no floor):       {s['dim_var_CV']}")
    print(f"  pooled residual anisotropy:  {s['pooled_residual_anisotropy']}")
    print(f"  pooled residual eff_rank:    {s['pooled_residual_eff_rank']}")
    print(f"  pooled residual condition:   {s['pooled_residual_condition']}")

    print("\n=== Floor Sweep Table ===")
    print(f"  {'floor_rel':>10} {'var_floor':>12} {'floor%':>8} "
          f"{'rho_all':>8} {'rho_lm':>8} {'top1':>8} {'top1_lm':>8}  action")
    for sv, rs in zip(sweep_results, sweep_verdicts):
        rho_lm_str = f"{rs['mean_rho_low_margin']:.4f}" if rs['mean_rho_low_margin'] is not None else "  N/A "
        t1lm_str = f"{rs['top1_agreement_low_margin']:.4f}" if rs['top1_agreement_low_margin'] is not None else "  N/A "
        print(f"  {sv['floor_rel']:>10.2f} {sv['var_floor']:>12.4e} "
              f"{sv['floor_active_frac']*100:>7.2f}% "
              f"{rs['mean_rho_all']:>8.4f} {rho_lm_str:>8} "
              f"{rs['top1_agreement_gauss_vs_cos']:>8.4f} {t1lm_str:>8}  {sv['action']}")

    print(f"\n=== Sweep Verdict ===")
    print(f"  ACTION: {verdict['action']}")
    print(f"  rationale: {verdict['rationale']}")
    sm = verdict['sweep_summary']
    print(f"  rho range:        [{sm['rho_min']:.4f}, {sm['rho_max']:.4f}] "
          f"(delta={sm['delta_rho']:.4f})")
    print(f"  top1 range:       [{sm['top1_min']:.4f}, {sm['top1_max']:.4f}] "
          f"(delta={sm['delta_top1']:.4f})")
    print(f"  rho_low_margin_min: {sm['rho_low_margin_min']}")

    save_obj = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'design_fixes': {
            'E1_gaussian_center': 'NCM_mean (not raw proxy)',
            'E2_variance_space': 'L2_normalized',
            'E3_var_floor': 'adaptive_sweep (FLOOR_REL_SWEEP × sigma_shared_scalar)',
            'E4_shrinkage_n0': SHRINKAGE_N0,
        },
        'variance_stats': var_models['stats'],
        'sweep_results': sweep_verdicts,
        'sweep_verdicts': sweep_results,
        'verdict': verdict,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(save_obj, f, indent=2, default=float)
    print(f"\n[1.3v3] saved: {args.output}")


if __name__ == '__main__':
    main()
