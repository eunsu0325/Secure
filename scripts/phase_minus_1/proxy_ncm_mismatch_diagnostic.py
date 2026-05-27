"""
Phase -1.8: Proxy-NCM Mismatch Diagnostic (Stage 0 — Smoke Test).

Goal: verify whether the geometric divergence between the ProxyAnchor training
anchor (p_u) and the NCM inference template (μ_u^NCM) is associated with
sequential open-set degradation (silent rate, score-overlap gap, spurious accept).

This is a DIAGNOSTIC-ONLY script. It does NOT implement GPAC loss, Gaussian
loss, alignment loss, GHOST pilot, or NA-QAR training. It only measures the
mismatch and reports a Strong/Weak/Negative verdict.

═══════════════════════════════════════════════════════════════════════════
Stage 0 (this run) — Minimal smoke test on a single checkpoint:
  * Load N=10 checkpoint only
  * Resolve enrolled_ids using enrollment order as canonical
  * Allow proxy/NCM metadata as superset; abort only if enrolled ids missing
  * Apply active masking before every argmax/score/threshold
  * Compute minimal metric subset: M1, M2, M5, M6, M8
  * Compute both fixed and recal τ partitions
  * Verify 6-way partition exclusivity for both τ policies
  * Print sanity report; save JSON with explicit schema
═══════════════════════════════════════════════════════════════════════════

Output JSON record schema (each record):
  {
    "checkpoint_N":    int,
    "cohort":          "all_users" | "early_user",
    "threshold_policy":"fixed_tau" | "recal_tau" | "N/A",
    "unknown_source":  "all" | "future" | "external" | "N/A",
    "partition":       "known_correct_accepted" | ... | "known_all" | "N/A",
    "mask":            "none" | "low_margin" | "proxy_confident" | "ncm_risk",
    "metric_name":     "M1_disagreement" | ... ,
    "value":           float | null,
    "n_samples":       int,
    "confidence_flag": "ok" | "low_confidence" | "insufficient_n"
  }

Usage (Stage 0 smoke):
  python scripts/phase_minus_1/proxy_ncm_mismatch_diagnostic.py \\
      --config   configs/proj_512_legacy_50u.yaml \\
      --checkpoint /content/drive/MyDrive/coconut_proj512_50u/coconut_results/checkpoint_exp_10.pth \\
      --output   /content/drive/MyDrive/phase_minus_1/mismatch_smoke_N10.json \\
      --stage    0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


# ═══════════════════════════════════════════════════════════════════════════
# Constants (GPT 1–8차 feedback)
# ═══════════════════════════════════════════════════════════════════════════
TARGET_FPIR = 0.01           # 1% FPIR for τ calibration
LOW_MARGIN_FACTOR = 0.5      # low_margin = 0.5 × median(margin_ncm)
MIN_N_FOR_SPEARMAN = 5
MIN_N_FOR_OK = 10


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint loading + enrolled-ID resolution
# ═══════════════════════════════════════════════════════════════════════════
def load_checkpoint_components(ckpt_path: str, device: torch.device):
    """Load CCNet + ProjectionHead + raw proxies + NCM means (L2-normalized).

    Returns:
        ckpt          : raw dict
        model         : CCNet (eval mode)
        projection    : ProjectionHead (eval mode)
        proxies       : (C_total, D) tensor — RAW, not L2-normalized
        class_to_idx  : dict {user_id: proxy_row_idx}
        class_means   : (max_id+1, D) tensor — already L2-normalized
    """
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

    pa = ckpt['proxy_anchor_data']
    proxies = pa['proxies'].to(device)
    class_to_idx = {int(k): int(v) for k, v in pa['class_to_idx'].items()}

    class_means = ckpt['ncm_state_dict']['class_means'].to(device)

    return ckpt, model, projection, proxies, class_to_idx, class_means


def resolve_enrolled_ids(
    ckpt: dict,
    class_to_idx: Dict[int, int],
    class_means: torch.Tensor,
    expected_N: int,
) -> List[int]:
    """Resolve enrolled_ids using enrollment order as canonical.

    Uses class_to_idx ordered by proxy index (= enrollment order if proxies
    are appended in order). Cross-checks proxy IDs and NCM non-zero rows with
    subset relation only (GPT 7차/8차 — strict equality는 superset checkpoint를
    잘못 abort시키므로 subset만 보장).

    Raises AssertionError only if any enrolled id is missing from proxy or NCM.
    """
    # Canonical source: class_to_idx keys sorted by their proxy row index
    keys_by_idx = sorted(class_to_idx.items(), key=lambda kv: kv[1])
    enrolled_ids_from_order = [uid for uid, _ in keys_by_idx]
    enrolled_set = set(enrolled_ids_from_order)

    proxy_ids = set(class_to_idx.keys())
    ncm_ids = set(
        int(i) for i in range(class_means.shape[0])
        if class_means[i].norm().item() > 1e-6
    )

    missing_in_proxy = enrolled_set - proxy_ids
    missing_in_ncm = enrolled_set - ncm_ids
    assert not missing_in_proxy, f"enrolled ids missing in proxy: {missing_in_proxy}"
    assert not missing_in_ncm, f"enrolled ids missing in NCM: {missing_in_ncm}"

    extra_in_proxy = proxy_ids - enrolled_set
    extra_in_ncm = ncm_ids - enrolled_set
    if extra_in_proxy:
        print(f"  [WARN] proxy contains {len(extra_in_proxy)} non-enrolled "
              f"classes — active masking applied")
    if extra_in_ncm:
        print(f"  [WARN] NCM contains {len(extra_in_ncm)} non-enrolled rows "
              f"— active masking applied")

    if expected_N is not None and len(enrolled_ids_from_order) != expected_N:
        print(f"  [WARN] expected N={expected_N}, got {len(enrolled_ids_from_order)}")

    return enrolled_ids_from_order


# ═══════════════════════════════════════════════════════════════════════════
# Feature extraction
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def extract_and_project(model, projection, paths, transform, device, channels):
    """Extract raw 2048-D features → project to 512-D → L2-normalize. (N, D)"""
    raw = extract_features(model, paths, transform, device, batch_size=64,
                           channels=channels)
    x = torch.from_numpy(np.asarray(raw)).float().to(device)
    h = projection(x)
    return F.normalize(h, p=2, dim=1)


# ═══════════════════════════════════════════════════════════════════════════
# τ calibration (fixed = N=10 dev, recal = each-N dev)
# ═══════════════════════════════════════════════════════════════════════════
def calibrate_tau(unknown_features: torch.Tensor,
                  class_means_active: torch.Tensor,
                  target_FPIR: float = TARGET_FPIR) -> float:
    """Calibrate τ so that P(top-1 cosine ≥ τ on impostors) ≈ target_FPIR."""
    scores = unknown_features @ class_means_active.T  # (N_unk, C)
    top1_scores = scores.max(dim=1).values.cpu().numpy()
    return float(np.quantile(top1_scores, 1.0 - target_FPIR))


def measure_realized_FPIR(unknown_features: torch.Tensor,
                          class_means_active: torch.Tensor,
                          tau: float) -> float:
    scores = unknown_features @ class_means_active.T
    top1_scores = scores.max(dim=1).values.cpu().numpy()
    return float((top1_scores >= tau).mean())


# ═══════════════════════════════════════════════════════════════════════════
# S-norm (NCM only — GPT 8차 R1: proxy에는 적용 금지)
# ═══════════════════════════════════════════════════════════════════════════
def snorm(scores: torch.Tensor, cohort_scores: torch.Tensor) -> torch.Tensor:
    """Per-class Z-score normalization using cohort statistics."""
    mean_c = cohort_scores.mean(dim=0, keepdim=True)
    std_c = cohort_scores.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (scores - mean_c) / std_c


# ═══════════════════════════════════════════════════════════════════════════
# Spearman ρ (NaN-safe + n-threshold confidence flag)
# ═══════════════════════════════════════════════════════════════════════════
def spearman_rho(
    x: np.ndarray, y: np.ndarray,
    min_n: int = MIN_N_FOR_SPEARMAN, ok_n: int = MIN_N_FOR_OK,
) -> Tuple[Optional[float], int, str]:
    """Returns (rho, n, confidence_flag)."""
    n = len(x)
    if n < min_n:
        return None, n, "insufficient_n"
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx_c = rx - rx.mean()
    ry_c = ry - ry.mean()
    denom = np.sqrt((rx_c ** 2).sum() * (ry_c ** 2).sum())
    if denom < 1e-12:
        return None, n, "insufficient_n"
    rho = float((rx_c * ry_c).sum() / denom)
    flag = "low_confidence" if n < ok_n else "ok"
    return rho, n, flag


# ═══════════════════════════════════════════════════════════════════════════
# 6-way primary partition (GPT 4차/5차)
# ═══════════════════════════════════════════════════════════════════════════
def partition_6way(
    pred_ncm: np.ndarray,
    score_top1_ncm: np.ndarray,
    in_gallery_labels: np.ndarray,
    tau: float,
) -> Dict[str, np.ndarray]:
    """Six exclusive partitions for given τ:
        known_correct_accepted, known_correct_rejected_silent,
        known_wrong_accepted_misidentified, known_wrong_rejected_degraded,
        unknown_rejected, unknown_spurious
    Verifies union==all and disjoint.
    """
    N = len(pred_ncm)
    in_gallery = in_gallery_labels >= 0
    accepted = score_top1_ncm >= tau
    rank1_correct = (pred_ncm == in_gallery_labels) & in_gallery
    rank1_wrong_in_gal = (~rank1_correct) & in_gallery

    parts = {
        'known_correct_accepted':             rank1_correct & accepted,
        'known_correct_rejected_silent':      rank1_correct & ~accepted,
        'known_wrong_accepted_misidentified': rank1_wrong_in_gal & accepted,
        'known_wrong_rejected_degraded':      rank1_wrong_in_gal & ~accepted,
        'unknown_rejected':                   ~in_gallery & ~accepted,
        'unknown_spurious':                   ~in_gallery & accepted,
    }
    # Disjoint + union check
    counts = np.zeros(N, dtype=int)
    for mask in parts.values():
        counts += mask.astype(int)
    assert (counts == 1).all(), \
        f"partition not exclusive: counts min={counts.min()}, max={counts.max()}"
    return parts


# ═══════════════════════════════════════════════════════════════════════════
# Metric implementations (Stage 0 minimal: M1, M2, M5, M6, M8)
# ═══════════════════════════════════════════════════════════════════════════
def compute_M1_disagreement(pred_proxy: np.ndarray, pred_ncm: np.ndarray,
                            mask: np.ndarray) -> Tuple[Optional[float], int]:
    """P(argmax_proxy ≠ argmax_ncm) on `mask`."""
    n = int(mask.sum())
    if n == 0:
        return None, 0
    return float((pred_proxy[mask] != pred_ncm[mask]).mean()), n


def compute_M5_overlap_gap(genuine_scores: np.ndarray,
                           impostor_scores: np.ndarray) -> Tuple[Optional[float], int, int]:
    """score_overlap_gap = P5(genuine) − P99(impostor)."""
    n_g, n_i = len(genuine_scores), len(impostor_scores)
    if n_g < 5 or n_i < 5:
        return None, n_g, n_i
    return float(np.percentile(genuine_scores, 5) - np.percentile(impostor_scores, 99)), n_g, n_i


def compute_M6_silent_rate(known_rank1_correct: np.ndarray,
                           known_score_top1: np.ndarray,
                           tau: float) -> Tuple[Optional[float], int]:
    """silent_rate = P(rank-1 correct AND s_top1 < τ) on known probes."""
    n = len(known_rank1_correct)
    if n == 0:
        return None, 0
    return float((known_rank1_correct & (known_score_top1 < tau)).mean()), n


def compute_M8_spurious(unknown_pred_idx: np.ndarray,
                        unknown_score_top1: np.ndarray,
                        tau: float, num_classes: int) -> Dict:
    """spurious_accept_rate + entropy + concentration (entropy normalized)."""
    n = len(unknown_score_top1)
    if n == 0:
        return {'n': 0, 'spurious_accept_rate': None}
    accepted_mask = unknown_score_top1 >= tau
    spurious_count = int(accepted_mask.sum())
    if spurious_count == 0:
        return {
            'n': n,
            'spurious_accept_rate': 0.0,
            'spurious_count': 0,
            'false_accept_entropy_raw': None,
            'false_accept_entropy_norm': None,
            'top_spurious_user_share': None,
            'spurious_user_gini': None,
        }
    accepted_users = unknown_pred_idx[accepted_mask]
    user_counts = np.bincount(accepted_users, minlength=num_classes)
    p = user_counts / max(user_counts.sum(), 1)
    p_pos = p[p > 0]

    H_raw = float(-(p_pos * np.log(p_pos)).sum())
    H_norm = float(H_raw / np.log(num_classes)) if num_classes > 1 else None
    top_share = float(p.max())
    # Gini coefficient over class probabilities
    sorted_p = np.sort(p)
    n_c = len(p)
    if sorted_p.sum() > 0:
        cum_idx_mul_p = np.sum((np.arange(1, n_c + 1)) * sorted_p)
        gini = float((2 * cum_idx_mul_p) / (n_c * sorted_p.sum()) - (n_c + 1) / n_c)
    else:
        gini = None

    return {
        'n': n,
        'spurious_accept_rate': float(spurious_count / n),
        'spurious_count': spurious_count,
        'false_accept_entropy_raw': H_raw,
        'false_accept_entropy_norm': H_norm,
        'top_spurious_user_share': top_share,
        'spurious_user_gini': gini,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stage', type=int, default=0,
                        help='0=smoke test (this implementation), 1/2=future')
    args = parser.parse_args()

    if args.stage != 0:
        raise NotImplementedError(
            f"Stage {args.stage} not implemented yet — this script supports Stage 0 only.")

    config = ConfigParser(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[1.8 Stage 0] device: {device}")
    print(f"[1.8 Stage 0] checkpoint: {args.checkpoint}")

    # ── Load checkpoint ─────────────────────────────────────────────────
    ckpt, model, projection, proxies, class_to_idx, class_means = \
        load_checkpoint_components(args.checkpoint, device)

    experience_count = ckpt.get('experience_count', None)
    print(f"  experience_count: {experience_count}")

    # ── Resolve enrolled_ids + active masking ───────────────────────────
    enrolled_ids_N = resolve_enrolled_ids(
        ckpt, class_to_idx, class_means, expected_N=experience_count
    )
    N = len(enrolled_ids_N)
    print(f"  enrolled_ids: {N} users (first 5: {enrolled_ids_N[:5]})")

    proxy_indices = [class_to_idx[u] for u in enrolled_ids_N]
    proxies_active_raw = proxies[proxy_indices]                       # (N, D) raw
    proxies_active = F.normalize(proxies_active_raw, p=2, dim=1)      # (N, D) L2-norm
    class_means_active = torch.stack(
        [class_means[u] for u in enrolled_ids_N], dim=0
    )                                                                  # (N, D) L2-norm
    print(f"  proxies_active: {tuple(proxies_active.shape)}, "
          f"class_means_active: {tuple(class_means_active.shape)}")

    # ── Probe sets ──────────────────────────────────────────────────────
    transform = get_scr_transforms(
        train=False,
        imside=config.dataset.height,
        channels=config.dataset.channels,
    )

    print("\n[1.8 Stage 0] extracting eval probes ...")
    eval_paths, eval_labels = load_paths_labels_from_txt(
        str(config.dataset.eval_probe_file))
    eval_features = extract_and_project(
        model, projection, eval_paths, transform, device,
        config.dataset.channels)

    print("[1.8 Stage 0] extracting unknown_dev (τ calibration) ...")
    udev_paths, _ = load_paths_labels_from_txt(str(config.dataset.unknown_dev_file))
    udev_features = extract_and_project(
        model, projection, udev_paths, transform, device,
        config.dataset.channels)

    uext_features = None
    if config.dataset.unknown_test_file and os.path.exists(
            str(config.dataset.unknown_test_file)):
        print("[1.8 Stage 0] extracting unknown_test (external) ...")
        uext_paths, _ = load_paths_labels_from_txt(
            str(config.dataset.unknown_test_file))
        uext_features = extract_and_project(
            model, projection, uext_paths, transform, device,
            config.dataset.channels)

    # ── Split eval probes by gallery membership ─────────────────────────
    enrolled_set = set(enrolled_ids_N)
    gallery_idx_of = {u: i for i, u in enumerate(enrolled_ids_N)}
    eval_in_gallery_mask = np.array(
        [int(y) in enrolled_set for y in eval_labels], dtype=bool)
    in_gallery_labels_array = np.array([
        gallery_idx_of[int(y)] if int(y) in enrolled_set else -1
        for y in eval_labels
    ], dtype=np.int64)

    n_known = int(eval_in_gallery_mask.sum())
    n_future = int((~eval_in_gallery_mask).sum())
    n_external = int(uext_features.shape[0]) if uext_features is not None else 0
    print(f"  probe counts: known={n_known}, future={n_future}, external={n_external}")

    # ── τ calibration (fixed + recal — Stage 0 both at this N) ─────────
    tau_fixed = calibrate_tau(udev_features, class_means_active, TARGET_FPIR)
    tau_recal = calibrate_tau(udev_features, class_means_active, TARGET_FPIR)
    realized_fpir_fixed = measure_realized_FPIR(
        udev_features, class_means_active, tau_fixed)
    realized_fpir_recal = measure_realized_FPIR(
        udev_features, class_means_active, tau_recal)
    print(f"  τ_fixed={tau_fixed:.4f} (realized FPIR={realized_fpir_fixed:.4f})")
    print(f"  τ_recal={tau_recal:.4f} (realized FPIR={realized_fpir_recal:.4f})")

    # ── Score computation ───────────────────────────────────────────────
    eval_ncm_scores = eval_features @ class_means_active.T              # (N_eval, C)
    eval_proxy_scores = eval_features @ proxies_active.T                # (N_eval, C)

    # S-norm (NCM only — R1)
    udev_ncm_cohort = udev_features @ class_means_active.T
    eval_ncm_scores_snorm = snorm(eval_ncm_scores, udev_ncm_cohort)

    # top-1 / top-2 / margin
    sorted_ncm = torch.sort(eval_ncm_scores, dim=1, descending=True).values
    sorted_proxy = torch.sort(eval_proxy_scores, dim=1, descending=True).values
    ncm_top1 = sorted_ncm[:, 0]
    ncm_top2 = sorted_ncm[:, 1] if eval_ncm_scores.shape[1] >= 2 else torch.zeros_like(ncm_top1)
    proxy_top1 = sorted_proxy[:, 0]
    proxy_top2 = sorted_proxy[:, 1] if eval_proxy_scores.shape[1] >= 2 else torch.zeros_like(proxy_top1)
    margin_ncm = (ncm_top1 - ncm_top2).cpu().numpy()
    margin_proxy = (proxy_top1 - proxy_top2).cpu().numpy()
    ncm_top1_np = ncm_top1.cpu().numpy()
    pred_ncm = eval_ncm_scores.argmax(dim=1).cpu().numpy()
    pred_proxy = eval_proxy_scores.argmax(dim=1).cpu().numpy()
    eval_ncm_snorm_np = eval_ncm_scores_snorm.cpu().numpy()

    # low-margin mask (NCM-based)
    median_margin_ncm = float(np.median(margin_ncm))
    low_margin_thresh = LOW_MARGIN_FACTOR * median_margin_ncm
    low_margin_mask = margin_ncm < low_margin_thresh

    # ── 6-way partitions (fixed + recal, separately) ────────────────────
    parts_fixed = partition_6way(
        pred_ncm, ncm_top1_np, in_gallery_labels_array, tau_fixed)
    parts_recal = partition_6way(
        pred_ncm, ncm_top1_np, in_gallery_labels_array, tau_recal)

    # ── Sanity report ───────────────────────────────────────────────────
    print("\n══════════════════════════ Sanity Report ══════════════════════════")
    print(f"  enrolled classes: {N}")
    print(f"  probe counts: known={n_known}, future={n_future}, external={n_external}")
    print(f"  median margin (NCM): {median_margin_ncm:.4f}")
    print(f"  low margin threshold: {low_margin_thresh:.4f} → "
          f"{int(low_margin_mask.sum())}/{len(eval_labels)} probes")
    print(f"\n  Partition counts (fixed τ = {tau_fixed:.4f}):")
    for k, mask in parts_fixed.items():
        print(f"    {k:42s} {int(mask.sum()):>5d}")
    print(f"\n  Partition counts (recal τ = {tau_recal:.4f}):")
    for k, mask in parts_recal.items():
        print(f"    {k:42s} {int(mask.sum()):>5d}")

    # ── Compute metrics — Stage 0 minimal subset ────────────────────────
    records: List[Dict] = []

    def push(*, metric_name, value, n_samples, confidence_flag="ok",
             cohort="all_users", threshold_policy="N/A",
             unknown_source="N/A", partition="N/A", mask="none"):
        records.append({
            "checkpoint_N": N,
            "cohort": cohort,
            "threshold_policy": threshold_policy,
            "unknown_source": unknown_source,
            "partition": partition,
            "mask": mask,
            "metric_name": metric_name,
            "value": value,
            "n_samples": n_samples,
            "confidence_flag": confidence_flag,
        })

    # ── M1: top1 disagreement (known + low_margin∩known) ───────────────
    for mask_name, mask_arr in [
        ("none", eval_in_gallery_mask),
        ("low_margin", eval_in_gallery_mask & low_margin_mask),
    ]:
        v, n = compute_M1_disagreement(pred_proxy, pred_ncm, mask_arr)
        flag = "ok" if n >= MIN_N_FOR_OK else (
            "low_confidence" if n >= MIN_N_FOR_SPEARMAN else "insufficient_n")
        push(metric_name="M1_top1_disagreement", value=v, n_samples=n,
             confidence_flag=flag, partition="known_all", mask=mask_name)

    # ── M2: margin Spearman (4 subsets: all/low_margin/silent/spurious) ─
    # All known
    rho, n, flag = spearman_rho(
        margin_proxy[eval_in_gallery_mask],
        margin_ncm[eval_in_gallery_mask])
    push(metric_name="M2_margin_spearman", value=rho, n_samples=n,
         confidence_flag=flag, partition="known_all", mask="none")
    # known ∩ low_margin
    lm_mask = eval_in_gallery_mask & low_margin_mask
    rho, n, flag = spearman_rho(margin_proxy[lm_mask], margin_ncm[lm_mask])
    push(metric_name="M2_margin_spearman", value=rho, n_samples=n,
         confidence_flag=flag, partition="known_all", mask="low_margin")
    # known_correct_rejected_silent (fixed)
    silent_mask_fixed = parts_fixed['known_correct_rejected_silent']
    rho, n, flag = spearman_rho(margin_proxy[silent_mask_fixed],
                                margin_ncm[silent_mask_fixed])
    push(metric_name="M2_margin_spearman", value=rho, n_samples=n,
         confidence_flag=flag, threshold_policy="fixed_tau",
         partition="known_correct_rejected_silent")
    # unknown_spurious (fixed)
    spurious_mask_fixed = parts_fixed['unknown_spurious']
    rho, n, flag = spearman_rho(margin_proxy[spurious_mask_fixed],
                                margin_ncm[spurious_mask_fixed])
    push(metric_name="M2_margin_spearman", value=rho, n_samples=n,
         confidence_flag=flag, threshold_policy="fixed_tau",
         partition="unknown_spurious")

    # ── M5: score_overlap_gap (NCM + S-norm, main) ──────────────────────
    # genuine = in-gallery rank-1 correct, score for own class (S-normed)
    rank1_correct_mask = (pred_ncm == in_gallery_labels_array) & eval_in_gallery_mask
    genuine_snorm = eval_ncm_snorm_np[
        rank1_correct_mask, in_gallery_labels_array[rank1_correct_mask]]
    # Impostor — future (BJTU 내 out-of-gallery from eval_probe_file)
    impostor_future_snorm = eval_ncm_snorm_np[~eval_in_gallery_mask].max(axis=1)
    v, n_g, n_i = compute_M5_overlap_gap(genuine_snorm, impostor_future_snorm)
    push(metric_name="M5_score_overlap_gap_main", value=v,
         n_samples=min(n_g, n_i), partition="known_vs_unknown",
         unknown_source="future")
    # Impostor — all (future ∪ external)
    if uext_features is not None:
        uext_ncm_scores = uext_features @ class_means_active.T
        uext_ncm_snorm = snorm(uext_ncm_scores, udev_ncm_cohort)
        impostor_ext_snorm = uext_ncm_snorm.max(dim=1).values.cpu().numpy()
        impostor_all_snorm = np.concatenate([
            impostor_future_snorm, impostor_ext_snorm])
        v, n_g, n_i = compute_M5_overlap_gap(genuine_snorm, impostor_all_snorm)
        push(metric_name="M5_score_overlap_gap_main", value=v,
             n_samples=min(n_g, n_i), partition="known_vs_unknown",
             unknown_source="all")
        v, n_g, n_i = compute_M5_overlap_gap(genuine_snorm, impostor_ext_snorm)
        push(metric_name="M5_score_overlap_gap_main", value=v,
             n_samples=min(n_g, n_i), partition="known_vs_unknown",
             unknown_source="external")

    # ── M6: silent_rate (fixed + recal) + realized_FPIR ─────────────────
    known_rank1_correct = rank1_correct_mask[eval_in_gallery_mask]
    known_top1 = ncm_top1_np[eval_in_gallery_mask]
    for policy, tau, rfpir in [
        ("fixed_tau", tau_fixed, realized_fpir_fixed),
        ("recal_tau", tau_recal, realized_fpir_recal),
    ]:
        sr, n = compute_M6_silent_rate(known_rank1_correct, known_top1, tau)
        push(metric_name="M6_silent_rate", value=sr, n_samples=n,
             threshold_policy=policy, partition="known_all")
        push(metric_name="M6_realized_FPIR", value=rfpir,
             n_samples=int(udev_features.shape[0]),
             threshold_policy=policy, unknown_source="dev")
        push(metric_name="M6_tau", value=tau, n_samples=int(udev_features.shape[0]),
             threshold_policy=policy)

    # ── M8: spurious accept + concentration (3 unknown sources × 2 τ) ──
    # future (BJTU 내 out-of-gallery from eval_probe_file)
    fut_pred = pred_ncm[~eval_in_gallery_mask]
    fut_top1 = ncm_top1_np[~eval_in_gallery_mask]
    for policy, tau in [("fixed_tau", tau_fixed), ("recal_tau", tau_recal)]:
        r = compute_M8_spurious(fut_pred, fut_top1, tau, num_classes=N)
        for mk in ["spurious_accept_rate", "false_accept_entropy_raw",
                   "false_accept_entropy_norm", "top_spurious_user_share",
                   "spurious_user_gini"]:
            push(metric_name=f"M8_{mk}", value=r.get(mk),
                 n_samples=r.get('spurious_count', 0) if mk != 'spurious_accept_rate'
                 else r['n'],
                 threshold_policy=policy, unknown_source="future")

    # external if available
    if uext_features is not None:
        uext_ncm_scores = uext_features @ class_means_active.T
        uext_top1 = uext_ncm_scores.max(dim=1).values.cpu().numpy()
        uext_argmax = uext_ncm_scores.argmax(dim=1).cpu().numpy()
        for policy, tau in [("fixed_tau", tau_fixed), ("recal_tau", tau_recal)]:
            r = compute_M8_spurious(uext_argmax, uext_top1, tau, num_classes=N)
            for mk in ["spurious_accept_rate", "false_accept_entropy_raw",
                       "false_accept_entropy_norm", "top_spurious_user_share",
                       "spurious_user_gini"]:
                push(metric_name=f"M8_{mk}", value=r.get(mk),
                     n_samples=r.get('spurious_count', 0) if mk != 'spurious_accept_rate'
                     else r['n'],
                     threshold_policy=policy, unknown_source="external")

    # ── NaN summary ─────────────────────────────────────────────────────
    nan_count = sum(
        1 for rec in records
        if rec['value'] is None
        or (isinstance(rec['value'], float) and np.isnan(rec['value']))
    )
    print(f"\n  records: {len(records)}, null/NaN: {nan_count}")

    # ── Save JSON ───────────────────────────────────────────────────────
    output = {
        'stage': args.stage,
        'config': args.config,
        'checkpoint': args.checkpoint,
        'metadata': {
            'experience_count': experience_count,
            'N_enrolled': N,
            'enrolled_ids': enrolled_ids_N,
            'n_known_probes': n_known,
            'n_future_probes': n_future,
            'n_external_probes': n_external,
            'tau_fixed': tau_fixed,
            'tau_recal': tau_recal,
            'realized_fpir_fixed': realized_fpir_fixed,
            'realized_fpir_recal': realized_fpir_recal,
            'median_margin_ncm': median_margin_ncm,
            'low_margin_threshold': low_margin_thresh,
            'partition_counts_fixed_tau': {k: int(v.sum()) for k, v in parts_fixed.items()},
            'partition_counts_recal_tau': {k: int(v.sum()) for k, v in parts_recal.items()},
            'total_records': len(records),
            'null_record_count': nan_count,
        },
        'records': records,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\n[1.8 Stage 0] saved: {args.output}")


if __name__ == '__main__':
    main()
