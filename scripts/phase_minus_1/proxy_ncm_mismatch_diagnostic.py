"""
Phase -1.8: Proxy-NCM Mismatch Diagnostic (Stage 2 — Full Sequential).

Tests whether the geometric divergence between the ProxyAnchor training
anchor (p_u) and the NCM inference template (μ_u^NCM) is associated with
sequential open-set degradation. DIAGNOSTIC-ONLY — no GPAC, no Gaussian loss,
no alignment loss, no GHOST pilot, no NA-QAR training.

═══════════════════════════════════════════════════════════════════════════
Stage 2 (this implementation):
  * 5 checkpoints: N=10/20/30/40/50
  * 2 cohorts: all_users + early_user (=enrolled_ids_10)
  * All M1-M14 metrics:
      M1 proxy_ncm_top1_disagreement (all + low_margin)
      M2 margin Spearman (4 subsets: all/low_margin/known_silent/unknown_spurious)
      M3 proxy_safe_ncm_risk_rate (dynamic + fixedQ)
      M4 ncm_silent_proxy_confident_rate (dynamic + fixedQ)
      M5 score_overlap_gap_main (NCM+S-norm, 3 unknown sources)
      M5b score_overlap_gap_aux (raw NCM, raw proxy)
      M6 silent_rate + realized_FPIR + tau (fixed + recal)
      M7 FNIR @ 1% FPIR (fixed + recal)
      M8 spurious + concentration (3 unknown sources × 2 tau)
      M9 per-user mismatch ↔ silent/FNIR/TAR correlation (with bootstrap CI)
      M10 proxy_ncm_margin_gap (mean/p95 + low-margin subset)
      M11 proxy_score_ncm_score_spearman
      M12 proxy_high_ncm_low_rate (dynamic + fixedQ)
      M13/M14 lagged temporal association (descriptive only)
  * fixed_tau = N=10 dev calibration; recal_tau = each-N dev calibration
  * fixed_quantiles = N=10 known-probe quantiles; dynamic = each-N
  * Plots: main curves, aux curves, per-user correlation heatmap
  * Verdict: Strong / Weak / Negative

═══════════════════════════════════════════════════════════════════════════
Usage:
  python scripts/phase_minus_1/proxy_ncm_mismatch_diagnostic.py \\
      --config configs/proj_512_legacy_50u.yaml \\
      --ckpt-dir /content/drive/MyDrive/coconut_proj512_50u/coconut_results \\
      --output-dir /content/drive/MyDrive/phase_minus_1
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
# Constants
# ═══════════════════════════════════════════════════════════════════════════
TARGET_FPIR = 0.01
LOW_MARGIN_FACTOR = 0.5
MIN_N_FOR_SPEARMAN = 5
MIN_N_FOR_OK = 10
BOOTSTRAP_RESAMPLES = 500
N_CHECKPOINTS = [10, 20, 30, 40, 50]


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint loading
# ═══════════════════════════════════════════════════════════════════════════
def load_checkpoint_components(ckpt_path: str, device: torch.device):
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
    class_to_idx = {int(k): int(v) for k, v in pa['class_to_idx'].items()}

    class_means = ckpt['ncm_state_dict']['class_means'].to(device)

    return ckpt, model, projection, proxies, class_to_idx, class_means


def resolve_enrolled_ids(class_to_idx, class_means, expected_N=None):
    keys_by_idx = sorted(class_to_idx.items(), key=lambda kv: kv[1])
    enrolled_ids = [uid for uid, _ in keys_by_idx]
    enrolled_set = set(enrolled_ids)

    proxy_ids = set(class_to_idx.keys())
    ncm_ids = set(int(i) for i in range(class_means.shape[0])
                  if class_means[i].norm().item() > 1e-6)

    missing_in_proxy = enrolled_set - proxy_ids
    missing_in_ncm = enrolled_set - ncm_ids
    assert not missing_in_proxy, f"enrolled missing in proxy: {missing_in_proxy}"
    assert not missing_in_ncm, f"enrolled missing in NCM: {missing_in_ncm}"

    extra_proxy = proxy_ids - enrolled_set
    extra_ncm = ncm_ids - enrolled_set
    if extra_proxy:
        print(f"  [WARN] proxy +{len(extra_proxy)} non-enrolled (active mask)")
    if extra_ncm:
        print(f"  [WARN] NCM +{len(extra_ncm)} non-enrolled (active mask)")

    if expected_N is not None and len(enrolled_ids) != expected_N:
        print(f"  [WARN] expected N={expected_N}, got {len(enrolled_ids)}")
    return enrolled_ids


@torch.no_grad()
def extract_and_project(model, projection, paths, transform, device, channels):
    raw = extract_features(model, paths, transform, device, batch_size=64,
                           channels=channels)
    x = torch.from_numpy(np.asarray(raw)).float().to(device)
    h = projection(x)
    return F.normalize(h, p=2, dim=1)


# ═══════════════════════════════════════════════════════════════════════════
# τ calibration
# ═══════════════════════════════════════════════════════════════════════════
def calibrate_tau(unknown_features, class_means_active, target_FPIR=TARGET_FPIR):
    scores = unknown_features @ class_means_active.T
    top1_scores = scores.max(dim=1).values.cpu().numpy()
    return float(np.quantile(top1_scores, 1.0 - target_FPIR))


def realized_fpir(unknown_features, class_means_active, tau):
    scores = unknown_features @ class_means_active.T
    top1_scores = scores.max(dim=1).values.cpu().numpy()
    return float((top1_scores >= tau).mean())


# ═══════════════════════════════════════════════════════════════════════════
# S-norm
# ═══════════════════════════════════════════════════════════════════════════
def snorm(scores, cohort_scores):
    mean_c = cohort_scores.mean(dim=0, keepdim=True)
    std_c = cohort_scores.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (scores - mean_c) / std_c


# ═══════════════════════════════════════════════════════════════════════════
# Spearman ρ (NaN-safe + n-threshold)
# ═══════════════════════════════════════════════════════════════════════════
def spearman_rho(x, y, min_n=MIN_N_FOR_SPEARMAN, ok_n=MIN_N_FOR_OK):
    n = len(x)
    if n < min_n:
        return None, n, "insufficient_n"
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx_c = rx - rx.mean()
    ry_c = ry - ry.mean()
    denom = np.sqrt((rx_c ** 2).sum() * (ry_c ** 2).sum())
    if denom < 1e-12:
        return None, n, "insufficient_n"
    rho = float((rx_c * ry_c).sum() / denom)
    flag = "low_confidence" if n < ok_n else "ok"
    return rho, n, flag


def spearman_with_ci(x, y, n_resamples=BOOTSTRAP_RESAMPLES):
    """Bootstrap 95% CI for Spearman ρ."""
    rho, n, flag = spearman_rho(x, y)
    if rho is None or n < MIN_N_FOR_OK:
        return rho, n, flag, None, None
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        r, _, f = spearman_rho(x[idx], y[idx])
        if r is not None:
            boot.append(r)
    if len(boot) < n_resamples * 0.5:
        return rho, n, flag, None, None
    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return rho, n, flag, ci_lo, ci_hi


# ═══════════════════════════════════════════════════════════════════════════
# 6-way partition
# ═══════════════════════════════════════════════════════════════════════════
def partition_6way(pred_ncm, score_top1_ncm, in_gallery_labels, tau):
    N = len(pred_ncm)
    in_gal = in_gallery_labels >= 0
    accepted = score_top1_ncm >= tau
    rank1_correct = (pred_ncm == in_gallery_labels) & in_gal
    rank1_wrong_in_gal = (~rank1_correct) & in_gal

    parts = {
        'known_correct_accepted':             rank1_correct & accepted,
        'known_correct_rejected_silent':      rank1_correct & ~accepted,
        'known_wrong_accepted_misidentified': rank1_wrong_in_gal & accepted,
        'known_wrong_rejected_degraded':      rank1_wrong_in_gal & ~accepted,
        'unknown_rejected':                   ~in_gal & ~accepted,
        'unknown_spurious':                   ~in_gal & accepted,
    }
    counts = np.zeros(N, dtype=int)
    for m in parts.values():
        counts += m.astype(int)
    assert (counts == 1).all(), \
        f"partition not exclusive: counts min={counts.min()}, max={counts.max()}"
    return parts


# ═══════════════════════════════════════════════════════════════════════════
# Per-user statistics (for M9)
# ═══════════════════════════════════════════════════════════════════════════
def per_user_stats(pred_proxy, pred_ncm, ncm_top1, margin_ncm,
                    in_gallery_labels, tau, enrolled_ids, low_margin_mask):
    """Returns dict {user_id: {mismatch_rate, silent_rate, fnir-ish, tar_rate, low_margin_rate}}.
    Computed on known probes only (in_gallery_labels >= 0).
    """
    per_user = {}
    in_gal = in_gallery_labels >= 0
    accepted = ncm_top1 >= tau
    correct = (pred_ncm == in_gallery_labels) & in_gal

    for gallery_idx, uid in enumerate(enrolled_ids):
        user_mask = (in_gallery_labels == gallery_idx)
        n_u = int(user_mask.sum())
        if n_u == 0:
            continue
        mismatch_u = float((pred_proxy[user_mask] != pred_ncm[user_mask]).mean())
        silent_u = float((correct[user_mask] & ~accepted[user_mask]).mean())
        tar_u = float((correct[user_mask] & accepted[user_mask]).mean())
        fnir_u = float((~correct[user_mask] | ~accepted[user_mask]).mean())
        low_margin_u = float(low_margin_mask[user_mask].mean())
        per_user[uid] = {
            'n_probes': n_u,
            'mismatch_rate': mismatch_u,
            'silent_rate': silent_u,
            'tar_rate': tar_u,
            'fnir_rate': fnir_u,
            'low_margin_rate': low_margin_u,
        }
    return per_user


# ═══════════════════════════════════════════════════════════════════════════
# Per-checkpoint per-cohort run
# ═══════════════════════════════════════════════════════════════════════════
def run_for_checkpoint(
    ckpt_path: str,
    config,
    transform,
    device,
    cohort_filter: Optional[set] = None,
    cohort_name: str = "all_users",
    fixed_quantiles: Optional[Dict[str, float]] = None,
    fixed_tau: Optional[float] = None,
) -> Tuple[List[Dict], Dict, Dict]:
    """One run for (checkpoint, cohort).

    Returns:
      records: list of metric records (JSON schema)
      metadata: per-checkpoint metadata (counts, taus, etc.)
      per_user: per-user stats for this run (for M9 across-N analysis)
    """
    ckpt, model, projection, proxies, class_to_idx, class_means = \
        load_checkpoint_components(ckpt_path, device)
    experience_count = ckpt.get('experience_count', None)
    enrolled_ids = resolve_enrolled_ids(class_to_idx, class_means,
                                         expected_N=experience_count)
    N = len(enrolled_ids)

    proxy_indices = [class_to_idx[u] for u in enrolled_ids]
    proxies_active_raw = proxies[proxy_indices]
    proxies_active = F.normalize(proxies_active_raw, p=2, dim=1)
    class_means_active = torch.stack([class_means[u] for u in enrolled_ids], dim=0)

    # ── Feature extraction
    eval_paths, eval_labels = load_paths_labels_from_txt(
        str(config.dataset.eval_probe_file))
    eval_features = extract_and_project(model, projection, eval_paths, transform,
                                         device, config.dataset.channels)
    udev_paths, _ = load_paths_labels_from_txt(str(config.dataset.unknown_dev_file))
    udev_features = extract_and_project(model, projection, udev_paths, transform,
                                         device, config.dataset.channels)
    uext_features = None
    if config.dataset.unknown_test_file and os.path.exists(
            str(config.dataset.unknown_test_file)):
        uext_paths, _ = load_paths_labels_from_txt(
            str(config.dataset.unknown_test_file))
        uext_features = extract_and_project(model, projection, uext_paths,
                                             transform, device,
                                             config.dataset.channels)

    # ── Free GPU memory
    del model, projection, proxies, class_means
    torch.cuda.empty_cache() if device.type == 'cuda' else None

    # ── Split eval by gallery membership
    enrolled_set = set(enrolled_ids)
    gallery_idx_of = {u: i for i, u in enumerate(enrolled_ids)}

    eval_in_gallery_mask = np.array(
        [int(y) in enrolled_set for y in eval_labels], dtype=bool)
    in_gallery_labels_array = np.array([
        gallery_idx_of[int(y)] if int(y) in enrolled_set else -1
        for y in eval_labels
    ], dtype=np.int64)

    # ── Apply cohort filter
    if cohort_filter is not None:
        # early_user cohort: keep only known probes whose label ∈ cohort_filter
        # unknown probes (not in current enrolled) stay (used for unknown_*)
        cohort_label_mask = np.array([
            (int(y) in cohort_filter and int(y) in enrolled_set)
            or (int(y) not in enrolled_set)
            for y in eval_labels
        ], dtype=bool)
        eval_features = eval_features[cohort_label_mask]
        eval_labels = [eval_labels[i] for i in range(len(eval_labels))
                       if cohort_label_mask[i]]
        eval_in_gallery_mask = eval_in_gallery_mask[cohort_label_mask]
        in_gallery_labels_array = in_gallery_labels_array[cohort_label_mask]

    n_known = int(eval_in_gallery_mask.sum())
    n_future = int((~eval_in_gallery_mask).sum())
    n_external = int(uext_features.shape[0]) if uext_features is not None else 0

    # ── τ calibration
    tau_recal = calibrate_tau(udev_features, class_means_active, TARGET_FPIR)
    if fixed_tau is None:
        tau_fixed = tau_recal
    else:
        tau_fixed = fixed_tau
    rfpir_fixed = realized_fpir(udev_features, class_means_active, tau_fixed)
    rfpir_recal = realized_fpir(udev_features, class_means_active, tau_recal)

    # ── Scores
    eval_ncm = eval_features @ class_means_active.T
    eval_proxy = eval_features @ proxies_active.T
    udev_ncm_cohort = udev_features @ class_means_active.T
    eval_ncm_snorm = snorm(eval_ncm, udev_ncm_cohort)

    sorted_ncm, _ = torch.sort(eval_ncm, dim=1, descending=True)
    sorted_proxy, _ = torch.sort(eval_proxy, dim=1, descending=True)
    ncm_top1 = sorted_ncm[:, 0]
    ncm_top2 = sorted_ncm[:, 1] if sorted_ncm.shape[1] >= 2 else torch.zeros_like(ncm_top1)
    proxy_top1 = sorted_proxy[:, 0]
    proxy_top2 = sorted_proxy[:, 1] if sorted_proxy.shape[1] >= 2 else torch.zeros_like(proxy_top1)
    margin_ncm = (ncm_top1 - ncm_top2).cpu().numpy()
    margin_proxy = (proxy_top1 - proxy_top2).cpu().numpy()
    ncm_top1_np = ncm_top1.cpu().numpy()
    proxy_top1_np = proxy_top1.cpu().numpy()
    pred_ncm = eval_ncm.argmax(dim=1).cpu().numpy()
    pred_proxy = eval_proxy.argmax(dim=1).cpu().numpy()
    eval_ncm_snorm_np = eval_ncm_snorm.cpu().numpy()

    # ── Low margin
    if len(margin_ncm) > 0:
        median_margin_ncm = float(np.median(margin_ncm))
    else:
        median_margin_ncm = 0.0
    low_margin_thresh = LOW_MARGIN_FACTOR * median_margin_ncm
    low_margin_mask = margin_ncm < low_margin_thresh

    # ── Partitions (fixed + recal)
    parts_fixed = partition_6way(pred_ncm, ncm_top1_np, in_gallery_labels_array, tau_fixed)
    parts_recal = partition_6way(pred_ncm, ncm_top1_np, in_gallery_labels_array, tau_recal)

    # ── Quantiles for M3/M4/M12 (computed on known probes)
    known_idx = np.where(eval_in_gallery_mask)[0]
    if len(known_idx) >= MIN_N_FOR_SPEARMAN:
        dyn_q = {
            'margin_proxy_q75': float(np.quantile(margin_proxy[known_idx], 0.75)),
            'margin_proxy_q25': float(np.quantile(margin_proxy[known_idx], 0.25)),
            'margin_ncm_q75':   float(np.quantile(margin_ncm[known_idx], 0.75)),
            'margin_ncm_q25':   float(np.quantile(margin_ncm[known_idx], 0.25)),
            's_proxy_q75':      float(np.quantile(proxy_top1_np[known_idx], 0.75)),
            's_ncm_q75':        float(np.quantile(ncm_top1_np[known_idx], 0.75)),
            's_ncm_q25':        float(np.quantile(ncm_top1_np[known_idx], 0.25)),
        }
    else:
        dyn_q = None

    quantiles_to_use = fixed_quantiles if fixed_quantiles is not None else dyn_q

    # ── Build records
    records: List[Dict] = []

    def push(*, metric_name, value, n_samples, confidence_flag="ok",
             threshold_policy="N/A", unknown_source="N/A",
             partition="N/A", mask="none", quantile_basis="N/A", extra=None):
        rec = {
            "checkpoint_N": N,
            "cohort": cohort_name,
            "threshold_policy": threshold_policy,
            "unknown_source": unknown_source,
            "partition": partition,
            "mask": mask,
            "quantile_basis": quantile_basis,
            "metric_name": metric_name,
            "value": value,
            "n_samples": n_samples,
            "confidence_flag": confidence_flag,
        }
        if extra is not None:
            rec.update(extra)
        records.append(rec)

    def _flag(n):
        return "ok" if n >= MIN_N_FOR_OK else (
            "low_confidence" if n >= MIN_N_FOR_SPEARMAN else "insufficient_n")

    # ── M1: top1 disagreement
    for mask_name, mask_arr in [
        ("none", eval_in_gallery_mask),
        ("low_margin", eval_in_gallery_mask & low_margin_mask),
    ]:
        n_m = int(mask_arr.sum())
        if n_m > 0:
            v = float((pred_proxy[mask_arr] != pred_ncm[mask_arr]).mean())
        else:
            v = None
        push(metric_name="M1_top1_disagreement", value=v, n_samples=n_m,
             confidence_flag=_flag(n_m), partition="known_all", mask=mask_name)

    # ── M2: margin Spearman (4 subsets)
    subsets = [
        ("known_all", "none", eval_in_gallery_mask),
        ("known_all", "low_margin", eval_in_gallery_mask & low_margin_mask),
        ("known_correct_rejected_silent", "none", parts_fixed['known_correct_rejected_silent']),
        ("unknown_spurious", "none", parts_fixed['unknown_spurious']),
    ]
    for partition, mask_name, mask_arr in subsets:
        if mask_arr.sum() == 0:
            push(metric_name="M2_margin_spearman", value=None, n_samples=0,
                 confidence_flag="insufficient_n",
                 threshold_policy=("fixed_tau" if partition != "known_all" else "N/A"),
                 partition=partition, mask=mask_name)
            continue
        rho, n, flag = spearman_rho(margin_proxy[mask_arr], margin_ncm[mask_arr])
        push(metric_name="M2_margin_spearman", value=rho, n_samples=n,
             confidence_flag=flag,
             threshold_policy=("fixed_tau" if partition != "known_all" else "N/A"),
             partition=partition, mask=mask_name)

    # ── M3: proxy_safe_ncm_risk_rate (dynamic + fixedQ)
    if quantiles_to_use is not None and eval_in_gallery_mask.any():
        for basis_name, qs in [("dynamic", dyn_q), ("fixedQ", quantiles_to_use)]:
            if qs is None:
                continue
            mask = (margin_proxy >= qs['margin_proxy_q75']) & (margin_ncm <= qs['margin_ncm_q25']) & eval_in_gallery_mask
            v = float(mask.sum()) / max(int(eval_in_gallery_mask.sum()), 1)
            push(metric_name="M3_proxy_safe_ncm_risk_rate", value=v,
                 n_samples=int(eval_in_gallery_mask.sum()),
                 confidence_flag=_flag(int(eval_in_gallery_mask.sum())),
                 quantile_basis=basis_name, partition="known_all")

    # ── M4: ncm_silent_proxy_confident_rate (dynamic + fixedQ)
    if quantiles_to_use is not None:
        for policy, parts in [("fixed_tau", parts_fixed), ("recal_tau", parts_recal)]:
            silent_mask = parts['known_correct_rejected_silent']
            for basis_name, qs in [("dynamic", dyn_q), ("fixedQ", quantiles_to_use)]:
                if qs is None:
                    continue
                proxy_conf = (margin_proxy >= qs['margin_proxy_q75']) & (proxy_top1_np >= qs['s_proxy_q75'])
                pattern_b = silent_mask & proxy_conf
                v = float(pattern_b.sum()) / max(int(silent_mask.sum()), 1) if silent_mask.sum() > 0 else None
                push(metric_name="M4_silent_proxy_confident_rate", value=v,
                     n_samples=int(silent_mask.sum()),
                     confidence_flag=_flag(int(silent_mask.sum())),
                     threshold_policy=policy,
                     quantile_basis=basis_name,
                     partition="known_correct_rejected_silent")

    # ── M5: score_overlap_gap_main (NCM+S-norm, 3 unknown sources)
    rank1_correct_mask = (pred_ncm == in_gallery_labels_array) & eval_in_gallery_mask
    if rank1_correct_mask.any():
        genuine_snorm = eval_ncm_snorm_np[
            rank1_correct_mask, in_gallery_labels_array[rank1_correct_mask]]
    else:
        genuine_snorm = np.array([])

    # future
    if (~eval_in_gallery_mask).any():
        impostor_fut = eval_ncm_snorm_np[~eval_in_gallery_mask].max(axis=1)
        n_g, n_i = len(genuine_snorm), len(impostor_fut)
        if n_g >= 5 and n_i >= 5:
            v = float(np.percentile(genuine_snorm, 5) - np.percentile(impostor_fut, 99))
        else:
            v = None
        push(metric_name="M5_score_overlap_gap_main", value=v,
             n_samples=min(n_g, n_i), confidence_flag=_flag(min(n_g, n_i)),
             unknown_source="future", partition="known_vs_unknown")

    # external
    if uext_features is not None:
        uext_ncm = uext_features @ class_means_active.T
        uext_snorm = snorm(uext_ncm, udev_ncm_cohort)
        impostor_ext = uext_snorm.max(dim=1).values.cpu().numpy()
        n_g, n_i = len(genuine_snorm), len(impostor_ext)
        if n_g >= 5 and n_i >= 5:
            v = float(np.percentile(genuine_snorm, 5) - np.percentile(impostor_ext, 99))
        else:
            v = None
        push(metric_name="M5_score_overlap_gap_main", value=v,
             n_samples=min(n_g, n_i), confidence_flag=_flag(min(n_g, n_i)),
             unknown_source="external", partition="known_vs_unknown")
        # all
        impostor_all = np.concatenate([impostor_fut, impostor_ext]) \
            if (~eval_in_gallery_mask).any() else impostor_ext
        n_g, n_i = len(genuine_snorm), len(impostor_all)
        if n_g >= 5 and n_i >= 5:
            v = float(np.percentile(genuine_snorm, 5) - np.percentile(impostor_all, 99))
        else:
            v = None
        push(metric_name="M5_score_overlap_gap_main", value=v,
             n_samples=min(n_g, n_i), confidence_flag=_flag(min(n_g, n_i)),
             unknown_source="all", partition="known_vs_unknown")

    # ── M5b: aux raw NCM, raw proxy
    if rank1_correct_mask.any():
        eval_ncm_np = eval_ncm.cpu().numpy()
        eval_proxy_np = eval_proxy.cpu().numpy()
        genuine_raw_ncm = eval_ncm_np[
            rank1_correct_mask, in_gallery_labels_array[rank1_correct_mask]]
        genuine_raw_proxy = eval_proxy_np[
            rank1_correct_mask, in_gallery_labels_array[rank1_correct_mask]]
        if (~eval_in_gallery_mask).any():
            impostor_raw_ncm = eval_ncm_np[~eval_in_gallery_mask].max(axis=1)
            impostor_raw_proxy = eval_proxy_np[~eval_in_gallery_mask].max(axis=1)
            for tag, gen, imp in [
                ("rawNCM", genuine_raw_ncm, impostor_raw_ncm),
                ("rawProxy", genuine_raw_proxy, impostor_raw_proxy),
            ]:
                n_g, n_i = len(gen), len(imp)
                if n_g >= 5 and n_i >= 5:
                    v = float(np.percentile(gen, 5) - np.percentile(imp, 99))
                else:
                    v = None
                push(metric_name=f"M5b_overlap_gap_{tag}", value=v,
                     n_samples=min(n_g, n_i),
                     confidence_flag=_flag(min(n_g, n_i)),
                     unknown_source="future", partition="known_vs_unknown")

    # ── M6: silent_rate + realized_FPIR + tau (fixed + recal)
    known_rank1_correct = rank1_correct_mask[eval_in_gallery_mask]
    known_top1 = ncm_top1_np[eval_in_gallery_mask]
    for policy, tau, rfpir in [
        ("fixed_tau", tau_fixed, rfpir_fixed),
        ("recal_tau", tau_recal, rfpir_recal),
    ]:
        n_k = len(known_top1)
        if n_k > 0:
            silent_rate = float((known_rank1_correct & (known_top1 < tau)).mean())
        else:
            silent_rate = None
        push(metric_name="M6_silent_rate", value=silent_rate, n_samples=n_k,
             confidence_flag=_flag(n_k),
             threshold_policy=policy, partition="known_all")
        push(metric_name="M6_realized_FPIR", value=rfpir,
             n_samples=int(udev_features.shape[0]),
             threshold_policy=policy, unknown_source="dev")
        push(metric_name="M6_tau", value=float(tau),
             n_samples=int(udev_features.shape[0]),
             threshold_policy=policy)

    # ── M7: FNIR @ 1% FPIR (fixed + recal)
    if n_k > 0:
        for policy, tau in [("fixed_tau", tau_fixed), ("recal_tau", tau_recal)]:
            tp = ((known_rank1_correct) & (known_top1 >= tau)).sum()
            fnir = float(1.0 - tp / n_k)
            push(metric_name="M7_FNIR_at_1pct_FPIR", value=fnir, n_samples=n_k,
                 threshold_policy=policy, partition="known_all")

    # ── M8: spurious + concentration (3 unknown × 2 tau)
    def _m8(unknown_pred, unknown_top1, tau, source_name, policy):
        n_un = len(unknown_top1)
        if n_un == 0:
            return
        accepted_mask_un = unknown_top1 >= tau
        sp_count = int(accepted_mask_un.sum())
        sp_rate = float(sp_count / n_un)
        push(metric_name="M8_spurious_accept_rate", value=sp_rate, n_samples=n_un,
             threshold_policy=policy, unknown_source=source_name)
        if sp_count == 0:
            for mk in ["false_accept_entropy_raw", "false_accept_entropy_norm",
                       "top_spurious_user_share", "spurious_user_gini"]:
                push(metric_name=f"M8_{mk}", value=None, n_samples=0,
                     confidence_flag="insufficient_n",
                     threshold_policy=policy, unknown_source=source_name)
            return
        accepted_users = unknown_pred[accepted_mask_un]
        user_counts = np.bincount(accepted_users, minlength=N)
        p = user_counts / max(user_counts.sum(), 1)
        p_pos = p[p > 0]
        H_raw = float(-(p_pos * np.log(p_pos)).sum())
        H_norm = float(H_raw / np.log(N)) if N > 1 else None
        sorted_p = np.sort(p)
        n_c = len(p)
        if sorted_p.sum() > 0:
            gini = float((2 * np.sum((np.arange(1, n_c + 1)) * sorted_p)) /
                          (n_c * sorted_p.sum()) - (n_c + 1) / n_c)
        else:
            gini = None
        push(metric_name="M8_false_accept_entropy_raw", value=H_raw,
             n_samples=sp_count, threshold_policy=policy, unknown_source=source_name)
        push(metric_name="M8_false_accept_entropy_norm", value=H_norm,
             n_samples=sp_count, threshold_policy=policy, unknown_source=source_name)
        push(metric_name="M8_top_spurious_user_share", value=float(p.max()),
             n_samples=sp_count, threshold_policy=policy, unknown_source=source_name)
        push(metric_name="M8_spurious_user_gini", value=gini,
             n_samples=sp_count, threshold_policy=policy, unknown_source=source_name)

    # future
    fut_mask = ~eval_in_gallery_mask
    if fut_mask.any():
        for policy, tau in [("fixed_tau", tau_fixed), ("recal_tau", tau_recal)]:
            _m8(pred_ncm[fut_mask], ncm_top1_np[fut_mask], tau, "future", policy)

    # external
    if uext_features is not None:
        uext_ncm = uext_features @ class_means_active.T
        uext_top1 = uext_ncm.max(dim=1).values.cpu().numpy()
        uext_argmax = uext_ncm.argmax(dim=1).cpu().numpy()
        for policy, tau in [("fixed_tau", tau_fixed), ("recal_tau", tau_recal)]:
            _m8(uext_argmax, uext_top1, tau, "external", policy)
        # all
        if fut_mask.any():
            all_pred = np.concatenate([pred_ncm[fut_mask], uext_argmax])
            all_top1 = np.concatenate([ncm_top1_np[fut_mask], uext_top1])
            for policy, tau in [("fixed_tau", tau_fixed), ("recal_tau", tau_recal)]:
                _m8(all_pred, all_top1, tau, "all", policy)

    # ── M9: per-user stats (returned for cross-N analysis later)
    per_user = per_user_stats(pred_proxy, pred_ncm, ncm_top1_np, margin_ncm,
                               in_gallery_labels_array, tau_fixed, enrolled_ids,
                               low_margin_mask)

    # M9 immediate: Spearman(mismatch, silent/FNIR/TAR) at this checkpoint
    if len(per_user) >= MIN_N_FOR_SPEARMAN:
        users = list(per_user.keys())
        mm = np.array([per_user[u]['mismatch_rate'] for u in users])
        sl = np.array([per_user[u]['silent_rate'] for u in users])
        fn = np.array([per_user[u]['fnir_rate'] for u in users])
        ta = np.array([per_user[u]['tar_rate'] for u in users])
        lm = np.array([per_user[u]['low_margin_rate'] for u in users])
        for tgt_name, tgt in [("silent", sl), ("FNIR", fn), ("TAR", ta), ("low_margin", lm)]:
            rho, n, flag, ci_lo, ci_hi = spearman_with_ci(mm, tgt)
            extra = {'ci_lo': ci_lo, 'ci_hi': ci_hi} if ci_lo is not None else None
            push(metric_name=f"M9_per_user_corr_mismatch_vs_{tgt_name}",
                 value=rho, n_samples=n, confidence_flag=flag,
                 threshold_policy="fixed_tau", extra=extra)

    # ── M10: proxy_ncm_margin_gap
    if eval_in_gallery_mask.any():
        gap = margin_proxy[eval_in_gallery_mask] - margin_ncm[eval_in_gallery_mask]
        push(metric_name="M10_margin_gap_mean", value=float(gap.mean()),
             n_samples=len(gap), partition="known_all")
        push(metric_name="M10_margin_gap_p95", value=float(np.percentile(gap, 95)),
             n_samples=len(gap), partition="known_all")
        lm_known = eval_in_gallery_mask & low_margin_mask
        if lm_known.any():
            gap_lm = margin_proxy[lm_known] - margin_ncm[lm_known]
            push(metric_name="M10_margin_gap_mean", value=float(gap_lm.mean()),
                 n_samples=len(gap_lm), partition="known_all", mask="low_margin")

    # ── M11: proxy_score_ncm_score_spearman
    if eval_in_gallery_mask.any():
        rho, n, flag = spearman_rho(
            proxy_top1_np[eval_in_gallery_mask], ncm_top1_np[eval_in_gallery_mask])
        push(metric_name="M11_proxy_score_ncm_score_spearman",
             value=rho, n_samples=n, confidence_flag=flag, partition="known_all")

    # ── M12: proxy_high_ncm_low_rate (dynamic + fixedQ)
    if quantiles_to_use is not None and eval_in_gallery_mask.any():
        for basis_name, qs in [("dynamic", dyn_q), ("fixedQ", quantiles_to_use)]:
            if qs is None:
                continue
            mask = (proxy_top1_np >= qs['s_proxy_q75']) & (ncm_top1_np <= qs['s_ncm_q25']) & eval_in_gallery_mask
            v = float(mask.sum()) / max(int(eval_in_gallery_mask.sum()), 1)
            push(metric_name="M12_proxy_high_ncm_low_rate", value=v,
                 n_samples=int(eval_in_gallery_mask.sum()),
                 quantile_basis=basis_name, partition="known_all")

    # ── Metadata
    metadata = {
        'checkpoint': ckpt_path,
        'cohort': cohort_name,
        'N_enrolled': N,
        'experience_count': experience_count,
        'n_known_probes': n_known,
        'n_future_probes': n_future,
        'n_external_probes': n_external,
        'tau_fixed': tau_fixed,
        'tau_recal': tau_recal,
        'realized_fpir_fixed': rfpir_fixed,
        'realized_fpir_recal': rfpir_recal,
        'median_margin_ncm': median_margin_ncm,
        'low_margin_threshold': low_margin_thresh,
        'partition_counts_fixed_tau': {k: int(v.sum()) for k, v in parts_fixed.items()},
        'partition_counts_recal_tau': {k: int(v.sum()) for k, v in parts_recal.items()},
        'dynamic_quantiles': dyn_q,
    }
    return records, metadata, per_user


# ═══════════════════════════════════════════════════════════════════════════
# Lagged temporal association (M13/M14)
# ═══════════════════════════════════════════════════════════════════════════
def compute_lagged(all_records: List[Dict], N_list: List[int]) -> List[Dict]:
    """M13: lagged(mismatch_t, silent_t+1).
       M14: lagged(mismatch_t, FNIR_t+1).
       n = len(N_list) - 1 (= 4 for [10,20,30,40,50]).
       DESCRIPTIVE TEMPORAL ASSOCIATION ONLY — NOT causal proof.
    """
    def _extract(metric_name, threshold_policy=None, cohort="all_users"):
        out = {}
        for r in all_records:
            if r['cohort'] != cohort:
                continue
            if r['metric_name'] != metric_name:
                continue
            if threshold_policy and r['threshold_policy'] != threshold_policy:
                continue
            out[r['checkpoint_N']] = r['value']
        return out

    # mismatch metric: M1 disagreement (known_all, mask=none)
    mismatch = {}
    for r in all_records:
        if (r['cohort'] == 'all_users' and r['metric_name'] == 'M1_top1_disagreement'
                and r['partition'] == 'known_all' and r['mask'] == 'none'):
            mismatch[r['checkpoint_N']] = r['value']
    silent_fixed = _extract('M6_silent_rate', 'fixed_tau')
    fnir_fixed = _extract('M7_FNIR_at_1pct_FPIR', 'fixed_tau')

    lagged_records = []
    sorted_N = sorted(N_list)
    for tgt_name, tgt_dict in [("silent", silent_fixed), ("FNIR", fnir_fixed)]:
        pairs_x, pairs_y = [], []
        for i in range(len(sorted_N) - 1):
            n_t, n_t1 = sorted_N[i], sorted_N[i + 1]
            if (mismatch.get(n_t) is not None
                    and tgt_dict.get(n_t1) is not None):
                pairs_x.append(mismatch[n_t])
                pairs_y.append(tgt_dict[n_t1])
        if len(pairs_x) >= MIN_N_FOR_SPEARMAN:
            rho, n, flag = spearman_rho(np.array(pairs_x), np.array(pairs_y))
        else:
            rho, n, flag = None, len(pairs_x), "insufficient_n"
        metric_id = "M13_lagged_assoc_mismatch_silent" if tgt_name == "silent" else "M14_lagged_assoc_mismatch_FNIR"
        lagged_records.append({
            "checkpoint_N": None, "cohort": "all_users",
            "threshold_policy": "fixed_tau", "unknown_source": "N/A",
            "partition": "known_all", "mask": "none",
            "quantile_basis": "N/A",
            "metric_name": metric_id, "value": rho, "n_samples": n,
            "confidence_flag": flag,
            "_note": "Descriptive temporal association only — NOT causal proof (n=4)",
        })
    return lagged_records


# ═══════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════
def plot_curves(all_records, output_path, title, metric_filter):
    """Plot metric vs N for both cohorts, multiple metrics."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  [WARN] matplotlib unavailable, skip plot {output_path}")
        return

    n_metrics = len(metric_filter)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes = np.atleast_2d(axes).flatten()
    for ax_idx, (label, criteria) in enumerate(metric_filter.items()):
        ax = axes[ax_idx]
        for cohort, color in [("all_users", "C0"), ("early_user", "C1")]:
            xs, ys = [], []
            for r in all_records:
                if r['cohort'] != cohort:
                    continue
                ok = True
                for k, v in criteria.items():
                    if r.get(k) != v:
                        ok = False
                        break
                if not ok:
                    continue
                if r['checkpoint_N'] is None or r['value'] is None:
                    continue
                xs.append(r['checkpoint_N'])
                ys.append(r['value'])
            if xs:
                order = np.argsort(xs)
                xs_sorted = np.array(xs)[order]
                ys_sorted = np.array(ys)[order]
                ax.plot(xs_sorted, ys_sorted, marker='o', color=color, label=cohort)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("N")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axes[n_metrics:]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    print(f"  saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Verdict
# ═══════════════════════════════════════════════════════════════════════════
def render_verdict(all_records, per_user_history):
    """Apply GPT 3-5차/8차 판정 기준 (M13/M14 제외, M2 subset 반영)."""
    def get_series(metric_name, cohort, **filters):
        out = {}
        for r in all_records:
            if r['cohort'] != cohort or r['metric_name'] != metric_name:
                continue
            ok = True
            for k, v in filters.items():
                if r.get(k) != v:
                    ok = False; break
            if not ok:
                continue
            if r['checkpoint_N'] is None or r['value'] is None:
                continue
            out[r['checkpoint_N']] = r['value']
        return out

    # M1 known_all mask=none
    m1_all = get_series('M1_top1_disagreement', 'all_users',
                        partition='known_all', mask='none')
    m1_early = get_series('M1_top1_disagreement', 'early_user',
                          partition='known_all', mask='none')

    # M3 fixedQ
    m3_all = get_series('M3_proxy_safe_ncm_risk_rate', 'all_users',
                        quantile_basis='fixedQ')
    m3_early = get_series('M3_proxy_safe_ncm_risk_rate', 'early_user',
                          quantile_basis='fixedQ')

    # M2 (all + low_margin + silent)
    def _last_m2(filters):
        latest_N = -1
        latest = None
        for r in all_records:
            if r['cohort'] != 'all_users' or r['metric_name'] != 'M2_margin_spearman':
                continue
            ok = True
            for k, v in filters.items():
                if r.get(k) != v: ok=False; break
            if not ok or r['value'] is None:
                continue
            if r['checkpoint_N'] is not None and r['checkpoint_N'] > latest_N:
                latest_N = r['checkpoint_N']
                latest = r['value']
        return latest

    m2_all = _last_m2({'partition': 'known_all', 'mask': 'none'})
    m2_low_margin = _last_m2({'partition': 'known_all', 'mask': 'low_margin'})
    m2_silent = _last_m2({'partition': 'known_correct_rejected_silent'})

    # M9 (mismatch vs silent / FNIR) — last checkpoint
    def _last_m9(target):
        latest_N = -1
        latest = None
        for r in all_records:
            if r['cohort'] != 'all_users':
                continue
            if r['metric_name'] != f'M9_per_user_corr_mismatch_vs_{target}':
                continue
            if r['value'] is None:
                continue
            if r['checkpoint_N'] is not None and r['checkpoint_N'] > latest_N:
                latest_N = r['checkpoint_N']
                latest = r['value']
        return latest

    m9_silent = _last_m9('silent')
    m9_fnir = _last_m9('FNIR')

    # Strong conditions:
    # 1. M1 increases with N for BOTH cohorts
    # 2. |M9 corr| ≥ 0.4 (mismatch_silent OR mismatch_FNIR)
    # 3. high-mismatch user group silent_rate ≥ 1.5× low-mismatch group
    # 4. M2_low_margin < 0.6 OR M2_known_silent < 0.6 (boundary subset signal)
    # M2_all ≥ 0.85 + low_margin/silent ≥ 0.6 → auto-demote to Weak

    def is_increasing(d):
        if not d:
            return False
        xs = sorted(d.keys())
        ys = [d[x] for x in xs]
        # at least 2 monotonic increases
        rises = sum(1 for i in range(1, len(ys)) if ys[i] > ys[i-1])
        return rises >= max(1, len(ys) - 2)

    cond_1 = is_increasing(m1_all) and is_increasing(m1_early)
    cond_2 = (m9_silent is not None and abs(m9_silent) >= 0.4) or \
             (m9_fnir is not None and abs(m9_fnir) >= 0.4)

    # cond_3: per-user mismatch high vs low quartile
    cond_3 = False
    if len(per_user_history) > 0:
        # use last checkpoint per_user
        last_N = max(per_user_history.keys())
        per_u = per_user_history[last_N]
        if len(per_u) >= 8:
            mm = np.array([v['mismatch_rate'] for v in per_u.values()])
            sl = np.array([v['silent_rate'] for v in per_u.values()])
            q75 = np.quantile(mm, 0.75)
            q25 = np.quantile(mm, 0.25)
            high_silent = sl[mm >= q75].mean() if (mm >= q75).any() else 0
            low_silent = sl[mm <= q25].mean() if (mm <= q25).any() else 0
            if low_silent > 1e-6:
                cond_3 = (high_silent / low_silent) >= 1.5

    cond_4 = (m2_low_margin is not None and m2_low_margin < 0.6) or \
             (m2_silent is not None and m2_silent < 0.6)

    # Auto-demote
    auto_demote = (m2_all is not None and m2_all >= 0.85
                   and (m2_low_margin is None or m2_low_margin >= 0.6)
                   and (m2_silent is None or m2_silent >= 0.6))

    n_strong = sum([cond_1, cond_2, cond_3, cond_4])

    if auto_demote:
        action = "WEAK_AUTODEMOTED_BY_M2"
        rationale = (f"M2_all={m2_all:.3f}≥0.85, M2_low_margin={m2_low_margin}, "
                     f"M2_silent={m2_silent} — mismatch contribution limited.")
    elif n_strong >= 3:
        action = "STRONG_NA_QAR_DESIGN"
        rationale = (f"{n_strong}/4 strong conditions met. Sequential mismatch "
                     f"linked to silent/FNIR. Proceed to NA-QAR v0 design (not training).")
    elif n_strong >= 2:
        action = "MARGINAL_CONSERVATIVE"
        rationale = (f"{n_strong}/4 strong conditions. Proceed to 1.6 GHOST pilot.")
    else:
        action = "WEAK_PROJ512_LEGACY_TERMINATE"
        rationale = (f"{n_strong}/4 strong conditions. Mismatch not linked to "
                     f"sequential degradation. Terminate Phase −1B, proceed Tier 1/multi-seed.")

    return {
        'action': action,
        'rationale': rationale,
        'conditions': {
            'cond_1_M1_increases_both_cohorts': cond_1,
            'cond_2_M9_corr_geq_0p4': cond_2,
            'cond_3_high_mismatch_1p5x_silent': cond_3,
            'cond_4_M2_boundary_subset_lt_0p6': cond_4,
            'auto_demote_by_M2_all': auto_demote,
        },
        'evidence': {
            'M1_all_series': m1_all,
            'M1_early_series': m1_early,
            'M3_fixedQ_all': m3_all,
            'M3_fixedQ_early': m3_early,
            'M2_all': m2_all,
            'M2_low_margin': m2_low_margin,
            'M2_known_silent': m2_silent,
            'M9_corr_silent': m9_silent,
            'M9_corr_FNIR': m9_fnir,
        }
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt-dir', required=True,
                        help='Directory containing checkpoint_exp_10.pth ... _50.pth')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--checkpoint-Ns', type=str, default="10,20,30,40,50",
                        help='Comma-separated N values')
    args = parser.parse_args()

    config = ConfigParser(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[1.8 Stage 2] device: {device}")
    print(f"[1.8 Stage 2] checkpoint dir: {args.ckpt_dir}")

    N_list = [int(x) for x in args.checkpoint_Ns.split(',')]
    print(f"[1.8 Stage 2] N values: {N_list}")

    transform = get_scr_transforms(
        train=False, imside=config.dataset.height,
        channels=config.dataset.channels)

    ckpt_paths = {N: os.path.join(args.ckpt_dir, f"checkpoint_exp_{N}.pth")
                  for N in N_list}
    for N, p in ckpt_paths.items():
        assert os.path.exists(p), f"checkpoint missing: {p}"

    all_records: List[Dict] = []
    all_metadata: Dict[int, Dict] = {}
    per_user_history: Dict[int, Dict] = {}

    # ── First pass: N=10 (canonical reference for fixed_tau + fixed_quantiles + enrolled_ids_10)
    print(f"\n{'=' * 70}\n[N=10] all_users (canonical reference)\n{'=' * 70}")
    recs_10, meta_10, per_user_10 = run_for_checkpoint(
        ckpt_paths[N_list[0]], config, transform, device,
        cohort_filter=None, cohort_name="all_users",
        fixed_quantiles=None, fixed_tau=None)
    all_records.extend(recs_10)
    all_metadata[N_list[0]] = {'all_users': meta_10}
    per_user_history[N_list[0]] = per_user_10
    fixed_tau = meta_10['tau_fixed']  # = tau_recal at N=10
    fixed_quantiles = meta_10['dynamic_quantiles']
    enrolled_ids_10_path = ckpt_paths[N_list[0]]
    print(f"  fixed_tau = {fixed_tau:.4f}")

    # Get enrolled_ids_10 (for early_user cohort)
    ckpt0 = torch.load(ckpt_paths[N_list[0]], map_location='cpu', weights_only=False)
    class_to_idx_0 = {int(k): int(v) for k, v in ckpt0['proxy_anchor_data']['class_to_idx'].items()}
    class_means_0 = ckpt0['ncm_state_dict']['class_means']
    enrolled_ids_10 = resolve_enrolled_ids(class_to_idx_0, class_means_0)
    del ckpt0, class_to_idx_0, class_means_0
    print(f"  enrolled_ids_10: {len(enrolled_ids_10)} users")

    # ── N=10 early_user cohort (= all_users at this N, but run for consistency)
    print(f"\n{'=' * 70}\n[N=10] early_user\n{'=' * 70}")
    recs_10e, meta_10e, per_user_10e = run_for_checkpoint(
        ckpt_paths[N_list[0]], config, transform, device,
        cohort_filter=set(enrolled_ids_10), cohort_name="early_user",
        fixed_quantiles=fixed_quantiles, fixed_tau=fixed_tau)
    all_records.extend(recs_10e)
    all_metadata[N_list[0]]['early_user'] = meta_10e

    # ── Remaining checkpoints
    for N in N_list[1:]:
        print(f"\n{'=' * 70}\n[N={N}] all_users\n{'=' * 70}")
        recs, meta, per_user_N = run_for_checkpoint(
            ckpt_paths[N], config, transform, device,
            cohort_filter=None, cohort_name="all_users",
            fixed_quantiles=fixed_quantiles, fixed_tau=fixed_tau)
        all_records.extend(recs)
        all_metadata[N] = {'all_users': meta}
        per_user_history[N] = per_user_N

        print(f"\n{'=' * 70}\n[N={N}] early_user\n{'=' * 70}")
        recs_e, meta_e, _ = run_for_checkpoint(
            ckpt_paths[N], config, transform, device,
            cohort_filter=set(enrolled_ids_10), cohort_name="early_user",
            fixed_quantiles=fixed_quantiles, fixed_tau=fixed_tau)
        all_records.extend(recs_e)
        all_metadata[N]['early_user'] = meta_e

    # ── M13/M14 lagged
    lagged = compute_lagged(all_records, N_list)
    all_records.extend(lagged)

    # ── Verdict
    verdict = render_verdict(all_records, per_user_history)

    # ── Save JSON
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "proxy_ncm_mismatch.json")
    nan_count = sum(1 for r in all_records
                    if r.get('value') is None
                    or (isinstance(r.get('value'), float) and np.isnan(r['value'])))
    output = {
        'stage': 2,
        'config': args.config,
        'ckpt_dir': args.ckpt_dir,
        'N_list': N_list,
        'fixed_tau': fixed_tau,
        'fixed_quantiles': fixed_quantiles,
        'enrolled_ids_10': enrolled_ids_10,
        'metadata_by_N': all_metadata,
        'records': all_records,
        'lagged_records_note': "M13/M14 are descriptive temporal associations only (n=4); not causal proof",
        'total_records': len(all_records),
        'null_record_count': nan_count,
        'verdict': verdict,
    }
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\n[1.8 Stage 2] saved: {json_path}")

    # ── Plots
    plot_curves(all_records,
                os.path.join(args.output_dir, "mismatch_main_curves.png"),
                "Proxy-NCM Mismatch — Main Metrics (M1/M3/M4/M5/M6/M7)",
                {
                    "M1 disagreement (known)": {
                        'metric_name': 'M1_top1_disagreement',
                        'partition': 'known_all', 'mask': 'none'},
                    "M3 proxy-safe-ncm-risk (fixedQ)": {
                        'metric_name': 'M3_proxy_safe_ncm_risk_rate',
                        'quantile_basis': 'fixedQ'},
                    "M4 silent ∩ proxy_confident (fixedQ, fixed_tau)": {
                        'metric_name': 'M4_silent_proxy_confident_rate',
                        'threshold_policy': 'fixed_tau',
                        'quantile_basis': 'fixedQ'},
                    "M5 overlap_gap (snorm, all unknown)": {
                        'metric_name': 'M5_score_overlap_gap_main',
                        'unknown_source': 'all'},
                    "M6 silent_rate (fixed_tau)": {
                        'metric_name': 'M6_silent_rate',
                        'threshold_policy': 'fixed_tau'},
                    "M7 FNIR (fixed_tau)": {
                        'metric_name': 'M7_FNIR_at_1pct_FPIR',
                        'threshold_policy': 'fixed_tau'},
                })

    plot_curves(all_records,
                os.path.join(args.output_dir, "mismatch_aux_curves.png"),
                "Proxy-NCM Mismatch — Aux Metrics (M8/M10/M11/M12)",
                {
                    "M8 spurious_accept (fixed_tau, all)": {
                        'metric_name': 'M8_spurious_accept_rate',
                        'threshold_policy': 'fixed_tau',
                        'unknown_source': 'all'},
                    "M8 false_accept_entropy_norm (fixed_tau, all)": {
                        'metric_name': 'M8_false_accept_entropy_norm',
                        'threshold_policy': 'fixed_tau',
                        'unknown_source': 'all'},
                    "M10 margin_gap_mean (known)": {
                        'metric_name': 'M10_margin_gap_mean',
                        'partition': 'known_all', 'mask': 'none'},
                    "M11 score Spearman (known)": {
                        'metric_name': 'M11_proxy_score_ncm_score_spearman',
                        'partition': 'known_all'},
                    "M12 proxy_high_ncm_low (fixedQ)": {
                        'metric_name': 'M12_proxy_high_ncm_low_rate',
                        'quantile_basis': 'fixedQ'},
                })

    # ── Print summary
    print(f"\n{'=' * 70}")
    print(f"=== Sweep Verdict ===")
    print(f"  ACTION: {verdict['action']}")
    print(f"  rationale: {verdict['rationale']}")
    print(f"\n  Conditions met:")
    for k, v in verdict['conditions'].items():
        print(f"    {k}: {v}")
    print(f"\n  Evidence summary:")
    print(f"    M1_all_series: {verdict['evidence']['M1_all_series']}")
    print(f"    M1_early_series: {verdict['evidence']['M1_early_series']}")
    print(f"    M2_all: {verdict['evidence']['M2_all']}")
    print(f"    M2_low_margin: {verdict['evidence']['M2_low_margin']}")
    print(f"    M2_known_silent: {verdict['evidence']['M2_known_silent']}")
    print(f"    M9 mismatch↔silent: {verdict['evidence']['M9_corr_silent']}")
    print(f"    M9 mismatch↔FNIR: {verdict['evidence']['M9_corr_FNIR']}")
    print(f"\n  Total records: {len(all_records)}, null/NaN: {nan_count}")


if __name__ == '__main__':
    main()
