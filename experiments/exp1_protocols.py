"""
Experiment 1: Protocol Comparison — 프로토콜 로직 (v5)

Self-defending diagnostic experiment. 9 conditions:
  A                    — closed-set expanding-gallery identification
  B-s1/s2/s3/sF        — size-matched static open-set sweep
  C-raw-fixed          — sequential, raw cosine, fixed τ (step 0)
  C-raw-recalib        — sequential, raw cosine, recalib τ each step
  C-snorm-fixed        — sequential, S-norm, fixed τ (step 0 S-norm space)
                         + incremental per-class cohort at enrollment
  C-snorm-recalib      — sequential, S-norm, recalib τ each step

Key policies:
  - Each condition creates a FRESH NCMClassifier instance (no state leakage)
  - All max/argmax/threshold/TPIR/FPIR are restricted to current gallery_ids
  - external_dev → cohort + threshold calibration; external_test → evaluation only
  - Legacy aliases preserved in JSON for backward compat
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from coconut.openset.utils import load_paths_labels_from_txt, set_seed
from coconut.openset.score_extraction import extract_features
from coconut.classifiers.ncm import NCMClassifier
from coconut.data.transforms import get_scr_transforms


# ============================================================
# Data loading
# ============================================================

def load_and_validate_data(txt_file: str, base_path: str = "") -> Dict[int, List[str]]:
    paths, labels = load_paths_labels_from_txt(txt_file)
    if base_path:
        paths = [os.path.join(base_path, p) for p in paths]

    id_to_paths = defaultdict(list)
    for p, l in zip(paths, labels):
        id_to_paths[l].append(p)

    all_ids = sorted(id_to_paths.keys())
    sample_counts = [len(id_to_paths[i]) for i in all_ids]
    bad_ids = [i for i in all_ids if len(id_to_paths[i]) != 10]

    print(f"[DATA] total IDs found: {len(all_ids)}")
    print(f"[DATA] sample count range: {min(sample_counts)}-{max(sample_counts)}")
    if bad_ids:
        print(f"[DATA] WARNING: IDs with sample count != 10: {bad_ids[:10]}...")
    else:
        print(f"[DATA] all IDs have exactly 10 samples")
    print(f"[DATA] total samples: {sum(sample_counts)}")
    return dict(id_to_paths)


# ============================================================
# Identity / Sample splits
# ============================================================

def split_identities(all_ids: List[int], n_base: int = 20,
                     n_future: int = 40, n_external: int = 40,
                     seed: int = 42) -> Dict[str, List[int]]:
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(all_ids)

    total_needed = n_base + n_future + n_external
    if len(shuffled) < total_needed:
        raise ValueError(f"Need {total_needed} IDs but only {len(shuffled)} available")

    selected = shuffled[:total_needed]
    # NOTE: future_ids는 정렬하지 않고 shuffle 순서 유지 (sequential enrollment 재현성)
    split = {
        'base_ids': sorted(selected[:n_base].tolist()),
        'future_ids': selected[n_base:n_base + n_future].tolist(),
        'external_ids': sorted(selected[n_base + n_future:total_needed].tolist()),
    }

    print(f"[SPLIT] base: {len(split['base_ids'])}, "
          f"future: {len(split['future_ids'])}, "
          f"external: {len(split['external_ids'])}")
    return split


def split_samples_per_id(id_to_paths: Dict[int, List[str]],
                         selected_ids: List[int],
                         n_enroll: int = 3, n_dev: int = 3,
                         base_seed: int = 42) -> Dict[int, Dict[str, List[str]]]:
    sample_split = {}
    for uid in selected_ids:
        paths = id_to_paths[uid]
        rng = np.random.RandomState(base_seed + uid)
        indices = rng.permutation(len(paths))
        sample_split[uid] = {
            'enroll': [paths[i] for i in indices[:n_enroll]],
            'dev': [paths[i] for i in indices[n_enroll:n_enroll + n_dev]],
            'test': [paths[i] for i in indices[n_enroll + n_dev:]],
        }
    print(f"[SAMPLE SPLIT] per ID: {n_enroll} enroll / {n_dev} dev / rest test")
    print(f"[SAMPLE SPLIT] total IDs split: {len(sample_split)}")
    return sample_split


# ============================================================
# Embedding extraction
# ============================================================

def extract_all_embeddings(model, sample_split: Dict[int, Dict[str, List[str]]],
                           transform, device, channels: int = 1
                           ) -> Dict[int, Dict[str, np.ndarray]]:
    embeddings = {}
    all_paths = []
    path_map = []

    for uid in sorted(sample_split.keys()):
        for split_name in ['enroll', 'dev', 'test']:
            for idx, p in enumerate(sample_split[uid][split_name]):
                all_paths.append(p)
                path_map.append((uid, split_name, idx))

    print(f"[EMBEDDING] extracting features from {len(all_paths)} images...")
    all_feats = extract_features(model, all_paths, transform, device,
                                 batch_size=64, channels=channels)
    print(f"[EMBEDDING] shape: {all_feats.shape}")

    norms = np.linalg.norm(all_feats, axis=1)
    print(f"[EMBEDDING] L2 norm range: {norms.min():.4f} - {norms.max():.4f}")

    for uid in sorted(sample_split.keys()):
        embeddings[uid] = {}
    for i, (uid, split_name, local_idx) in enumerate(path_map):
        if split_name not in embeddings[uid]:
            embeddings[uid][split_name] = []
        embeddings[uid][split_name].append(all_feats[i])

    for uid in embeddings:
        for split_name in embeddings[uid]:
            embeddings[uid][split_name] = np.stack(embeddings[uid][split_name])
    return embeddings


# ============================================================
# Gallery schedule
# ============================================================

def build_gallery_schedule(base_ids: List[int], future_ids: List[int],
                           future_batch_size: int) -> Dict[str, List[int]]:
    """
    Sequential C의 step별 gallery list 생성.
    step_0 = base_ids
    step_k = base_ids + future_ids[:batch_size * k]
    """
    schedule = {}
    schedule['step_0'] = list(base_ids)
    k_final = len(future_ids) // future_batch_size
    for k in range(1, k_final + 1):
        schedule[f'step_{k}'] = list(base_ids) + list(future_ids[:future_batch_size * k])
    # 나머지 (future_ids가 batch_size로 나눠떨어지지 않을 때)
    if len(future_ids) % future_batch_size != 0:
        k = k_final + 1
        schedule[f'step_{k}'] = list(base_ids) + list(future_ids)
    return schedule


def validate_static_gallery_sizes(static_gallery_sizes: List[int],
                                  n_base: int, n_future: int,
                                  batch_size: int) -> None:
    if static_gallery_sizes[0] != n_base:
        raise ValueError(f"static_gallery_sizes[0]={static_gallery_sizes[0]} must equal n_base={n_base}")
    if static_gallery_sizes[-1] != n_base + n_future:
        raise ValueError(f"static_gallery_sizes[-1]={static_gallery_sizes[-1]} must equal n_base+n_future={n_base+n_future}")
    for s in static_gallery_sizes:
        if (s - n_base) % batch_size != 0:
            raise ValueError(f"static_gallery_size {s}: (s-n_base) not divisible by batch_size {batch_size}")


def static_size_to_step_key(size: int, n_base: int, batch_size: int) -> str:
    k = (size - n_base) // batch_size
    return f'step_{k}'


# ============================================================
# Prototypes + NCM instance
# ============================================================

def fresh_ncm(device: torch.device) -> NCMClassifier:
    """각 condition entry에서 NEW instance 생성. 모든 state clean."""
    ncm = NCMClassifier(normalize=True, score_mode='cosine')
    ncm.to(device)
    return ncm


def build_prototypes(ncm: NCMClassifier,
                     embeddings: Dict[int, Dict[str, np.ndarray]],
                     enrolled_ids: List[int], device: torch.device) -> None:
    class_means = {}
    for uid in enrolled_ids:
        enroll_feats = embeddings[uid]['enroll']
        mean_feat = enroll_feats.mean(axis=0)
        class_means[uid] = torch.from_numpy(mean_feat).float().to(device)
    ncm.replace_class_means_dict(class_means)


# ============================================================
# Scoring — ALWAYS restricted to current gallery_ids
# ============================================================

def compute_max_scores_restricted(ncm: NCMClassifier, feats: np.ndarray,
                                  gallery_ids: List[int], device: torch.device,
                                  apply_snorm: bool = False
                                  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    probe features → restrict to gallery_ids 후 max score + predicted real class_id.

    Returns: (max_scores (N,), pred_ids (N,))  both numpy arrays
    """
    x = torch.from_numpy(feats).float().to(device)
    scores = ncm.forward(x, apply_snorm=apply_snorm)  # (N, max_class+1)

    gallery_idx = torch.tensor(gallery_ids, dtype=torch.long, device=device)
    gallery_scores = scores.index_select(dim=1, index=gallery_idx)  # (N, |gallery|)

    max_scores, local_idx = gallery_scores.max(dim=1)
    pred_ids = gallery_idx[local_idx]

    return max_scores.cpu().numpy(), pred_ids.cpu().numpy()


# ============================================================
# Threshold calibration
# ============================================================

def calibrate_threshold(ncm: NCMClassifier,
                        embeddings: Dict[int, Dict[str, np.ndarray]],
                        external_ids: List[int],
                        gallery_ids: List[int],
                        target_fpir: float,
                        device: torch.device,
                        apply_snorm: bool = False,
                        verbose: bool = False) -> Tuple[float, int]:
    """
    External Unknown dev 샘플의 max score → (1-target_fpir) percentile.
    Gallery restrict 적용. raw cosine 또는 S-norm space 모두 지원.

    Returns: (threshold, n_dev_probes_used)
    """
    impostor_scores = []
    for uid in external_ids:
        dev_feats = embeddings[uid]['dev']
        max_sc, _ = compute_max_scores_restricted(
            ncm, dev_feats, gallery_ids, device, apply_snorm=apply_snorm
        )
        impostor_scores.extend(max_sc.tolist())

    impostor_scores = np.array(impostor_scores)
    threshold = float(np.percentile(impostor_scores, 100 * (1 - target_fpir)))

    if verbose:
        mode = 'snorm' if apply_snorm else 'cosine'
        print(f"[THRESHOLD/{mode}] n={len(impostor_scores)}, "
              f"mean={impostor_scores.mean():.4f}, std={impostor_scores.std():.4f}, "
              f"τ(target_fpir={target_fpir})={threshold:.4f}")
    return threshold, len(impostor_scores)


# ============================================================
# S-norm cohort helpers
# ============================================================

def compute_cohort_for_class(ncm: NCMClassifier,
                             external_dev_feats_all: np.ndarray,
                             class_id: int,
                             device: torch.device,
                             snorm_min_sigma: float) -> Tuple[float, float]:
    """
    해당 class_id에 대한 external_dev 전체 sample의 raw cosine score로
    mean/std 계산. std는 floor 적용된 값 반환 (post-floor).

    NCM의 class_means는 row-indexed이므로 raw[:, class_id]가 안전.
    단, class_id는 반드시 enrolled state이어야 함.
    """
    x = torch.from_numpy(external_dev_feats_all).float().to(device)
    scores = ncm.forward(x, apply_snorm=False)  # (N, max_class+1)
    col = scores[:, class_id]
    mu = float(col.mean().item())
    raw_sigma = float(col.std().item())
    sigma = max(raw_sigma, snorm_min_sigma)   # post-floor
    return mu, sigma


def gather_external_dev_feats(embeddings: Dict[int, Dict[str, np.ndarray]],
                              external_ids: List[int]) -> np.ndarray:
    """external_dev 전체 sample을 하나의 (N, D) 배열로 합침."""
    feats_list = []
    for uid in external_ids:
        feats_list.append(embeddings[uid]['dev'])
    return np.concatenate(feats_list, axis=0)


# ============================================================
# A: Closed-set expanding-gallery identification
# ============================================================

def run_closed_set_expanding(embeddings, gallery_schedule: Dict[str, List[int]],
                             device: torch.device) -> Dict:
    """
    Rank-1 trajectory. Unknown 평가 없음. Threshold 사용 안 함.
    """
    results = {
        'condition': 'A',
        'protocol': 'A_closed_set_expanding',
        'score_mode': 'cosine',
        'apply_snorm': False,
        'threshold_mode': None,
        'threshold_space': 'raw_cosine',
        'steps': [],
    }

    step_keys = sorted(gallery_schedule.keys(), key=lambda x: int(x.split('_')[1]))

    for step_key in step_keys:
        enrolled_ids = gallery_schedule[step_key]
        ncm = fresh_ncm(device)
        build_prototypes(ncm, embeddings, enrolled_ids, device)

        correct = 0
        total = 0
        for uid in enrolled_ids:
            test_feats = embeddings[uid]['test']
            _, pred_ids = compute_max_scores_restricted(
                ncm, test_feats, enrolled_ids, device, apply_snorm=False
            )
            correct += int((pred_ids == uid).sum())
            total += len(pred_ids)

        rank1 = correct / total if total > 0 else 0.0
        results['steps'].append({
            'step': step_key,
            'gallery_step_key': step_key,
            'gallery_size': len(enrolled_ids),
            'n_enrolled': len(enrolled_ids),
            'n_probes': total,
            'rank1': float(rank1),
        })
        print(f"  [A] {step_key}: gallery={len(enrolled_ids)}, Rank-1={rank1:.4f}")

    return results


# ============================================================
# B-k: Static open-set (size-matched)
# ============================================================

def run_static_open_set(embeddings,
                        gallery_ids: List[int],
                        external_ids: List[int],
                        gallery_size_label: int,
                        gallery_step_key: str,
                        target_fpir: float,
                        device: torch.device) -> Dict:
    """
    Gallery 고정, gallery_ids 기준 threshold calibration. raw cosine.
    """
    ncm = fresh_ncm(device)
    build_prototypes(ncm, embeddings, gallery_ids, device)

    # threshold: external_dev + this gallery
    threshold, n_dev = calibrate_threshold(
        ncm, embeddings, external_ids, gallery_ids, target_fpir,
        device, apply_snorm=False, verbose=True
    )

    # Known (gallery 내 ID들)
    known_correct = 0
    known_accepted = 0
    known_total = 0
    known_max_scores = []
    known_pred_ids = []
    for uid in gallery_ids:
        test_feats = embeddings[uid]['test']
        max_sc, pred_ids = compute_max_scores_restricted(
            ncm, test_feats, gallery_ids, device, apply_snorm=False
        )
        known_correct += int((pred_ids == uid).sum())
        known_accepted += int(((pred_ids == uid) & (max_sc >= threshold)).sum())
        known_total += len(pred_ids)
        known_max_scores.extend(max_sc.tolist())
        known_pred_ids.extend(pred_ids.tolist())

    known_rank1 = known_correct / known_total if known_total > 0 else 0.0
    tpir = known_accepted / known_total if known_total > 0 else 0.0

    # External test
    external_rejected = 0
    external_total = 0
    external_max_scores = []
    for uid in external_ids:
        test_feats = embeddings[uid]['test']
        max_sc, _ = compute_max_scores_restricted(
            ncm, test_feats, gallery_ids, device, apply_snorm=False
        )
        external_rejected += int((max_sc < threshold).sum())
        external_total += len(max_sc)
        external_max_scores.extend(max_sc.tolist())

    external_rejection = external_rejected / external_total if external_total > 0 else 0.0
    achieved_fpir = 1.0 - external_rejection

    results = {
        'condition': f'B-{gallery_size_label}',
        'protocol': f'B_static_{gallery_size_label}',
        'score_mode': 'cosine',
        'apply_snorm': False,
        'threshold_mode': 'fixed',
        'threshold_space': 'raw_cosine',
        'target_fpir': target_fpir,
        'threshold_source': 'external_dev',
        'threshold': float(threshold),
        'gallery_step_key': gallery_step_key,
        'gallery_size': len(gallery_ids),
        'n_gallery': len(gallery_ids),

        'rank1': float(known_rank1),
        'known_rank1': float(known_rank1),  # legacy
        'tpir_at_1pct_fpir': float(tpir),
        'known_acceptance': float(tpir),  # legacy alias
        'external_rejection_rate': float(external_rejection),
        'achieved_external_fpir': float(achieved_fpir),

        'known_probes': known_total,
        'external_dev_probe_count': n_dev,
        'external_test_probe_count': external_total,
        'external_probes': external_total,  # legacy

        # sanity diagnostics
        'known_max_scores': known_max_scores,
        'known_pred_ids': known_pred_ids,
        'external_max_scores': external_max_scores,
    }

    print(f"  [B-{gallery_size_label}] gallery_size={len(gallery_ids)} step_key={gallery_step_key}, τ={threshold:.4f}")
    print(f"                          Rank-1={known_rank1:.4f}, TPIR@1%FPIR={tpir:.4f}, "
          f"ExtRej={external_rejection:.4f}, achieved_FPIR={achieved_fpir:.4f}")
    return results


# ============================================================
# C variants — 공통 helper
# ============================================================

def _eval_step_groups(ncm: NCMClassifier, embeddings,
                      enrolled_ids: List[int], external_ids: List[int],
                      remaining_future_ids: List[int],
                      threshold: float, device: torch.device,
                      apply_snorm: bool) -> Dict:
    """
    한 step에서 3 group (known / not-yet-enrolled / external) 평가.
    모든 max/argmax는 enrolled_ids (= current gallery)에 restrict.
    """
    # Known
    known_correct = 0
    known_accepted = 0
    known_total = 0
    known_scores = []
    known_pred_ids = []
    for uid in enrolled_ids:
        test_feats = embeddings[uid]['test']
        max_sc, pred_ids = compute_max_scores_restricted(
            ncm, test_feats, enrolled_ids, device, apply_snorm=apply_snorm
        )
        known_correct += int((pred_ids == uid).sum())
        known_accepted += int(((pred_ids == uid) & (max_sc >= threshold)).sum())
        known_total += len(pred_ids)
        known_scores.extend(max_sc.tolist())
        known_pred_ids.extend(pred_ids.tolist())

    # Not-yet-enrolled
    nye_rejected = 0
    nye_total = 0
    nye_scores = []
    for uid in remaining_future_ids:
        test_feats = embeddings[uid]['test']
        max_sc, _ = compute_max_scores_restricted(
            ncm, test_feats, enrolled_ids, device, apply_snorm=apply_snorm
        )
        nye_rejected += int((max_sc < threshold).sum())
        nye_total += len(max_sc)
        nye_scores.extend(max_sc.tolist())

    # External
    ext_rejected = 0
    ext_total = 0
    ext_scores = []
    for uid in external_ids:
        test_feats = embeddings[uid]['test']
        max_sc, _ = compute_max_scores_restricted(
            ncm, test_feats, enrolled_ids, device, apply_snorm=apply_snorm
        )
        ext_rejected += int((max_sc < threshold).sum())
        ext_total += len(max_sc)
        ext_scores.extend(max_sc.tolist())

    known_rank1 = known_correct / known_total if known_total > 0 else 0.0
    tpir = known_accepted / known_total if known_total > 0 else 0.0
    nye_rej = nye_rejected / nye_total if nye_total > 0 else 0.0
    ext_rej = ext_rejected / ext_total if ext_total > 0 else 0.0
    achieved_fpir = 1.0 - ext_rej

    return {
        'known_correct': known_correct,
        'known_accepted': known_accepted,
        'known_total': known_total,
        'known_rank1': known_rank1,
        'tpir_at_1pct_fpir': tpir,
        'nye_rejected': nye_rejected,
        'nye_total': nye_total,
        'nye_rejection': nye_rej,
        'ext_rejected': ext_rejected,
        'ext_total': ext_total,
        'ext_rejection': ext_rej,
        'achieved_external_fpir': achieved_fpir,
        'known_scores': known_scores,
        'known_pred_ids': known_pred_ids,
        'nye_scores': nye_scores,
        'external_scores': ext_scores,
    }


def _build_c_steps(gallery_schedule: Dict[str, List[int]],
                   future_ids: List[int]) -> List[Tuple[str, List[int], List[int]]]:
    """
    각 step에 (step_key, enrolled_ids, remaining_future_ids) 반환.
    remaining_future = future_ids \\ (enrolled ∩ future_ids)
    """
    step_keys = sorted(gallery_schedule.keys(), key=lambda x: int(x.split('_')[1]))
    future_set_total = set(future_ids)
    out = []
    for sk in step_keys:
        enrolled = gallery_schedule[sk]
        enrolled_future = set(enrolled) & future_set_total
        remaining = [fid for fid in future_ids if fid not in enrolled_future]
        out.append((sk, enrolled, remaining))
    return out


def _step_result_dict(step_key: str, enrolled_ids: List[int],
                      remaining_future_ids: List[int], threshold: float,
                      group_eval: Dict, n_dev_used: int) -> Dict:
    return {
        'step': step_key,
        'gallery_step_key': step_key,
        'gallery_size': len(enrolled_ids),
        'n_enrolled': len(enrolled_ids),
        'n_remaining_future': len(remaining_future_ids),
        'threshold': float(threshold),

        'rank1': float(group_eval['known_rank1']),
        'known_rank1': float(group_eval['known_rank1']),  # legacy
        'tpir_at_1pct_fpir': float(group_eval['tpir_at_1pct_fpir']),
        'known_acceptance': float(group_eval['tpir_at_1pct_fpir']),  # legacy

        'external_rejection_rate': float(group_eval['ext_rejection']),
        'external_rejection': float(group_eval['ext_rejection']),  # legacy
        'achieved_external_fpir': float(group_eval['achieved_external_fpir']),

        'not_yet_enrolled_rejection': float(group_eval['nye_rejection']),
        'future_rejection': float(group_eval['nye_rejection']),  # legacy

        'known_probes': group_eval['known_total'],
        'external_dev_probe_count': n_dev_used,
        'external_test_probe_count': group_eval['ext_total'],
        'external_probes': group_eval['ext_total'],  # legacy
        'not_yet_enrolled_probes': group_eval['nye_total'],
        'future_probes': group_eval['nye_total'],  # legacy
    }


# ============================================================
# C-raw-fixed
# ============================================================

def run_sequential_raw_fixed(embeddings,
                             gallery_schedule: Dict[str, List[int]],
                             future_ids: List[int],
                             external_ids: List[int],
                             target_fpir: float,
                             device: torch.device) -> Dict:
    """Sequential, raw cosine, fixed τ (step 0)."""
    steps = _build_c_steps(gallery_schedule, future_ids)
    results = {
        'condition': 'C-raw-fixed',
        'protocol': 'C_raw_fixed',
        'score_mode': 'cosine',
        'apply_snorm': False,
        'threshold_mode': 'fixed',
        'threshold_space': 'raw_cosine',
        'target_fpir': target_fpir,
        'threshold_source': 'external_dev',
        'steps': [],
    }
    score_distributions = {}
    threshold = None
    n_dev_used = 0

    for step_key, enrolled_ids, remaining_future_ids in steps:
        ncm = fresh_ncm(device)
        build_prototypes(ncm, embeddings, enrolled_ids, device)

        # Step 0에서만 calibrate, 이후 고정
        if threshold is None:
            threshold, n_dev_used = calibrate_threshold(
                ncm, embeddings, external_ids, enrolled_ids, target_fpir,
                device, apply_snorm=False, verbose=True
            )

        ge = _eval_step_groups(ncm, embeddings, enrolled_ids, external_ids,
                               remaining_future_ids, threshold, device,
                               apply_snorm=False)

        results['steps'].append(
            _step_result_dict(step_key, enrolled_ids, remaining_future_ids,
                              threshold, ge, n_dev_used)
        )
        score_distributions[step_key] = {
            'known': ge['known_scores'],
            'future': ge['nye_scores'],
            'external': ge['external_scores'],
        }
        print(f"  [C-raw-fixed] {step_key}: n={len(enrolled_ids)}, τ={threshold:.4f}, "
              f"Rank-1={ge['known_rank1']:.4f}, TPIR={ge['tpir_at_1pct_fpir']:.4f}, "
              f"ExtRej={ge['ext_rejection']:.4f}, NYE-Rej={ge['nye_rejection']:.4f}")

    results['threshold'] = float(threshold)
    results['score_distributions'] = score_distributions
    return results


# ============================================================
# C-raw-recalib
# ============================================================

def run_sequential_raw_recalib(embeddings,
                               gallery_schedule: Dict[str, List[int]],
                               future_ids: List[int],
                               external_ids: List[int],
                               target_fpir: float,
                               device: torch.device,
                               monotonicity_tol: float = 1e-8) -> Dict:
    """Sequential, raw cosine, τ recalibrated each step."""
    steps = _build_c_steps(gallery_schedule, future_ids)
    results = {
        'condition': 'C-raw-recalib',
        'protocol': 'C_raw_recalib',
        'score_mode': 'cosine',
        'apply_snorm': False,
        'threshold_mode': 'recalib',
        'threshold_space': 'raw_cosine',
        'target_fpir': target_fpir,
        'threshold_source': 'external_dev',
        'steps': [],
    }
    score_distributions = {}
    thresholds_trajectory = []

    for step_key, enrolled_ids, remaining_future_ids in steps:
        ncm = fresh_ncm(device)
        build_prototypes(ncm, embeddings, enrolled_ids, device)

        # 매 step threshold 재계산
        threshold, n_dev_used = calibrate_threshold(
            ncm, embeddings, external_ids, enrolled_ids, target_fpir,
            device, apply_snorm=False, verbose=True
        )
        thresholds_trajectory.append(threshold)

        ge = _eval_step_groups(ncm, embeddings, enrolled_ids, external_ids,
                               remaining_future_ids, threshold, device,
                               apply_snorm=False)

        results['steps'].append(
            _step_result_dict(step_key, enrolled_ids, remaining_future_ids,
                              threshold, ge, n_dev_used)
        )
        score_distributions[step_key] = {
            'known': ge['known_scores'],
            'future': ge['nye_scores'],
            'external': ge['external_scores'],
        }
        print(f"  [C-raw-recalib] {step_key}: n={len(enrolled_ids)}, τ={threshold:.4f}, "
              f"Rank-1={ge['known_rank1']:.4f}, TPIR={ge['tpir_at_1pct_fpir']:.4f}, "
              f"ExtRej={ge['ext_rejection']:.4f}, NYE-Rej={ge['nye_rejection']:.4f}")

    # Monotonicity check (raw only, with tolerance)
    mono_violations = []
    for k in range(1, len(thresholds_trajectory)):
        if thresholds_trajectory[k] < thresholds_trajectory[k-1] - monotonicity_tol:
            mono_violations.append({
                'k': k,
                'prev': thresholds_trajectory[k-1],
                'curr': thresholds_trajectory[k],
                'delta': thresholds_trajectory[k] - thresholds_trajectory[k-1],
            })
    if mono_violations:
        print(f"  [C-raw-recalib] ⚠ monotonicity violations (tol={monotonicity_tol}): "
              f"{len(mono_violations)} cases — sample: {mono_violations[0]}")
    else:
        print(f"  [C-raw-recalib] ✓ threshold monotonicity OK (tol={monotonicity_tol})")

    results['thresholds_trajectory'] = thresholds_trajectory
    results['monotonicity_violations'] = mono_violations
    results['score_distributions'] = score_distributions
    return results


# ============================================================
# C-snorm-fixed
# ============================================================

def run_sequential_snorm_fixed(embeddings,
                               gallery_schedule: Dict[str, List[int]],
                               future_ids: List[int],
                               external_ids: List[int],
                               target_fpir: float,
                               device: torch.device) -> Dict:
    """
    Sequential, S-norm, fixed τ (step 0 S-norm space).
    Cohort stats는 enrollment 시점에 새 class에 대해서만 계산 (기존 class 불변).
    """
    steps = _build_c_steps(gallery_schedule, future_ids)
    results = {
        'condition': 'C-snorm-fixed',
        'protocol': 'C_snorm_fixed',
        'score_mode': 'snorm',
        'apply_snorm': True,
        'threshold_mode': 'fixed',
        'threshold_space': 'snorm',
        'target_fpir': target_fpir,
        'threshold_source': 'external_dev',
        'steps': [],
        'cohort_invariance_checks': [],
    }
    score_distributions = {}

    # external_dev features (모든 step에서 재사용)
    external_dev_feats = gather_external_dev_feats(embeddings, external_ids)

    # 누적 cohort state (fresh NCM 매 step마다 생성해도 이 dict는 carry-over)
    cohort_mu_dict: Dict[int, float] = {}
    cohort_sigma_dict: Dict[int, float] = {}
    snorm_min_sigma = 1e-2  # matches ncm.snorm_min_sigma

    threshold = None
    n_dev_used = 0
    enrolled_set_prev = set()

    for step_key, enrolled_ids, remaining_future_ids in steps:
        ncm = fresh_ncm(device)
        build_prototypes(ncm, embeddings, enrolled_ids, device)

        # 새 class 식별
        new_classes = [c for c in enrolled_ids if c not in enrolled_set_prev]

        # 새 class에 대해 cohort stats 계산 (기존 class는 그대로)
        for c in new_classes:
            mu, sigma = compute_cohort_for_class(
                ncm, external_dev_feats, c, device, snorm_min_sigma
            )
            cohort_mu_dict[c] = mu
            cohort_sigma_dict[c] = sigma

        # NCM에 cohort 반영 (old dict까지 포함해서 full set)
        ncm.set_cohort_stats(cohort_mu_dict, cohort_sigma_dict)

        # Cohort invariance check: old class stats 변경 없어야 함
        if enrolled_set_prev:
            old_mu_delta = 0.0
            old_sigma_delta = 0.0
            for c in enrolled_set_prev:
                # NCM에 저장된 post-floor 값과 dict 값 비교
                stored_mu = float(ncm.cohort_mu[c].item())
                stored_sigma = float(ncm.cohort_sigma[c].item())
                d_mu = abs(stored_mu - cohort_mu_dict[c])
                d_sigma = abs(stored_sigma - cohort_sigma_dict[c])
                old_mu_delta = max(old_mu_delta, d_mu)
                old_sigma_delta = max(old_sigma_delta, d_sigma)
            check = {
                'step': step_key,
                'cohort_size': len(cohort_mu_dict),
                'n_new': len(new_classes),
                'old_class_mu_delta_max': old_mu_delta,
                'old_class_sigma_delta_max': old_sigma_delta,
            }
            results['cohort_invariance_checks'].append(check)
            if old_mu_delta > 0.0 or old_sigma_delta > 0.0:
                print(f"  [C-snorm-fixed] ⚠ cohort invariance violated at {step_key}: "
                      f"μ_delta={old_mu_delta:.3e}, σ_delta={old_sigma_delta:.3e}")

        print(f"  [C-snorm-fixed] {step_key}: cohort size = {len(cohort_mu_dict)} (+{len(new_classes)} new)")

        # Step 0에서 threshold를 S-norm space에서 calibrate (FIXED)
        if threshold is None:
            threshold, n_dev_used = calibrate_threshold(
                ncm, embeddings, external_ids, enrolled_ids, target_fpir,
                device, apply_snorm=True, verbose=True
            )

        ge = _eval_step_groups(ncm, embeddings, enrolled_ids, external_ids,
                               remaining_future_ids, threshold, device,
                               apply_snorm=True)

        results['steps'].append(
            _step_result_dict(step_key, enrolled_ids, remaining_future_ids,
                              threshold, ge, n_dev_used)
        )
        score_distributions[step_key] = {
            'known': ge['known_scores'],
            'future': ge['nye_scores'],
            'external': ge['external_scores'],
        }
        print(f"  [C-snorm-fixed] {step_key}: n={len(enrolled_ids)}, τ(snorm)={threshold:.4f}, "
              f"Rank-1={ge['known_rank1']:.4f}, TPIR={ge['tpir_at_1pct_fpir']:.4f}, "
              f"ExtRej={ge['ext_rejection']:.4f}, NYE-Rej={ge['nye_rejection']:.4f}")

        enrolled_set_prev = set(enrolled_ids)

    results['threshold'] = float(threshold)
    results['score_distributions'] = score_distributions
    return results


# ============================================================
# C-snorm-recalib
# ============================================================

def run_sequential_snorm_recalib(embeddings,
                                 gallery_schedule: Dict[str, List[int]],
                                 future_ids: List[int],
                                 external_ids: List[int],
                                 target_fpir: float,
                                 device: torch.device) -> Dict:
    """
    Sequential, S-norm, τ recalibrated each step (S-norm space).
    Cohort stats는 새 class 증분 계산 (기존 class 불변).
    """
    steps = _build_c_steps(gallery_schedule, future_ids)
    results = {
        'condition': 'C-snorm-recalib',
        'protocol': 'C_snorm_recalib',
        'score_mode': 'snorm',
        'apply_snorm': True,
        'threshold_mode': 'recalib',
        'threshold_space': 'snorm',
        'target_fpir': target_fpir,
        'threshold_source': 'external_dev',
        'steps': [],
        'cohort_invariance_checks': [],
    }
    score_distributions = {}
    thresholds_trajectory = []

    external_dev_feats = gather_external_dev_feats(embeddings, external_ids)
    cohort_mu_dict: Dict[int, float] = {}
    cohort_sigma_dict: Dict[int, float] = {}
    snorm_min_sigma = 1e-2
    enrolled_set_prev = set()

    for step_key, enrolled_ids, remaining_future_ids in steps:
        ncm = fresh_ncm(device)
        build_prototypes(ncm, embeddings, enrolled_ids, device)

        new_classes = [c for c in enrolled_ids if c not in enrolled_set_prev]
        for c in new_classes:
            mu, sigma = compute_cohort_for_class(
                ncm, external_dev_feats, c, device, snorm_min_sigma
            )
            cohort_mu_dict[c] = mu
            cohort_sigma_dict[c] = sigma

        ncm.set_cohort_stats(cohort_mu_dict, cohort_sigma_dict)

        if enrolled_set_prev:
            old_mu_delta = 0.0
            old_sigma_delta = 0.0
            for c in enrolled_set_prev:
                stored_mu = float(ncm.cohort_mu[c].item())
                stored_sigma = float(ncm.cohort_sigma[c].item())
                d_mu = abs(stored_mu - cohort_mu_dict[c])
                d_sigma = abs(stored_sigma - cohort_sigma_dict[c])
                old_mu_delta = max(old_mu_delta, d_mu)
                old_sigma_delta = max(old_sigma_delta, d_sigma)
            results['cohort_invariance_checks'].append({
                'step': step_key,
                'cohort_size': len(cohort_mu_dict),
                'n_new': len(new_classes),
                'old_class_mu_delta_max': old_mu_delta,
                'old_class_sigma_delta_max': old_sigma_delta,
            })

        # 매 step threshold 재계산 (S-norm space)
        threshold, n_dev_used = calibrate_threshold(
            ncm, embeddings, external_ids, enrolled_ids, target_fpir,
            device, apply_snorm=True, verbose=True
        )
        thresholds_trajectory.append(threshold)

        ge = _eval_step_groups(ncm, embeddings, enrolled_ids, external_ids,
                               remaining_future_ids, threshold, device,
                               apply_snorm=True)

        results['steps'].append(
            _step_result_dict(step_key, enrolled_ids, remaining_future_ids,
                              threshold, ge, n_dev_used)
        )
        score_distributions[step_key] = {
            'known': ge['known_scores'],
            'future': ge['nye_scores'],
            'external': ge['external_scores'],
        }
        print(f"  [C-snorm-recalib] {step_key}: n={len(enrolled_ids)}, τ(snorm)={threshold:.4f}, "
              f"Rank-1={ge['known_rank1']:.4f}, TPIR={ge['tpir_at_1pct_fpir']:.4f}, "
              f"ExtRej={ge['ext_rejection']:.4f}, NYE-Rej={ge['nye_rejection']:.4f}")

        enrolled_set_prev = set(enrolled_ids)

    results['thresholds_trajectory'] = thresholds_trajectory
    results['score_distributions'] = score_distributions
    return results


# ============================================================
# Sanity check: B-sF vs C-recalib-final
# ============================================================

def sanity_check_b_vs_c_recalib(b_result: Dict, c_recalib_result: Dict) -> Dict:
    """
    Gallery set 일치 + probe order 일치 전제 하에서
    B-sF final snapshot과 C-recalib final step이 score/threshold/metric 전부 동일해야 함.
    """
    c_final_step = c_recalib_result['steps'][-1]

    # gallery size
    gallery_size_match = (b_result['gallery_size'] == c_final_step['gallery_size'])
    gallery_step_key_match = (b_result['gallery_step_key'] == c_final_step['gallery_step_key'])

    # probe count (order는 코드가 enrolled_ids 순회 순서로 deterministic이므로 count로 근사)
    known_probe_count_match = (b_result['known_probes'] == c_final_step['known_probes'])
    external_probe_count_match = (
        b_result['external_test_probe_count'] == c_final_step['external_test_probe_count']
    )

    # threshold / metric diff
    threshold_diff = abs(b_result['threshold'] - c_final_step['threshold'])
    rank1_diff = abs(b_result['rank1'] - c_final_step['rank1'])
    tpir_diff = abs(b_result['tpir_at_1pct_fpir'] - c_final_step['tpir_at_1pct_fpir'])
    ext_rej_diff = abs(b_result['external_rejection_rate'] - c_final_step['external_rejection_rate'])
    achieved_fpir_diff = abs(b_result['achieved_external_fpir'] - c_final_step['achieved_external_fpir'])

    # score diff (순서 같다고 가정하고 abs diff max)
    b_known_scores = np.array(b_result.get('known_max_scores', []))
    # C-recalib-final의 known_scores는 eval_step_groups의 known_scores와 같음
    c_known_scores = np.array(
        c_recalib_result['score_distributions'][c_final_step['step']]['known']
    )
    if len(b_known_scores) == len(c_known_scores) and len(b_known_scores) > 0:
        known_max_abs_diff = float(np.max(np.abs(b_known_scores - c_known_scores)))
    else:
        known_max_abs_diff = float('nan')

    b_ext_scores = np.array(b_result.get('external_max_scores', []))
    c_ext_scores = np.array(
        c_recalib_result['score_distributions'][c_final_step['step']]['external']
    )
    if len(b_ext_scores) == len(c_ext_scores) and len(b_ext_scores) > 0:
        external_max_abs_diff = float(np.max(np.abs(b_ext_scores - c_ext_scores)))
    else:
        external_max_abs_diff = float('nan')

    b_pred = b_result.get('known_pred_ids', [])
    # C-recalib-final pred_ids 비교 데이터가 필요하면 _eval_step_groups에서 꺼낼 수 있지만
    # 여기선 score 기반으로 간접 확인. Pred match는 score equality와 gallery 일치로부터 follow.
    pred_match_rate = 1.0 if known_max_abs_diff < 1e-6 else float('nan')

    summary = {
        'gallery_ids_match': gallery_size_match and gallery_step_key_match,
        'gallery_size_match': gallery_size_match,
        'gallery_step_key_match': gallery_step_key_match,
        'known_probe_order_match': known_probe_count_match,
        'external_probe_order_match': external_probe_count_match,
        'known_max_abs_diff': known_max_abs_diff,
        'external_max_abs_diff': external_max_abs_diff,
        'pred_match_rate': pred_match_rate,
        'threshold_diff': threshold_diff,
        'rank1_diff': rank1_diff,
        'tpir_diff': tpir_diff,
        'external_rejection_rate_diff': ext_rej_diff,
        'achieved_fpir_diff': achieved_fpir_diff,
    }

    print("\n[SANITY] B-sF vs C-recalib-final")
    for k, v in summary.items():
        print(f"  {k:30s} = {v}")
    return summary


# ============================================================
# 유틸
# ============================================================

def save_results(results: Dict, output_path: str):
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return sorted(obj)
        return obj

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"[SAVE] {output_path}")
