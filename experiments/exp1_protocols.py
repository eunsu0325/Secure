"""
Experiment 1: Protocol Comparison — 프로토콜 로직
split, prototype, scoring, Protocol A/B/C 구현
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

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from coconut.openset.utils import load_paths_labels_from_txt, set_seed
from coconut.openset.score_extraction import extract_features
from coconut.classifiers.ncm import NCMClassifier
from coconut.data.transforms import get_scr_transforms


# ============================================================
# 1. 데이터 로드 및 검증
# ============================================================

def load_and_validate_data(txt_file: str, base_path: str = "") -> Dict[int, List[str]]:
    """
    txt 파일에서 paths/labels 로드 후 ID별로 그룹핑.
    검증 로그 출력.
    """
    paths, labels = load_paths_labels_from_txt(txt_file)

    if base_path:
        paths = [os.path.join(base_path, p) for p in paths]

    # ID별 샘플 그룹핑
    id_to_paths = defaultdict(list)
    for p, l in zip(paths, labels):
        id_to_paths[l].append(p)

    # 검증 로그
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
# 2. Identity / Sample 분할
# ============================================================

def split_identities(all_ids: List[int], n_base: int = 20,
                     n_future: int = 40, n_external: int = 40,
                     seed: int = 42) -> Dict[str, List[int]]:
    """20/40/40 identity 분할. 결정론적."""
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(all_ids)

    total_needed = n_base + n_future + n_external
    if len(shuffled) < total_needed:
        raise ValueError(f"Need {total_needed} IDs but only {len(shuffled)} available")

    selected = shuffled[:total_needed]
    split = {
        'base_ids': sorted(selected[:n_base].tolist()),
        'future_ids': sorted(selected[n_base:n_base + n_future].tolist()),
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
    """
    각 ID의 샘플을 enroll/dev/test로 분할.
    seed = base_seed + id 로 결정론적.
    """
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

    # 검증 로그
    n_test = len(paths) - n_enroll - n_dev
    print(f"[SAMPLE SPLIT] per ID: {n_enroll} enroll / {n_dev} dev / {n_test} test")
    print(f"[SAMPLE SPLIT] total IDs split: {len(sample_split)}")
    return sample_split


# ============================================================
# 3. Embedding 추출
# ============================================================

def extract_all_embeddings(model, sample_split: Dict[int, Dict[str, List[str]]],
                           transform, device, channels: int = 1
                           ) -> Dict[int, Dict[str, np.ndarray]]:
    """
    모든 ID의 enroll/dev/test 샘플에서 embedding 추출.
    Returns: {id: {'enroll': (3, 2048), 'dev': (3, 2048), 'test': (4, 2048)}}
    """
    embeddings = {}
    all_paths = []
    path_map = []  # (id, split_name, local_idx)

    # 모든 경로 수집
    for uid in sorted(sample_split.keys()):
        for split_name in ['enroll', 'dev', 'test']:
            for idx, p in enumerate(sample_split[uid][split_name]):
                all_paths.append(p)
                path_map.append((uid, split_name, idx))

    # 일괄 추출
    print(f"[EMBEDDING] extracting features from {len(all_paths)} images...")
    all_feats = extract_features(model, all_paths, transform, device,
                                 batch_size=64, channels=channels)
    print(f"[EMBEDDING] shape: {all_feats.shape}")

    # L2 norm 검증
    norms = np.linalg.norm(all_feats, axis=1)
    print(f"[EMBEDDING] L2 norm range: {norms.min():.4f} - {norms.max():.4f}")

    # ID/split별로 재분배
    for uid in sorted(sample_split.keys()):
        embeddings[uid] = {}
    for i, (uid, split_name, local_idx) in enumerate(path_map):
        if split_name not in embeddings[uid]:
            embeddings[uid][split_name] = []
        embeddings[uid][split_name].append(all_feats[i])

    # list → numpy array
    for uid in embeddings:
        for split_name in embeddings[uid]:
            embeddings[uid][split_name] = np.stack(embeddings[uid][split_name])

    return embeddings


# ============================================================
# 4. Prototype 구성
# ============================================================

def build_prototypes(ncm: NCMClassifier,
                     embeddings: Dict[int, Dict[str, np.ndarray]],
                     enrolled_ids: List[int], device: torch.device):
    """
    enrolled_ids의 enrollment embedding 평균 → NCM prototype 설정.
    """
    class_means = {}
    for uid in enrolled_ids:
        enroll_feats = embeddings[uid]['enroll']  # (3, 2048)
        mean_feat = enroll_feats.mean(axis=0)     # (2048,)
        class_means[uid] = torch.from_numpy(mean_feat).float().to(device)

    ncm.replace_class_means_dict(class_means)
    ncm._vectorize_means_dict()
    print(f"[PROTOTYPE] {len(enrolled_ids)} prototypes set in NCM")


# ============================================================
# 5. Scoring 헬퍼
# ============================================================

def compute_max_scores(ncm: NCMClassifier, feats: np.ndarray,
                       device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    probe features → gallery 대비 max cosine score + predicted ID.
    Returns: (max_scores (N,), pred_ids (N,))
    """
    x = torch.from_numpy(feats).float().to(device)
    scores = ncm.forward(x, apply_snorm=False)  # (N, C) raw cosine
    max_scores, pred_indices = scores.max(dim=1)

    # pred_indices는 class_means 텐서의 인덱스 → 실제 class ID와 동일
    # (NCM이 sparse ID space를 그대로 사용하므로)
    return max_scores.cpu().numpy(), pred_indices.cpu().numpy()


# ============================================================
# 6. Threshold Calibration
# ============================================================

def calibrate_threshold(ncm: NCMClassifier,
                        embeddings: Dict[int, Dict[str, np.ndarray]],
                        external_ids: List[int],
                        target_fpir: float, device: torch.device) -> float:
    """
    External Unknown dev 샘플의 max cosine score → 99th percentile threshold.
    Protocol B와 C에서 동일한 threshold 사용.
    """
    # external unknown dev score 수집
    impostor_scores = []
    for uid in external_ids:
        dev_feats = embeddings[uid]['dev']  # (3, 2048)
        max_sc, _ = compute_max_scores(ncm, dev_feats, device)
        impostor_scores.extend(max_sc.tolist())

    impostor_scores = np.array(impostor_scores)
    # 99th percentile = FPIR 1% 타겟
    threshold = float(np.percentile(impostor_scores, 100 * (1 - target_fpir)))

    print(f"[THRESHOLD] external unknown dev scores: n={len(impostor_scores)}, "
          f"mean={impostor_scores.mean():.4f}, std={impostor_scores.std():.4f}")
    print(f"[THRESHOLD] target FPIR={target_fpir}, threshold={threshold:.4f}")
    return threshold


# ============================================================
# 7. Protocol A: Closed-set CIL
# ============================================================

def run_protocol_a(embeddings: Dict[int, Dict[str, np.ndarray]],
                   identity_split: Dict[str, List[int]],
                   ncm: NCMClassifier, device: torch.device,
                   future_batch_size: int = 5) -> Dict:
    """
    Closed-set continual evaluation.
    - Base IDs로 시작, Future IDs를 5명씩 추가
    - 테스트는 known(등록된) IDs의 test probe만
    - Unknown 평가 없음
    - 측정: Rank-1만
    """
    base_ids = identity_split['base_ids']
    future_ids = identity_split['future_ids']

    # step 구성: step 0 = base만, step k = +5 future
    steps = []
    steps.append(('step_0', list(base_ids)))
    for i in range(0, len(future_ids), future_batch_size):
        batch = future_ids[i:i + future_batch_size]
        prev_enrolled = steps[-1][1]
        steps.append((f'step_{len(steps)}', prev_enrolled + batch))

    results = {'steps': [], 'protocol': 'A_closed_set_cil'}

    for step_name, enrolled_ids in steps:
        # prototype 구성
        build_prototypes(ncm, embeddings, enrolled_ids, device)

        # known test probes만 평가
        correct = 0
        total = 0
        for uid in enrolled_ids:
            test_feats = embeddings[uid]['test']
            _, pred_ids = compute_max_scores(ncm, test_feats, device)
            correct += (pred_ids == uid).sum()
            total += len(pred_ids)

        rank1 = correct / total if total > 0 else 0.0

        step_result = {
            'step': step_name,
            'n_enrolled': len(enrolled_ids),
            'n_probes': total,
            'rank1': float(rank1),
        }
        results['steps'].append(step_result)
        print(f"  [Proto A] {step_name}: enrolled={len(enrolled_ids)}, "
              f"Rank-1={rank1:.4f}")

    return results


# ============================================================
# 8. Protocol B: Static Open-set
# ============================================================

def run_protocol_b(embeddings: Dict[int, Dict[str, np.ndarray]],
                   identity_split: Dict[str, List[int]],
                   ncm: NCMClassifier, threshold: float,
                   device: torch.device) -> Dict:
    """
    Static open-set evaluation.
    - Gallery = Base IDs만 (고정)
    - Probe = Base genuine test + External Unknown test
    - 측정: known Rank-1, external unknown rejection rate
    """
    base_ids = identity_split['base_ids']
    external_ids = identity_split['external_ids']

    # prototype 구성: Base만
    build_prototypes(ncm, embeddings, base_ids, device)

    # Known (Base) test 평가
    known_correct = 0
    known_accepted = 0  # threshold 통과 + 분류 정확
    known_total = 0
    for uid in base_ids:
        test_feats = embeddings[uid]['test']
        max_sc, pred_ids = compute_max_scores(ncm, test_feats, device)
        known_correct += (pred_ids == uid).sum()
        known_accepted += ((pred_ids == uid) & (max_sc >= threshold)).sum()
        known_total += len(pred_ids)

    known_rank1 = known_correct / known_total if known_total > 0 else 0.0
    known_acceptance = known_accepted / known_total if known_total > 0 else 0.0

    # External Unknown test 평가
    external_rejected = 0
    external_total = 0
    external_max_scores = []
    for uid in external_ids:
        test_feats = embeddings[uid]['test']
        max_sc, _ = compute_max_scores(ncm, test_feats, device)
        external_rejected += (max_sc < threshold).sum()
        external_total += len(max_sc)
        external_max_scores.extend(max_sc.tolist())

    external_rejection = external_rejected / external_total if external_total > 0 else 0.0

    results = {
        'protocol': 'B_static_open_set',
        'n_gallery': len(base_ids),
        'threshold': threshold,
        'known_rank1': float(known_rank1),
        'known_acceptance': float(known_acceptance),
        'known_probes': known_total,
        'external_rejection_rate': float(external_rejection),
        'external_probes': external_total,
        'external_max_score_mean': float(np.mean(external_max_scores)),
        'external_max_score_std': float(np.std(external_max_scores)),
    }

    print(f"  [Proto B] gallery={len(base_ids)}, threshold={threshold:.4f}")
    print(f"  [Proto B] known Rank-1={known_rank1:.4f}, accept={known_acceptance:.4f} ({known_total} probes)")
    print(f"  [Proto B] external rejection={external_rejection:.4f} ({external_total} probes)")
    return results


# ============================================================
# 9. Protocol C: Sequential Enrollment (C-fixed)
# ============================================================

def run_protocol_c(embeddings: Dict[int, Dict[str, np.ndarray]],
                   identity_split: Dict[str, List[int]],
                   ncm: NCMClassifier, threshold: float,
                   device: torch.device,
                   future_batch_size: int = 5) -> Dict:
    """
    Sequential enrollment evaluation (C-fixed).
    - threshold는 step 0에서 calibration한 값 그대로 유지
    - 각 step에서 3그룹 평가: known, future-known, external unknown
    - known: Rank-1 + acceptance rate
    """
    base_ids = identity_split['base_ids']
    future_ids = identity_split['future_ids']
    external_ids = identity_split['external_ids']

    # step 구성
    steps = []
    steps.append(('step_0', list(base_ids), list(future_ids)))
    for i in range(0, len(future_ids), future_batch_size):
        batch = future_ids[i:i + future_batch_size]
        prev_enrolled = steps[-1][1]
        remaining_future = [fid for fid in future_ids if fid not in prev_enrolled and fid not in batch]
        steps.append((f'step_{len(steps)}', prev_enrolled + batch, remaining_future))

    results = {'steps': [], 'protocol': 'C_sequential_enrollment_fixed',
               'threshold': threshold}

    # score distributions 저장 (Figure 3용)
    score_distributions = {}

    for step_name, enrolled_ids, remaining_future_ids in steps:
        # prototype 구성
        build_prototypes(ncm, embeddings, enrolled_ids, device)

        # --- (a) Known: 등록된 IDs test probe ---
        known_correct = 0
        known_accepted = 0  # threshold 통과 + 분류 정확
        known_total = 0
        known_scores = []
        for uid in enrolled_ids:
            test_feats = embeddings[uid]['test']
            max_sc, pred_ids = compute_max_scores(ncm, test_feats, device)
            known_correct += (pred_ids == uid).sum()
            known_accepted += ((pred_ids == uid) & (max_sc >= threshold)).sum()
            known_total += len(pred_ids)
            known_scores.extend(max_sc.tolist())

        known_rank1 = known_correct / known_total if known_total > 0 else 0.0
        known_acceptance = known_accepted / known_total if known_total > 0 else 0.0

        # --- (b) Future-known: 아직 미등록 Future IDs ---
        future_rejected = 0
        future_total = 0
        future_scores = []
        for uid in remaining_future_ids:
            test_feats = embeddings[uid]['test']
            max_sc, _ = compute_max_scores(ncm, test_feats, device)
            future_rejected += (max_sc < threshold).sum()
            future_total += len(max_sc)
            future_scores.extend(max_sc.tolist())

        future_rejection = future_rejected / future_total if future_total > 0 else 0.0

        # --- (c) External Unknown ---
        external_rejected = 0
        external_total = 0
        external_scores = []
        for uid in external_ids:
            test_feats = embeddings[uid]['test']
            max_sc, _ = compute_max_scores(ncm, test_feats, device)
            external_rejected += (max_sc < threshold).sum()
            external_total += len(max_sc)
            external_scores.extend(max_sc.tolist())

        external_rejection = external_rejected / external_total if external_total > 0 else 0.0

        step_result = {
            'step': step_name,
            'n_enrolled': len(enrolled_ids),
            'n_remaining_future': len(remaining_future_ids),
            'known_rank1': float(known_rank1),
            'known_acceptance': float(known_acceptance),
            'known_probes': known_total,
            'future_rejection': float(future_rejection),
            'future_probes': future_total,
            'external_rejection': float(external_rejection),
            'external_probes': external_total,
        }
        results['steps'].append(step_result)

        # score 분포 저장 (step 0과 final만 Figure 3에 사용)
        score_distributions[step_name] = {
            'known': known_scores,
            'future': future_scores,
            'external': external_scores,
        }

        print(f"  [Proto C] {step_name}: enrolled={len(enrolled_ids)}, "
              f"Rank-1={known_rank1:.4f}, accept={known_acceptance:.4f}, "
              f"future_rej={future_rejection:.4f}, ext_rej={external_rejection:.4f}")

    results['score_distributions'] = score_distributions
    return results


# ============================================================
# 유틸: 결과 저장
# ============================================================

def save_results(results: Dict, output_path: str):
    """JSON 저장 (numpy 타입 변환 포함)"""
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"[SAVE] {output_path}")
