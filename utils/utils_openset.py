# utils/utils_openset.py
"""오픈셋 관련 유틸리티 함수들 (TTA 지원, 채널 유연성 추가)"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple, Set, Dict, Optional, Union
import os
from collections import defaultdict, Counter
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode  # 🥩 deprecation 방지
import random

# 🥩 시드 고정 (재현성)
def set_seed(seed: int = 42):
    """실험 재현성을 위한 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 🥩 === 채널 처리 헬퍼 ===
def _open_with_channels(path: str, channels: int) -> Image.Image:
    """이미지를 열고 채널 수에 맞게 변환 (I/O 안전성)"""
    with Image.open(path) as img:  # 🥩 컨텍스트 매니저로 안전성 향상
        if channels == 1:
            return img.convert('L')
        else:
            return img.convert('RGB')


def split_user_data(paths: List[str], labels: List[int], 
                   dev_ratio: float = 0.2, min_dev: int = 2) -> Tuple:
    """사용자 데이터를 Train/Dev로 분리"""
    idx = np.arange(len(paths))
    np.random.shuffle(idx)
    n_dev = max(min_dev, int(len(paths) * dev_ratio))
    
    dev_idx = idx[:n_dev]
    train_idx = idx[n_dev:]
    
    train_paths = [paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    dev_paths = [paths[i] for i in dev_idx]
    dev_labels = [labels[i] for i in dev_idx]
    
    return train_paths, train_labels, dev_paths, dev_labels


def load_paths_labels_from_txt(txt_file: str) -> Tuple[List[str], List[int]]:
    """txt 파일에서 경로와 레이블 읽기"""
    paths, labels = [], []
    if not os.path.exists(txt_file):
        return paths, labels
        
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) == 2:
                paths.append(parts[0])
                labels.append(int(parts[1]))
    return paths, labels


def load_paths_labels_excluding(txt_file: str, exclude_labels: Set[int]) -> Tuple[List[str], List[int]]:
    """특정 레이블을 제외하고 로드"""
    all_paths, all_labels = load_paths_labels_from_txt(txt_file)
    paths, labels = [], []
    
    for p, l in zip(all_paths, all_labels):
        if l not in exclude_labels:
            paths.append(p)
            labels.append(l)
    
    return paths, labels


@torch.no_grad()
def extract_features(model, paths: List[str], transform, device, 
                    batch_size: int = 64, channels: int = 1) -> np.ndarray:
    """
    이미지 경로에서 특징 추출 (L2 정규화 보장)
    
    Returns:
        np.ndarray: L2 정규화된 특징 벡터 (N, D)
    """
    if not paths:
        return np.array([])
        
    feats = []
    model.eval()
    
    for i in range(0, len(paths), batch_size):
        batch = []
        for p in paths[i:i+batch_size]:
            try:
                img = _open_with_channels(p, channels)
                img = transform(img)
                batch.append(img)
            except Exception as e:
                print(f"⚠️ Failed to load {p}: {e}")
                continue
        
        if not batch:
            continue
            
        x = torch.stack(batch, dim=0).to(device)
        f = model.getFeatureCode(x)  # (B, D)
        # 추가 정규화 (안전성 보장)
        f = F.normalize(f, p=2, dim=1, eps=1e-12)
        feats.append(f.cpu())
    
    return torch.cat(feats, dim=0).numpy() if feats else np.array([])


def get_light_augmentation(strength: float = 0.5, img_size: int = 128) -> T.Compose:
    """
    인증용 가벼운 증강 (학습 증강보다 훨씬 약함)
    """
    if strength <= 0:
        return T.Lambda(lambda x: x)
    
    transforms = []
    
    if strength > 0.2:
        transforms.append(T.RandomRotation(degrees=5 * strength))
    
    if strength > 0.3:
        transforms.append(T.RandomPerspective(
            distortion_scale=0.05 * strength, 
            p=0.5
        ))
    
    if strength > 0.4:
        transforms.append(T.RandomResizedCrop(
            size=img_size,
            scale=(1.0 - 0.05*strength, 1.0),
            ratio=(1.0, 1.0),
            interpolation=InterpolationMode.BICUBIC  # 🥩 deprecation 방지
        ))
    
    return T.Compose(transforms) if transforms else T.Lambda(lambda x: x)


# 🥩 === TTA 메인 함수 ===
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
    verbose: bool = False  # 🥩 디버그용
) -> Union[np.ndarray, Tuple[np.ndarray, List[Dict]]]:
    """
    Test-Time Augmentation 기반 Multi-view Consensus 예측
    """
    if not paths:
        return (np.array([]), []) if return_details else np.array([])
    
    # n_views=1이면 기존 함수 호출
    if n_views <= 1:
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
    
    # 🥩 메모리 여유화: n_views가 많으면 배치 크기 조정
    eff_batch_size = max(1, batch_size // max(1, n_views // 2))
    
    # 합의 기준 설정
    if agree_k <= 0:
        agree_k = (n_views + 1) // 2
    else:
        agree_k = min(max(agree_k, 1), n_views)
    
    light_aug = get_light_augmentation(aug_strength, img_size)
    
    model.eval()
    all_predictions = []
    all_details = []
    
    # 🥩 디버그용 통계
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
                print(f"⚠️ Failed to load {path}: {e}")
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
        # 🥩 TTA 경로에서도 L2 정규화 통일
        features = F.normalize(features, p=2, dim=2, eps=1e-12)
        
        for i in range(B):
            view_scores = []
            view_preds = []
            view_accepts = []
            
            for v in range(n_views):
                feat_v = features[i, v:v+1]
                
                # 중복 forward 제거: 한 번만 호출
                scores = ncm.forward(feat_v)  # (1, C)
                
                # 게이트 스코어 계산
                if hasattr(ncm, 'use_energy') and ncm.use_energy:
                    gate_score = ncm.compute_energy_score(feat_v).item()
                    # 🥩 에너지 점수 방향성 확인 (클수록 genuine 가정)
                    # 만약 작을수록 genuine이면 여기서 gate_score = -gate_score
                else:
                    gate_score = scores.max().item() if scores.numel() > 0 else -1000.0
                
                view_scores.append(gate_score)
                
                # 클래스 예측 (이미 계산된 scores 재사용)
                if scores.numel() > 0:
                    pred = scores.argmax(dim=1).item()
                else:
                    pred = -1
                view_preds.append(pred)
                
                tau_s = ncm.tau_s if hasattr(ncm, 'tau_s') and ncm.tau_s is not None else 0.0
                # 🥩 임계치 방향성: 점수가 클수록 genuine
                accept = gate_score >= tau_s
                view_accepts.append(accept)
            
            accept_count = sum(view_accepts)
            gate_consensus = accept_count >= agree_k
            
            # 수락된 뷰만 투표
            valid_preds = [p for p, a in zip(view_preds, view_accepts) if a]
            
            if valid_preds:
                pred_counter = Counter(valid_preds)
                most_common_class, max_count = pred_counter.most_common(1)[0]
                class_consensus = max_count > len(valid_preds) / 2
                disagreement_rate = 1 - (max_count / len(valid_preds))
            else:
                most_common_class = -1
                class_consensus = False
                disagreement_rate = 1.0
            
            # 🥩 디버그 통계 수집
            if verbose:
                accept_ratio = accept_count / n_views
                all_accept_ratios.append(accept_ratio)
                all_disagreement_rates.append(disagreement_rate)
            
            if gate_consensus and class_consensus:
                final_pred = most_common_class
                reject_reason = None
            else:
                final_pred = -1
                if not gate_consensus and not class_consensus:
                    reject_reason = "both"
                elif not gate_consensus:
                    reject_reason = "gate"
                else:
                    reject_reason = "class"
            
            all_predictions.append(final_pred)
            
            if return_details:
                all_details.append({
                    'view_scores': view_scores,
                    'view_preds': view_preds,
                    'agree_count': accept_count,
                    'agree_k': agree_k,
                    'final_decision': (final_pred != -1, final_pred),
                    'reject_reason': reject_reason,
                    # 🥩 추가 디버그 정보
                    'accept_ratio': accept_count / n_views,
                    'disagreement_rate': disagreement_rate,
                    'score_stats': {
                        'median': np.median(view_scores),
                        'std': np.std(view_scores)
                    }
                })
    
    # 🥩 디버그 출력
    if verbose and all_accept_ratios:
        print(f"📊 TTA Statistics:")
        print(f"   Avg accept ratio: {np.mean(all_accept_ratios):.2%}")
        print(f"   Avg disagreement: {np.mean(all_disagreement_rates):.2%}")
        print(f"   Score spread: {np.mean([d['score_stats']['std'] for d in all_details if 'score_stats' in d]):.4f}")
    
    predictions = np.array(all_predictions)
    
    if return_details:
        return predictions, all_details
    return predictions


# ⭐️ === 에너지 스코어 버전 ===

@torch.no_grad()
def extract_scores_genuine_energy(model, ncm, dev_paths: List[str], dev_labels: List[int],
                                 transform, device, batch_size: int = 64,
                                 channels: int = 1) -> np.ndarray:
    """
    ⭐️ Genuine 샘플의 에너지 스코어 추출
    """
    if not dev_paths:
        return np.array([])
    
    model.eval()
    scores = []
    
    for i in range(0, len(dev_paths), batch_size):
        batch_paths = dev_paths[i:i+batch_size]
        batch_imgs = []
        
        for path in batch_paths:
            img = _open_with_channels(path, channels)
            img = transform(img)
            batch_imgs.append(img)
        
        if batch_imgs:
            x = torch.stack(batch_imgs).to(device)
            fe = model.getFeatureCode(x)
            # 🥩 정규화 추가
            fe = F.normalize(fe, p=2, dim=1, eps=1e-12)
            
            gate_scores = ncm.compute_energy_score(fe)
            scores.extend(gate_scores.cpu().numpy())
    
    return np.array(scores)


@torch.no_grad()
def extract_scores_impostor_between_energy(model, ncm, dev_paths: List[str], dev_labels: List[int],
                                          transform, device, max_pairs: int = 2000,
                                          channels: int = 1) -> np.ndarray:
    """
    ⭐️ Between impostor의 에너지 스코어 (자기 클래스 제외)
    """
    if not dev_paths:
        return np.array([])
    
    by_class = defaultdict(list)
    for p, y in zip(dev_paths, dev_labels):
        by_class[int(y)].append(p)
    
    model.eval()
    scores = []
    
    for cls_id, cls_paths in by_class.items():
        if len(cls_paths) < 1:
            continue
        
        # 샘플 다양화
        sample_paths = cls_paths[:min(10, len(cls_paths))]
        if len(sample_paths) > 5:
            sample_paths = random.sample(sample_paths, 5)
        
        for path in sample_paths:
            img = _open_with_channels(path, channels)
            img = transform(img)
            x = img.unsqueeze(0).to(device)
            
            fe = model.getFeatureCode(x)
            # 🥩 정규화 추가
            fe = F.normalize(fe, p=2, dim=1, eps=1e-12)
            
            label_tensor = torch.tensor([cls_id], device=device)
            if hasattr(ncm, 'compute_energy_masked'):
                gate_score = ncm.compute_energy_masked(fe, label_tensor).item()
            else:
                gate_score = ncm.compute_energy_score(fe).item()
            
            scores.append(gate_score)
            
            if len(scores) >= max_pairs:
                return np.array(scores)
    
    return np.array(scores)


@torch.no_grad()
def extract_scores_impostor_negref_energy(model, ncm, negref_file: str, transform, device,
                                         max_eval: int = 5000, channels: int = 1) -> np.ndarray:
    """
    ⭐️ NegRef의 에너지 스코어
    """
    if not negref_file or not os.path.exists(negref_file):
        return np.array([])
    
    paths, labels = load_paths_labels_from_txt(negref_file)
    
    if not paths:
        return np.array([])
    
    if len(paths) > max_eval:
        idx = np.random.choice(len(paths), max_eval, replace=False)
        paths = [paths[i] for i in idx]
    
    model.eval()
    scores = []
    
    batch_size = 64
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i+batch_size]
        batch = []
        
        for p in batch_paths:
            img = _open_with_channels(p, channels)
            img = transform(img)
            batch.append(img)
        
        if batch:
            x = torch.stack(batch).to(device)
            fe = model.getFeatureCode(x)
            # 🥩 정규화 추가
            fe = F.normalize(fe, p=2, dim=1, eps=1e-12)
            
            gate_scores = ncm.compute_energy_score(fe)
            scores.extend(gate_scores.cpu().numpy())
    
    return np.array(scores)


# 🥩 === TTA 점수 추출 함수들 ===

@torch.no_grad()
def extract_scores_genuine_tta(
    model, ncm, dev_paths: List[str], dev_labels: List[int],
    transform, device, n_views: int = 3, include_original: bool = True,
    aug_strength: float = 0.5, aggregation: str = 'median',
    img_size: int = 128, channels: int = 1
) -> np.ndarray:
    """TTA genuine 점수"""
    if not dev_paths:
        return np.array([])
    
    light_aug = get_light_augmentation(aug_strength, img_size)
    
    model.eval()
    scores = []
    
    for path, label in zip(dev_paths, dev_labels):
        if hasattr(ncm, 'class_means_dict') and label not in ncm.class_means_dict:
            continue
        
        try:
            img = _open_with_channels(path, channels)
        except:
            continue
        
        views = []
        if include_original:
            views.append(transform(img))
            for _ in range(n_views - 1):
                aug_img = light_aug(img) if aug_strength > 0 else img
                views.append(transform(aug_img))
        else:
            for _ in range(n_views):
                aug_img = light_aug(img) if aug_strength > 0 else img
                views.append(transform(aug_img))
        
        x = torch.stack(views).to(device)
        features = model.getFeatureCode(x)
        # 🥩 정규화 추가
        features = F.normalize(features, p=2, dim=1, eps=1e-12)
        
        view_scores = []
        for i in range(n_views):
            feat_i = features[i:i+1]
            
            if hasattr(ncm, 'use_energy') and ncm.use_energy:
                score = ncm.compute_energy_score(feat_i).item()
            else:
                scores_i = ncm.forward(feat_i)
                score = scores_i.max().item() if scores_i.numel() > 0 else -1000
            
            view_scores.append(score)
        
        if view_scores:
            if aggregation == 'median':
                final_score = np.median(view_scores)
            else:
                final_score = np.mean(view_scores)
            scores.append(final_score)
    
    return np.array(scores)


@torch.no_grad()
def extract_scores_impostor_between_tta(
    model, ncm, dev_paths: List[str], dev_labels: List[int],
    transform, device, n_views: int = 3, include_original: bool = True,
    aug_strength: float = 0.5, aggregation: str = 'median',
    max_pairs: int = 2000, img_size: int = 128, channels: int = 1
) -> np.ndarray:
    """TTA between impostor"""
    if not dev_paths:
        return np.array([])
    
    by_class = defaultdict(list)
    for p, l in zip(dev_paths, dev_labels):
        by_class[int(l)].append(p)
    
    light_aug = get_light_augmentation(aug_strength, img_size)
    
    model.eval()
    scores = []
    
    for cls_id, cls_paths in by_class.items():
        if len(scores) >= max_pairs:
            break
        
        sample_paths = cls_paths[:min(5, len(cls_paths))]
        if len(sample_paths) > 3:
            sample_paths = random.sample(sample_paths, 3)
        
        for path in sample_paths:
            if len(scores) >= max_pairs:
                break
            
            try:
                img = _open_with_channels(path, channels)
            except:
                continue
            
            views = []
            if include_original:
                views.append(transform(img))
                for _ in range(n_views - 1):
                    aug_img = light_aug(img) if aug_strength > 0 else img
                    views.append(transform(aug_img))
            else:
                for _ in range(n_views):
                    aug_img = light_aug(img) if aug_strength > 0 else img
                    views.append(transform(aug_img))
            
            x = torch.stack(views).to(device)
            features = model.getFeatureCode(x)
            # 🥩 정규화 추가
            features = F.normalize(features, p=2, dim=1, eps=1e-12)
            
            view_scores = []
            for i in range(n_views):
                feat_i = features[i:i+1]
                
                if hasattr(ncm, 'use_energy') and ncm.use_energy:
                    label_tensor = torch.tensor([cls_id], device=device)
                    if hasattr(ncm, 'compute_energy_masked'):
                        score = ncm.compute_energy_masked(feat_i, label_tensor).item()
                    else:
                        score = ncm.compute_energy_score(feat_i).item()
                else:
                    scores_i = ncm.forward(feat_i)
                    if cls_id < scores_i.shape[1]:
                        scores_i[0, cls_id] = -float('inf')
                    score = scores_i.max().item()
                
                view_scores.append(score)
            
            if aggregation == 'median':
                final_score = np.median(view_scores)
            else:
                final_score = np.mean(view_scores)
            scores.append(final_score)
    
    return np.array(scores)


@torch.no_grad()
def extract_scores_impostor_negref_tta(
    model, ncm, negref_file: str, transform, device,
    n_views: int = 3, include_original: bool = True,
    aug_strength: float = 0.5, aggregation: str = 'median',
    max_eval: int = 5000, img_size: int = 128, channels: int = 1
) -> np.ndarray:
    """TTA negref impostor"""
    paths, _ = load_paths_labels_from_txt(negref_file)
    if not paths:
        return np.array([])
    
    if len(paths) > max_eval:
        idx = np.random.choice(len(paths), max_eval, replace=False)
        paths = [paths[i] for i in idx]
    
    light_aug = get_light_augmentation(aug_strength, img_size)
    
    model.eval()
    scores = []
    
    for path in paths[:max_eval]:
        try:
            img = _open_with_channels(path, channels)
        except:
            continue
        
        views = []
        if include_original:
            views.append(transform(img))
            for _ in range(n_views - 1):
                aug_img = light_aug(img) if aug_strength > 0 else img
                views.append(transform(aug_img))
        else:
            for _ in range(n_views):
                aug_img = light_aug(img) if aug_strength > 0 else img
                views.append(transform(aug_img))
        
        x = torch.stack(views).to(device)
        features = model.getFeatureCode(x)
        # 🥩 정규화 추가
        features = F.normalize(features, p=2, dim=1, eps=1e-12)
        
        view_scores = []
        for i in range(n_views):
            feat_i = features[i:i+1]
            
            if hasattr(ncm, 'use_energy') and ncm.use_energy:
                score = ncm.compute_energy_score(feat_i).item()
            else:
                scores_i = ncm.forward(feat_i)
                score = scores_i.max().item() if scores_i.numel() > 0 else -1000
            
            view_scores.append(score)
        
        if aggregation == 'median':
            final_score = np.median(view_scores)
        else:
            final_score = np.mean(view_scores)
        scores.append(final_score)
    
    return np.array(scores[:max_eval])


# === 기존 최댓값 기반 함수들 (개선) ===

def extract_scores_genuine(model, ncm, dev_paths: List[str], dev_labels: List[int],
                          transform, device, channels: int = 1) -> np.ndarray:
    """MAX 모드 genuine 점수 (게이트 스코어)"""
    if not dev_paths:
        return np.array([])
    
    model.eval()
    scores = []
    
    batch_size = 64
    for i in range(0, len(dev_paths), batch_size):
        batch_paths = dev_paths[i:i+batch_size]
        
        feats = extract_features(model, batch_paths, transform, device, 
                                batch_size=32, channels=channels)
        if len(feats) == 0:
            continue
        
        feats_tensor = torch.from_numpy(feats).to(device)
        ncm_scores = ncm.forward(feats_tensor)
        
        if ncm_scores.numel() > 0:
            max_scores = ncm_scores.max(dim=1).values
            scores.extend(max_scores.cpu().numpy())
    
    return np.array(scores)


def extract_scores_impostor_between(model, ncm, dev_paths: List[str], dev_labels: List[int],
                                   transform, device, max_pairs: int = 2000,
                                   channels: int = 1) -> np.ndarray:
    """MAX 모드 between impostor (자기 클래스 제외)"""
    if not dev_paths:
        return np.array([])
    
    by_class = defaultdict(list)
    for p, y in zip(dev_paths, dev_labels):
        by_class[int(y)].append(p)
    
    model.eval()
    scores = []
    
    for cls_id, cls_paths in by_class.items():
        if len(scores) >= max_pairs:
            break
        
        sample_paths = cls_paths[:min(10, len(cls_paths))]
        if len(sample_paths) > 5:
            sample_paths = random.sample(sample_paths, 5)
        
        for path in sample_paths:
            if len(scores) >= max_pairs:
                break
            
            feat = extract_features(model, [path], transform, device, channels=channels)
            if len(feat) == 0:
                continue
            
            feat_tensor = torch.from_numpy(feat).to(device)
            ncm_scores = ncm.forward(feat_tensor)
            
            if ncm_scores.numel() > 0 and cls_id < ncm_scores.shape[1]:
                ncm_scores[0, cls_id] = -float('inf')
                max_score = ncm_scores.max().item()
                scores.append(max_score)
    
    return np.array(scores)


def extract_scores_impostor_unknown(model, ncm, txt_file: str, registered_users: Set[int],
                                   transform, device, max_eval: int = 3000,
                                   channels: int = 1) -> np.ndarray:
    """미등록 사용자의 impostor 점수 (최댓값 방식)"""
    paths, labels = load_paths_labels_excluding(txt_file, registered_users)
    
    if not paths:
        return np.array([])
    
    if len(paths) > max_eval:
        idx = np.random.choice(len(paths), max_eval, replace=False)
        paths = [paths[i] for i in idx]
    
    feats = extract_features(model, paths, transform, device, channels=channels)
    
    if len(feats) == 0:
        return np.array([])
    
    if not hasattr(ncm, 'class_means_dict') or not ncm.class_means_dict:
        return np.array([])
    
    feats_tensor = torch.from_numpy(feats).to(device)
    ncm_scores = ncm.forward(feats_tensor)
    
    if ncm_scores.numel() > 0:
        max_scores = ncm_scores.max(dim=1).values
        return max_scores.cpu().numpy()
    
    return np.array([])


def extract_scores_impostor_negref(model, ncm, negref_file: str, transform, device,
                                  max_eval: int = 5000, channels: int = 1) -> np.ndarray:
    """NegRef의 impostor 점수 (최댓값 방식)"""
    if not negref_file or not os.path.exists(negref_file):
        return np.array([])
    
    paths, labels = load_paths_labels_from_txt(negref_file)
    
    if not paths:
        return np.array([])
    
    if len(paths) > max_eval:
        idx = np.random.choice(len(paths), max_eval, replace=False)
        paths = [paths[i] for i in idx]
    
    feats = extract_features(model, paths, transform, device, channels=channels)
    
    if len(feats) == 0:
        return np.array([])
    
    if not hasattr(ncm, 'class_means_dict') or not ncm.class_means_dict:
        return np.array([])
    
    feats_tensor = torch.from_numpy(feats).to(device)
    ncm_scores = ncm.forward(feats_tensor)
    
    if ncm_scores.numel() > 0:
        max_scores = ncm_scores.max(dim=1).values
        return max_scores.cpu().numpy()
    
    return np.array([])


# ✖️ 중복 제거: balance_impostor_scores는 하나만 유지
def balance_impostor_scores(s_between: np.ndarray, s_unknown: np.ndarray, s_negref: np.ndarray,
                           ratio: Tuple[float, float, float] = (0.5, 0.0, 0.5),
                           total: int = 4000) -> np.ndarray:
    """impostor 점수 균형 맞추기 - None 안전 버전"""
    
    sources = []
    weights = []
    
    if s_between is not None and isinstance(s_between, (list, np.ndarray)):
        arr = np.asarray(s_between)
        if arr.size > 0:
            sources.append(arr)
            weights.append(ratio[0])
    
    if s_unknown is not None and isinstance(s_unknown, (list, np.ndarray)):
        arr = np.asarray(s_unknown)
        if arr.size > 0:
            sources.append(arr)
            weights.append(ratio[1])
    
    if s_negref is not None and isinstance(s_negref, (list, np.ndarray)):
        arr = np.asarray(s_negref)
        if arr.size > 0:
            sources.append(arr)
            weights.append(ratio[2])
    
    if not sources:
        return np.array([])
    
    if len(sources) == 1:
        arr = sources[0]
        return arr[:total] if len(arr) > total else arr
    
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    out = []
    for arr, weight in zip(sources, normalized_weights):
        k = int(total * weight)
        if k == 0:
            continue
            
        if len(arr) <= k:
            out.append(arr)
        else:
            out.append(np.random.choice(arr, k, replace=False))
    
    if not out:
        return np.array([])
    
    return np.concatenate(out)


@torch.no_grad()
def predict_batch(model, ncm, paths: List[str], transform, device, 
                 batch_size: int = 128, channels: int = 1) -> np.ndarray:
    """배치 예측 (기존 단일 뷰)"""
    if not paths:
        return np.array([])
    
    model.eval()
    preds = []
    
    for i in range(0, len(paths), batch_size):
        batch = []
        for p in paths[i:i+batch_size]:
            img = _open_with_channels(p, channels)
            img = transform(img)
            batch.append(img)
        
        x = torch.stack(batch, dim=0).to(device)
        feat = model.getFeatureCode(x)
        # 🥩 정규화 추가
        feat = F.normalize(feat, p=2, dim=1, eps=1e-12)
        pred = ncm.predict_openset(feat)
        preds.extend(pred.cpu().numpy())
    
    return np.array(preds)