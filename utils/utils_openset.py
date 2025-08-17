# utils/utils_openset.py
"""오픈셋 관련 유틸리티 함수들"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple, Set
import os
from collections import defaultdict


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
def extract_features(model, paths: List[str], transform, device, batch_size: int = 64) -> np.ndarray:
    """이미지 경로에서 특징 추출"""
    if not paths:
        return np.array([])
        
    feats = []
    model.eval()
    
    for i in range(0, len(paths), batch_size):
        batch = []
        for p in paths[i:i+batch_size]:
            img = Image.open(p).convert('L')
            img = transform(img)
            batch.append(img)
        
        x = torch.stack(batch, dim=0).to(device)
        f = model.getFeatureCode(x)  # (B, D)
        f = F.normalize(f, p=2, dim=1)  # 코사인용 정규화
        feats.append(f.cpu())
    
    return torch.cat(feats, dim=0).numpy() if feats else np.array([])


# ⭐️ === 에너지 스코어 버전 추가 ===

@torch.no_grad()
def extract_scores_genuine_energy(model, ncm, dev_paths: List[str], dev_labels: List[int],
                                 transform, device, batch_size: int = 64) -> np.ndarray:
    """
    ⭐️ Genuine 샘플의 에너지 스코어 추출
    
    Returns:
        np.array: genuine 에너지 스코어들
    """
    if not dev_paths:
        return np.array([])
    
    model.eval()
    scores = []
    
    # 배치 처리를 위한 간단한 로더
    for i in range(0, len(dev_paths), batch_size):
        batch_paths = dev_paths[i:i+batch_size]
        batch_labels = dev_labels[i:i+batch_size]
        
        # 이미지 로드 및 변환
        batch_imgs = []
        for path in batch_paths:
            img = Image.open(path).convert('L')
            img = transform(img)
            batch_imgs.append(img)
        
        if batch_imgs:
            x = torch.stack(batch_imgs).to(device)
            
            # 특징 추출
            fe = model.getFeatureCode(x)
            
            # 에너지 스코어 계산
            gate_scores = ncm.compute_energy_score(fe)
            scores.extend(gate_scores.cpu().numpy())
    
    return np.array(scores)


@torch.no_grad()
def extract_scores_impostor_between_energy(model, ncm, dev_paths: List[str], dev_labels: List[int],
                                          transform, device, max_pairs: int = 2000) -> np.ndarray:
    """
    ⭐️ Between impostor의 에너지 스코어 (자기 클래스 제외)
    
    Returns:
        np.array: impostor 에너지 스코어들
    """
    if not dev_paths:
        return np.array([])
    
    by_class = defaultdict(list)
    for p, y in zip(dev_paths, dev_labels):
        by_class[int(y)].append(p)
    
    model.eval()
    scores = []
    
    # 각 클래스에서 샘플링
    for cls_id, cls_paths in by_class.items():
        if len(cls_paths) < 1:
            continue
        
        # 클래스당 최대 10개만 사용 (효율성)
        sample_paths = cls_paths[:min(10, len(cls_paths))]
        
        for path in sample_paths:
            img = Image.open(path).convert('L')
            img = transform(img)
            x = img.unsqueeze(0).to(device)
            
            # 특징 추출
            fe = model.getFeatureCode(x)
            
            # 자기 클래스 제외 에너지 (마스킹된 버전)
            label_tensor = torch.tensor([cls_id], device=device)
            gate_score = ncm.compute_energy_masked(fe, label_tensor)
            
            scores.append(gate_score.item())
            
            if len(scores) >= max_pairs:
                return np.array(scores)
    
    return np.array(scores)


@torch.no_grad()
def extract_scores_impostor_negref_energy(model, ncm, negref_file: str, transform, device,
                                         max_eval: int = 5000) -> np.ndarray:
    """
    ⭐️ NegRef의 에너지 스코어
    
    Returns:
        np.array: negref impostor 에너지 스코어들
    """
    if not negref_file or not os.path.exists(negref_file):
        return np.array([])
    
    paths, labels = load_paths_labels_from_txt(negref_file)
    
    if not paths:
        return np.array([])
    
    # 샘플링
    if len(paths) > max_eval:
        idx = np.random.choice(len(paths), max_eval, replace=False)
        paths = [paths[i] for i in idx]
    
    model.eval()
    scores = []
    
    # 배치 처리
    batch_size = 64
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i+batch_size]
        batch = []
        
        for p in batch_paths:
            img = Image.open(p).convert('L')
            img = transform(img)
            batch.append(img)
        
        if batch:
            x = torch.stack(batch).to(device)
            fe = model.getFeatureCode(x)
            gate_scores = ncm.compute_energy_score(fe)
            scores.extend(gate_scores.cpu().numpy())
    
    return np.array(scores)


# === 기존 최댓값 기반 함수들 (유지) ===

def extract_scores_genuine(model, ncm, dev_paths: List[str], dev_labels: List[int],
                          transform, device) -> np.ndarray:
    """같은 클래스 내 genuine 점수 (최댓값 방식)"""
    by_class = defaultdict(list)
    for p, y in zip(dev_paths, dev_labels):
        by_class[int(y)].append(p)
    
    scores = []
    for cls_id, cls_paths in by_class.items():
        if len(cls_paths) < 2:
            continue
            
        # 클래스의 프로토타입이 있는지 확인
        if cls_id not in ncm.class_means_dict:
            continue
            
        # 특징 추출
        feats = extract_features(model, cls_paths, transform, device)
        
        # 같은 클래스 내 페어 점수
        for i in range(len(feats)):
            for j in range(i+1, len(feats)):
                score = (feats[i] @ feats[j]).item()
                scores.append(score)
    
    return np.array(scores)


def extract_scores_impostor_between(model, ncm, dev_paths: List[str], dev_labels: List[int],
                                   transform, device, max_pairs: int = 2000) -> np.ndarray:
    """다른 클래스 간 impostor 점수 (최댓값 방식)"""
    by_class = defaultdict(list)
    for p, y in zip(dev_paths, dev_labels):
        by_class[int(y)].append(p)
    
    class_ids = list(by_class.keys())
    scores = []
    
    for i in range(len(class_ids)):
        for j in range(i+1, len(class_ids)):
            if len(scores) >= max_pairs:
                break
                
            # 각 클래스에서 첫 샘플만
            path_i = by_class[class_ids[i]][0]
            path_j = by_class[class_ids[j]][0]
            
            feat_i = extract_features(model, [path_i], transform, device)
            feat_j = extract_features(model, [path_j], transform, device)
            
            score = (feat_i @ feat_j.T).item()
            scores.append(score)
    
    return np.array(scores)


def extract_scores_impostor_unknown(model, ncm, txt_file: str, registered_users: Set[int],
                                   transform, device, max_eval: int = 3000) -> np.ndarray:
    """미등록 사용자의 impostor 점수 (최댓값 방식)"""
    paths, labels = load_paths_labels_excluding(txt_file, registered_users)
    
    if not paths:
        return np.array([])
    
    # 샘플링
    if len(paths) > max_eval:
        idx = np.random.choice(len(paths), max_eval, replace=False)
        paths = [paths[i] for i in idx]
    
    # 특징 추출
    feats = extract_features(model, paths, transform, device)
    
    # 프로토타입과의 최대 유사도
    if not ncm.class_means_dict:
        return np.array([])
    
    protos = []
    for cid, mu in ncm.class_means_dict.items():
        protos.append(mu.cpu().numpy())
    
    P = np.stack(protos, axis=0)  # (C, D)
    scores = (feats @ P.T).max(axis=1)  # 각 unknown의 최대 매칭
    
    return scores


def extract_scores_impostor_negref(model, ncm, negref_file: str, transform, device,
                                  max_eval: int = 5000) -> np.ndarray:
    """NegRef의 impostor 점수 (최댓값 방식)"""
    if not negref_file or not os.path.exists(negref_file):
        return np.array([])
    
    paths, labels = load_paths_labels_from_txt(negref_file)
    
    if not paths:
        return np.array([])
    
    # 샘플링
    if len(paths) > max_eval:
        idx = np.random.choice(len(paths), max_eval, replace=False)
        paths = [paths[i] for i in idx]
    
    # 특징 추출
    feats = extract_features(model, paths, transform, device)
    
    # 프로토타입과의 최대 유사도
    if not ncm.class_means_dict:
        return np.array([])
    
    protos = []
    for cid, mu in ncm.class_means_dict.items():
        protos.append(mu.cpu().numpy())
    
    P = np.stack(protos, axis=0)
    scores = (feats @ P.T).max(axis=1)
    
    return scores


def balance_impostor_scores(s_between: np.ndarray, s_unknown: np.ndarray, s_negref: np.ndarray,
                           ratio: Tuple[float, float, float] = (0.5, 0.0, 0.5),
                           total: int = 4000) -> np.ndarray:
    """impostor 점수 균형 맞추기 - None 안전 버전"""
    
    # 입력 검증 및 변환
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
    
    # 소스가 없으면 빈 배열 반환
    if not sources:
        return np.array([])
    
    # 소스가 하나면 그대로 반환
    if len(sources) == 1:
        arr = sources[0]
        return arr[:total] if len(arr) > total else arr
    
    # 가중치 정규화
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    # 각 소스에서 샘플링
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
def predict_batch(model, ncm, paths: List[str], transform, device, batch_size: int = 128) -> np.ndarray:
    """배치 예측"""
    if not paths:
        return np.array([])
    
    model.eval()
    preds = []
    
    for i in range(0, len(paths), batch_size):
        batch = []
        for p in paths[i:i+batch_size]:
            img = Image.open(p).convert('L')
            img = transform(img)
            batch.append(img)
        
        x = torch.stack(batch, dim=0).to(device)
        feat = model.getFeatureCode(x)
        pred = ncm.predict_openset(feat)
        preds.extend(pred.cpu().numpy())
    
    return np.array(preds)