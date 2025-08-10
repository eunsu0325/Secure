# 🐋 scr/utils_openset.py (신규)
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


def extract_scores_genuine(model, ncm, dev_paths: List[str], dev_labels: List[int],
                          transform, device) -> np.ndarray:
    """같은 클래스 내 genuine 점수"""
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
    """다른 클래스 간 impostor 점수"""
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
    """미등록 사용자의 impostor 점수"""
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
    """NegRef의 impostor 점수"""
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
                           ratio: Tuple[float, float, float] = (0.2, 0.5, 0.3),
                           total: int = 4000) -> np.ndarray:
    """impostor 점수 균형 맞추기"""
    pools = [np.asarray(s_between), np.asarray(s_unknown), np.asarray(s_negref)]
    want = [int(total * r) for r in ratio]
    out = []
    
    for arr, k in zip(pools, want):
        if arr.size == 0:
            continue
        if arr.size <= k:
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