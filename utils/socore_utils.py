# 🐋 scr/score_utils.py (신규 - 수정 버전)
"""
오픈셋 EER 계산을 위한 점수 추출 유틸리티
- genuine scores: 같은 클래스 점수
- impostor scores: 다른 클래스/unknown 점수
"""

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from pathlib import Path
from PIL import Image
from typing import Optional, List, Tuple, Dict, Set
from collections import defaultdict


def to_loader(dataset, batch_size=128, num_workers=0):
    """🐋 데이터셋을 DataLoader로 변환"""
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


def load_txt_dataset(txt_path: str) -> Tuple[List[str], List[int]]:
    """
    🐋 txt 파일에서 이미지 경로와 레이블 읽기
    
    Args:
        txt_path: train_palma.txt 또는 train_Tongji.txt 경로
        
    Returns:
        (paths, labels) 튜플
    """
    paths = []
    labels = []
    
    if not Path(txt_path).exists():
        print(f"⚠️ File not found: {txt_path}")
        return paths, labels
    
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) == 2:
                path, label = parts
                paths.append(path)
                labels.append(int(label))
    
    print(f"📂 Loaded {len(paths)} samples from {Path(txt_path).name}")
    return paths, labels


class SimpleImageDataset(Dataset):
    """🐋 간단한 이미지 데이터셋 (경로 리스트로부터)"""
    
    def __init__(self, paths: List[str], labels: List[int], transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        
        if self.transform:
            img = self.transform(img)
            
        return img, self.labels[idx]


@torch.no_grad()
def extract_scores_genuine(model, ncm, known_dev_paths: List[str], 
                          known_dev_labels: List[int], transform, device, 
                          batch_size=128) -> np.ndarray:
    """
    🐋 Known-Dev 샘플의 '자기 클래스' 점수 s_genuine 수집
    
    Args:
        model: 특징 추출 모델
        ncm: NCM classifier
        known_dev_paths: Dev 샘플 경로들
        known_dev_labels: Dev 샘플 레이블들
        transform: 이미지 전처리
        device: cuda/cpu
        
    Returns:
        np.array: genuine 점수들 (코사인 유사도)
    """
    if len(known_dev_paths) == 0:
        return np.array([])
    
    model.eval()
    scores = []
    
    # 데이터셋 생성
    dataset = SimpleImageDataset(known_dev_paths, known_dev_labels, transform)
    loader = to_loader(dataset, batch_size=batch_size)
    
    with torch.no_grad():
        for data, labels in loader:
            x = data.to(device)
            y = labels.to(device)
            
            # 특징 추출
            fe = model.getFeatureCode(x)  # (B, D), normalized if cosine
            
            # NCM 점수 계산
            if ncm.class_means is None or ncm.class_means.shape[0] == 0:
                continue
                
            S = fe @ ncm.class_means.to(device).T  # (B, C) 코사인 유사도
            
            # 자기 클래스 점수만 추출
            batch_size = len(y)
            for i in range(batch_size):
                class_id = y[i].item()
                # 클래스가 NCM에 존재하는지 확인
                if class_id < S.shape[1] and class_id in ncm.class_means_dict:
                    score = S[i, class_id].item()
                    scores.append(score)
    
    return np.array(scores)


@torch.no_grad()
def extract_scores_impostor_between_registered(model, ncm, known_dev_paths: List[str],
                                              known_dev_labels: List[int], transform, 
                                              device, max_pairs=2000) -> np.ndarray:
    """
    🐋 등록된 클래스 간 impostor 점수 (다른 클래스에 대한 점수)
    
    Returns:
        np.array: 등록 사용자 간 impostor 점수들
    """
    if len(known_dev_paths) == 0:
        return np.array([])
    
    model.eval()
    scores = []
    
    # 데이터셋 생성
    dataset = SimpleImageDataset(known_dev_paths, known_dev_labels, transform)
    loader = to_loader(dataset, batch_size=128)
    
    with torch.no_grad():
        for data, labels in loader:
            x = data.to(device)
            y = labels.to(device)
            
            # 특징 추출
            fe = model.getFeatureCode(x)
            
            # NCM 점수 계산
            if ncm.class_means is None or ncm.class_means.shape[0] == 0:
                continue
                
            S = fe @ ncm.class_means.to(device).T  # (B, C)
            
            # 다른 클래스들에 대한 점수 수집
            batch_size = len(y)
            for i in range(batch_size):
                true_class = y[i].item()
                
                # 자기 클래스를 제외한 모든 등록 클래스의 점수
                for class_id in ncm.class_means_dict.keys():
                    if class_id != true_class and class_id < S.shape[1]:
                        scores.append(S[i, class_id].item())
                        
                        if len(scores) >= max_pairs:
                            return np.array(scores)
    
    return np.array(scores)


@torch.no_grad()
def extract_scores_impostor_unknown(model, ncm, unknown_txt_path: str,
                                   registered_users: Set[int], transform,
                                   device, max_eval=5000) -> np.ndarray:
    """
    🐋 아직 등록 안 된 사용자들의 impostor 점수
    
    Args:
        unknown_txt_path: 전체 데이터셋 txt 경로 (train_palma.txt)
        registered_users: 현재 등록된 사용자 ID 집합
        
    Returns:
        np.array: 미등록 사용자의 impostor 점수들
    """
    # 전체 데이터 읽기
    all_paths, all_labels = load_txt_dataset(unknown_txt_path)
    
    if len(all_paths) == 0:
        return np.array([])
    
    # 등록 안 된 사용자만 필터링
    unknown_paths = []
    unknown_labels = []
    
    for path, label in zip(all_paths, all_labels):
        if label not in registered_users:
            unknown_paths.append(path)
            unknown_labels.append(label)
    
    if len(unknown_paths) == 0:
        print("⚠️ No unknown users found")
        return np.array([])
    
    # 샘플링 (너무 많으면)
    if max_eval and len(unknown_paths) > max_eval:
        indices = random.sample(range(len(unknown_paths)), max_eval)
        unknown_paths = [unknown_paths[i] for i in indices]
        unknown_labels = [unknown_labels[i] for i in indices]
    
    print(f"📊 Using {len(unknown_paths)} unknown samples for impostor scores")
    
    model.eval()
    scores = []
    
    # 데이터셋 생성
    dataset = SimpleImageDataset(unknown_paths, unknown_labels, transform)
    loader = to_loader(dataset, batch_size=128)
    
    with torch.no_grad():
        for data, labels in loader:
            x = data.to(device)
            
            # 특징 추출
            fe = model.getFeatureCode(x)
            
            # NCM 점수 계산
            if ncm.class_means is None or ncm.class_means.shape[0] == 0:
                continue
                
            S = fe @ ncm.class_means.to(device).T  # (B, C)
            
            # 최고 점수 (가장 가까운 등록 클래스)
            max_scores = S.max(dim=1).values  # (B,)
            scores.extend(max_scores.cpu().numpy().tolist())
    
    return np.array(scores)


@torch.no_grad()
def extract_scores_impostor_negref(model, ncm, negref_txt_path: str,
                                  transform, device, max_eval=5000) -> np.ndarray:
    """
    🐋 Negative Reference (train_Tongji.txt)의 impostor 점수
    
    Args:
        negref_txt_path: negative_samples_file 경로
        
    Returns:
        np.array: negative reference impostor 점수들
    """
    # Negative 데이터 읽기
    neg_paths, neg_labels = load_txt_dataset(negref_txt_path)
    
    if len(neg_paths) == 0:
        return np.array([])
    
    # 샘플링
    if max_eval and len(neg_paths) > max_eval:
        indices = random.sample(range(len(neg_paths)), max_eval)
        neg_paths = [neg_paths[i] for i in indices]
        neg_labels = [neg_labels[i] for i in indices]
    
    print(f"📊 Using {len(neg_paths)} NegRef samples for impostor scores")
    
    model.eval()
    scores = []
    
    # 데이터셋 생성
    dataset = SimpleImageDataset(neg_paths, neg_labels, transform)
    loader = to_loader(dataset, batch_size=128)
    
    with torch.no_grad():
        for data, labels in loader:
            x = data.to(device)
            
            # 특징 추출
            fe = model.getFeatureCode(x)
            
            # NCM 점수 계산
            if ncm.class_means is None or ncm.class_means.shape[0] == 0:
                continue
                
            S = fe @ ncm.class_means.to(device).T  # (B, C)
            
            # 최고 점수 (가장 가까운 등록 클래스)
            max_scores = S.max(dim=1).values  # (B,)
            scores.extend(max_scores.cpu().numpy().tolist())
    
    return np.array(scores)


def balance_impostor_scores(scores_between: np.ndarray,
                           scores_unknown: np.ndarray, 
                           scores_negref: np.ndarray,
                           ratio: Tuple[float, float, float] = (0.25, 0.25, 0.5)) -> np.ndarray:
    """
    🐋 세 종류의 impostor 점수를 균형있게 섞기
    
    Args:
        scores_between: 등록 클래스 간 impostor
        scores_unknown: 미등록 사용자 impostor
        scores_negref: NegRef impostor
        ratio: (between, unknown, negref) 비율
        
    Returns:
        np.array: 균형잡힌 impostor 점수
    """
    available = []
    if len(scores_between) > 0:
        available.append(('between', scores_between))
    if len(scores_unknown) > 0:
        available.append(('unknown', scores_unknown))
    if len(scores_negref) > 0:
        available.append(('negref', scores_negref))
    
    if len(available) == 0:
        return np.array([])
    
    # 🐋 가용한 소스에 따라 비율 재조정
    if len(available) == 1:
        return available[0][1]
    
    elif len(available) == 2:
        # 두 개만 있으면 50:50
        source1, scores1 = available[0]
        source2, scores2 = available[1]
        
        min_size = min(len(scores1), len(scores2))
        sampled1 = np.random.choice(scores1, min_size, replace=False)
        sampled2 = np.random.choice(scores2, min_size, replace=False)
        
        balanced = np.concatenate([sampled1, sampled2])
        print(f"📊 Balanced impostor: {source1}({min_size}) + {source2}({min_size})")
        
    else:
        # 세 개 모두 있으면 ratio 사용
        total_size = min(
            sum(len(s) for _, s in available),
            10000  # 최대 10000개
        )
        
        n_between = int(total_size * ratio[0])
        n_unknown = int(total_size * ratio[1])  
        n_negref = int(total_size * ratio[2])
        
        # 실제 가용 개수로 조정
        n_between = min(n_between, len(scores_between))
        n_unknown = min(n_unknown, len(scores_unknown))
        n_negref = min(n_negref, len(scores_negref))
        
        sampled = []
        if n_between > 0:
            sampled.append(np.random.choice(scores_between, n_between, replace=False))
        if n_unknown > 0:
            sampled.append(np.random.choice(scores_unknown, n_unknown, replace=False))
        if n_negref > 0:
            sampled.append(np.random.choice(scores_negref, n_negref, replace=False))
        
        balanced = np.concatenate(sampled)
        print(f"📊 Balanced impostor: between({n_between}) + unknown({n_unknown}) + negref({n_negref})")
    
    return balanced


def compute_score_statistics(genuine_scores: np.ndarray, 
                            impostor_scores: np.ndarray) -> dict:
    """
    🐋 점수 분포 통계 계산
    
    Returns:
        dict: 평균, 표준편차, 분리도 등
    """
    stats = {
        'genuine_mean': float(np.mean(genuine_scores)) if len(genuine_scores) > 0 else 0,
        'genuine_std': float(np.std(genuine_scores)) if len(genuine_scores) > 0 else 0,
        'genuine_min': float(np.min(genuine_scores)) if len(genuine_scores) > 0 else 0,
        'genuine_max': float(np.max(genuine_scores)) if len(genuine_scores) > 0 else 0,
        'genuine_count': len(genuine_scores),
        
        'impostor_mean': float(np.mean(impostor_scores)) if len(impostor_scores) > 0 else 0,
        'impostor_std': float(np.std(impostor_scores)) if len(impostor_scores) > 0 else 0,
        'impostor_min': float(np.min(impostor_scores)) if len(impostor_scores) > 0 else 0,
        'impostor_max': float(np.max(impostor_scores)) if len(impostor_scores) > 0 else 0,
        'impostor_count': len(impostor_scores),
    }
    
    # 🐋 분리도 (클수록 좋음)
    if len(genuine_scores) > 0 and len(impostor_scores) > 0:
        stats['separation'] = stats['genuine_mean'] - stats['impostor_mean']
        # D-prime (신호 검출 이론)
        pooled_std = np.sqrt((stats['genuine_std']**2 + stats['impostor_std']**2) / 2 + 1e-6)
        stats['d_prime'] = stats['separation'] / pooled_std
    else:
        stats['separation'] = 0
        stats['d_prime'] = 0
    
    return stats


# 🐋 Dev/Train 분리 헬퍼 함수
def split_user_data(user_paths: List[str], user_labels: List[int], 
                   dev_ratio: float = 0.2) -> Tuple[List, List, List, List]:
    """
    🐋 사용자 데이터를 Train/Dev로 분리
    
    Returns:
        (train_paths, train_labels, dev_paths, dev_labels)
    """
    n = len(user_paths)
    indices = np.random.permutation(n)
    n_dev = max(2, int(n * dev_ratio))  # 최소 2개는 Dev
    
    dev_indices = indices[:n_dev]
    train_indices = indices[n_dev:]
    
    dev_paths = [user_paths[i] for i in dev_indices]
    dev_labels = [user_labels[i] for i in dev_indices]
    train_paths = [user_paths[i] for i in train_indices]
    train_labels = [user_labels[i] for i in train_indices]
    
    return train_paths, train_labels, dev_paths, dev_labels


# 🐋 테스트 코드
if __name__ == "__main__":
    print("=== Score Utils Test ===")
    
    # 가짜 데이터로 테스트
    genuine = np.random.normal(0.8, 0.1, 1000)  # 높은 점수
    impostor = np.random.normal(0.3, 0.15, 1000)  # 낮은 점수
    
    stats = compute_score_statistics(genuine, impostor)
    print(f"Genuine: {stats['genuine_mean']:.3f} ± {stats['genuine_std']:.3f}")
    print(f"Impostor: {stats['impostor_mean']:.3f} ± {stats['impostor_std']:.3f}")
    print(f"Separation: {stats['separation']:.3f}")
    print(f"D-prime: {stats['d_prime']:.3f}")
    
    # 균형 테스트
    between_scores = np.random.normal(0.5, 0.1, 500)
    unknown_scores = np.random.normal(0.4, 0.1, 800)
    negref_scores = np.random.normal(0.2, 0.1, 1500)
    
    balanced = balance_impostor_scores(between_scores, unknown_scores, negref_scores)
    print(f"\nBalanced shape: {balanced.shape}")