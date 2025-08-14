#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Supervised Contrastive Replay (SCR) Training Script with Open-set Support
CCNet + SCR for Continual Learning
"""

import os
import argparse
import time
import numpy as np
from collections import defaultdict
from typing import Dict, List
import json
import random

import torch
from torch.utils.data import DataLoader, Subset

# Project imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ConfigParser
# 👻 사전훈련 로더 import 추가
from utils.pretrained_loader import PretrainedLoader  # 👻

# 🎨 시각화 관련 import 추가
import matplotlib
matplotlib.use('Agg')  # 서버 환경용
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from PIL import Image
import torch.nn.functional as F
# 🎨

from models import ccnet, MyDataset, get_scr_transforms
from scr import (
    ExperienceStream, 
    ClassBalancedBuffer, 
    NCMClassifier, 
    SCRTrainer
)

def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# 🌽 BASE_ID 계산 함수 추가
def compute_safe_base_id(*txt_files):
    """모든 txt 파일에서 최대 user ID를 찾아 안전한 BASE_ID 계산"""
    max_id = 0
    for path in txt_files:
        with open(path, 'r', encoding='utf-8') as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                try:
                    lab = int(parts[1])
                except ValueError:
                    continue
                max_id = max(max_id, lab)
    base = max_id + 1
    print(f"[NEG-BASE] max_user_id={max_id}, BASE_ID={base}")
    return base

# 🌽 purge_negatives 함수 추가
def purge_negatives(memory_buffer, base_id):
    """메모리 버퍼에서 모든 네거티브 클래스 제거"""
    to_del = [int(c) for c in list(memory_buffer.seen_classes) if int(c) >= base_id]
    removed = 0
    for c in to_del:
        if c in memory_buffer.buffer_groups:
            removed += len(memory_buffer.buffer_groups[c].buffer)
            del memory_buffer.buffer_groups[c]
        memory_buffer.seen_classes.discard(c)
    print(f"🧹 Purged {len(to_del)} negative classes ({removed} samples)")

# 🌽 remove_negative_samples_gradually 함수 수정 (버그 수정)
def remove_negative_samples_gradually(memory_buffer: ClassBalancedBuffer, 
                                    base_id: int,  # 🌽 base_id 파라미터 추가
                                    removal_ratio: float = 0.2):
    """
    메모리 버퍼에서 negative 샘플을 점진적으로 제거
    
    :param removal_ratio: 제거할 비율 (0.2 = 20%)
    """
    # negative_classes = [c for c in memory_buffer.seen_classes if c < 0]  # 🪵 버그: 네거티브는 음수가 아님
    negative_classes = [int(c) for c in memory_buffer.seen_classes if int(c) >= base_id]  # 🌽 수정
    
    if not negative_classes:
        return 0
    
    # 제거할 클래스 수 계산
    num_to_remove = max(1, int(len(negative_classes) * removal_ratio))
    
    # 랜덤하게 선택하여 제거
    classes_to_remove = np.random.choice(negative_classes, size=num_to_remove, replace=False)
    
    removed_count = 0
    for class_id in classes_to_remove:
        if class_id in memory_buffer.buffer_groups:
            # 해당 클래스의 샘플 수
            removed_count += len(memory_buffer.buffer_groups[class_id].buffer)
            # 버퍼에서 제거
            del memory_buffer.buffer_groups[class_id]
            memory_buffer.seen_classes.remove(class_id)
    
    print(f"Removed {num_to_remove} negative classes ({removed_count} samples)")
    return removed_count

# 🎨 t-SNE 시각화 함수 추가
@torch.no_grad()
def plot_tsne_from_memory(trainer,
                          save_path,
                          per_class=150,
                          space='fe',     # 'fe' = 6144D, 'z' = 128D
                          perplexity=30,
                          max_points=5000,
                          seed=42):
    """
    메모리 버퍼의 임베딩을 t-SNE로 시각화
    """
    model = trainer.model
    model.eval()
    device = trainer.device
    
    # 1) 메모리 버퍼에서 데이터 가져오기
    paths, labels = trainer.memory_buffer.get_all_data()
    if len(paths) == 0:
        print("[t-SNE] Memory buffer is empty.")
        return
    
    # 가짜 클래스 제외 (네거티브 샘플)
    base_id = getattr(trainer, 'base_id', 
                     getattr(trainer.config.negative, 'base_id', 10000))
    real_data = [(p, l) for p, l in zip(paths, labels) 
                 if int(l) < base_id]
    
    if len(real_data) == 0:
        print("[t-SNE] No real users in buffer.")
        return
    
    # 클래스별 균형 샘플링
    from collections import defaultdict
    by_cls = defaultdict(list)
    for p, y in real_data:
        by_cls[int(y)].append(p)
    
    # 균등 샘플링
    rng = random.Random(seed)
    sampled = []
    for c, lst in by_cls.items():
        k = min(per_class, len(lst))
        sampled += [(p, c) for p in rng.sample(lst, k)]
    
    # 전체 상한선
    if len(sampled) > max_points:
        sampled = rng.sample(sampled, max_points)
    
    print(f"[t-SNE] Processing {len(sampled)} samples from {len(by_cls)} classes")
    
    # 2) Transform 정의
    transform = trainer.test_transform
    ch = trainer.config.dataset.channels
    
    # 3) 배치로 임베딩 추출
    batch_size = 128
    features_list = []
    labels_list = []
    
    for i in range(0, len(sampled), batch_size):
        batch = sampled[i:i+batch_size]
        
        # 이미지 로드
        imgs = []
        for path, _ in batch:
            with Image.open(path) as img:
                img = img.convert('L' if ch == 1 else 'RGB')
                imgs.append(transform(img))
        
        imgs = torch.stack(imgs).to(device)
        
        # Feature extraction
        features = model(imgs)  # eval 모드에서는 6144D L2-normalized
        
        if space == 'z' and hasattr(model, 'projection_head'):
            # 128D projection 공간
            features = F.normalize(model.projection_head(features), dim=-1)
        
        features_list.append(features.cpu())
        labels_list.extend([y for _, y in batch])
    
    # 합치기
    X = torch.cat(features_list, dim=0).numpy()
    Y = np.array(labels_list)
    
    # 4) t-SNE 실행
    perplexity = min(perplexity, max(5, len(Y)//4))
    
    print(f"[t-SNE] Running t-SNE with perplexity={perplexity}...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=1500,
        metric='euclidean',  # L2-정규화 후 euclidean = 코사인과 단조 동등
        init='pca',
        learning_rate='auto',
        random_state=seed,
        verbose=0  # 간결한 출력
    )
    
    Z = tsne.fit_transform(X)
    
    # 5) 플롯 그리기
    plt.figure(figsize=(12, 8))
    
    # 색상맵 사용
    n_classes = len(set(Y))
    cmap = cm.get_cmap('tab20' if n_classes <= 20 else 'hsv', n_classes)
    
    # 클래스별로 그리기
    for idx, c in enumerate(sorted(set(Y))):
        mask = (Y == c)
        color = cmap(idx)
        plt.scatter(
            Z[mask, 0], Z[mask, 1],
            c=[color],
            s=15,
            alpha=0.7,
            edgecolors='none',
            label=f'User {c}' if n_classes <= 10 else None
        )
    
    # 제목 및 스타일
    plt.title(f't-SNE Visualization ({space.upper()}) | {n_classes} classes, {len(Y)} points', 
              fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # 범례 (클래스가 적을 때만)
    if n_classes <= 10:
        plt.legend(loc='best', markerscale=2)
    
    plt.tight_layout()
    
    # 저장
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[t-SNE] Saved to {save_path}")
# 🎨

class ContinualLearningEvaluator:
    """
    Continual Learning 평가 메트릭 계산
    - Average Accuracy
    - Forgetting Measure
    🐋 - Open-set metrics (TAR, TRR, FAR)
    """
    def __init__(self, num_experiences: int):
        self.num_experiences = num_experiences
        self.accuracy_history = defaultdict(list)  # {exp_id: [acc1, acc2, ...]}
        self.openset_history = []  # 🐋 오픈셋 메트릭 히스토리
        
    def update(self, experience_id: int, accuracy: float):
        """experience_id 학습 후 정확도 업데이트"""
        self.accuracy_history[experience_id].append(accuracy)
    
    def update_openset(self, metrics: dict):  # 🐋 새 메서드
        """오픈셋 메트릭 업데이트"""
        self.openset_history.append(metrics)
    
    def get_average_accuracy(self) -> float:
        """현재까지의 평균 정확도"""
        all_accs = []
        for accs in self.accuracy_history.values():
            if accs:
                all_accs.append(accs[-1])  # 각 experience의 최신 정확도
        return np.mean(all_accs) if all_accs else 0.0
    
    def get_forgetting_measure(self) -> float:
        """
        Forgetting Measure 계산
        각 experience에 대해: max(이전 정확도) - 현재 정확도
        """
        forgetting = []
        for exp_id, accs in self.accuracy_history.items():
            if len(accs) > 1:
                max_acc = max(accs[:-1])
                curr_acc = accs[-1]
                forgetting.append(max_acc - curr_acc)
        
        return np.mean(forgetting) if forgetting else 0.0
    
    def get_latest_openset_metrics(self) -> dict:  # 🐋 새 메서드
        """최신 오픈셋 메트릭 반환"""
        if self.openset_history:
            return self.openset_history[-1]
        return {}


def evaluate_on_test_set(trainer: SCRTrainer, config, openset_mode=False) -> tuple:  # 🐋 수정
    """
    test_set_file을 사용한 전체 평가
    🐋 openset_mode=True면 오픈셋 평가도 수행
    
    Returns:
        (accuracy, openset_metrics) if openset_mode else (accuracy, None)
    """
    # 전체 테스트셋 로드
    test_dataset = MyDataset(
        txt=config.dataset.test_set_file,
        transforms=get_scr_transforms(
            train=False,
            imside=config.dataset.height,
            channels=config.dataset.channels
        ),
        train=False,
        imside=config.dataset.height,
        outchannels=config.dataset.channels
    )
    
    # 현재까지 학습한 클래스만 필터링
    known_classes = set(trainer.ncm.class_means_dict.keys())
    
    if not known_classes:
        return (0.0, {}) if openset_mode else (0.0, None)  # 🐋
    
    # 🐋 오픈셋 모드: Known과 Unknown 분리
    if openset_mode and trainer.openset_enabled:
        known_indices = []
        unknown_indices = []
        
        for i in range(len(test_dataset)):
            label = int(test_dataset.images_label[i])
            if label in known_classes:
                known_indices.append(i)
            else:
                unknown_indices.append(i)
        
        # Known 평가 (Closed-set accuracy)
        if known_indices:
            known_subset = Subset(test_dataset, known_indices)
            accuracy = trainer.evaluate(known_subset)
        else:
            accuracy = 0.0
        
        # 🐋 오픈셋 메트릭 계산
        openset_metrics = {}
        
        if known_indices and trainer.ncm.tau_s is not None:
            # Known에서 TAR/FRR 계산
            from utils.utils_openset import predict_batch
            
            # 샘플링 (너무 많으면)
            if len(known_indices) > 500:
                known_indices = np.random.choice(known_indices, 500, replace=False)
            
            known_paths = [test_dataset.images_path[i] for i in known_indices] 
            known_labels = [int(test_dataset.images_label[i]) for i in known_indices]
            
            preds = predict_batch(
                trainer.model, trainer.ncm,
                known_paths, trainer.test_transform, trainer.device
            )
            
            correct = sum(1 for p, l in zip(preds, known_labels) if p == l)
            rejected = sum(1 for p in preds if p == -1)
            
            openset_metrics['TAR'] = correct / max(1, len(preds))
            openset_metrics['FRR'] = rejected / max(1, len(preds))
        
        if unknown_indices and trainer.ncm.tau_s is not None:
            # Unknown에서 TRR/FAR 계산
            from utils.utils_openset import predict_batch
            
            # 샘플링
            if len(unknown_indices) > 500:
                unknown_indices = np.random.choice(unknown_indices, 500, replace=False)
            
            unknown_paths = [test_dataset.images_path[i] for i in unknown_indices]  # 🐋
            
            preds = predict_batch(
                trainer.model, trainer.ncm,
                unknown_paths, trainer.test_transform, trainer.device
            )
            
            openset_metrics['TRR_unknown'] = sum(1 for p in preds if p == -1) / len(preds)
            openset_metrics['FAR_unknown'] = 1 - openset_metrics['TRR_unknown']
        
        openset_metrics['tau_s'] = trainer.ncm.tau_s
        openset_metrics['tau_m'] = trainer.ncm.tau_m if trainer.ncm.use_margin else None
        
        print(f"Evaluating on {len(known_indices)} known + {len(unknown_indices)} unknown samples")
        
        return accuracy, openset_metrics
    
    else:
        # 🐋 기존 방식 (Closed-set만)
        # 필터링된 인덱스 찾기
        filtered_indices = []
        for i in range(len(test_dataset)):
            label = int(test_dataset.images_label[i])
            if label in known_classes:
                filtered_indices.append(i)
        
        if not filtered_indices:
            return (0.0, None)  # 🐋
        
        # Subset 생성
        filtered_test = Subset(test_dataset, filtered_indices)
        
        print(f"Evaluating on {len(filtered_indices)} test samples from {len(known_classes)} classes")
        
        # 평가
        accuracy = trainer.evaluate(filtered_test)  # 🐋
        return accuracy, None  # 🐋


def main(args):
    """메인 실행 함수"""
    
    # 1. Configuration 로드
    config = ConfigParser(args.config)
    # 👻 config 객체 가져오기
    config_obj = config.get_config()  # 👻
    print(f"Using config: {args.config}")
    print(config)
    
    # 🌽 BASE_ID 계산 및 설정
    config_obj.negative.base_id = compute_safe_base_id(
        config_obj.dataset.train_set_file,
        config_obj.dataset.test_set_file
    )
    
    # 🐋 오픈셋 모드 확인
    openset_enabled = hasattr(config_obj, 'openset') and config_obj.openset.enabled
    if openset_enabled:
        print("\n🐋 ========== OPEN-SET MODE ENABLED ==========")
        print(f"   Warmup users: {config_obj.openset.warmup_users}")
        print(f"   Initial tau: {config_obj.openset.initial_tau}")
        print(f"   Margin: {config_obj.openset.use_margin} (tau={config_obj.openset.margin_tau})")
        print("🐋 =========================================\n")
    
    # GPU 설정
    device = torch.device(
        f"cuda:{config_obj.training.gpu_ids}" 
        if torch.cuda.is_available() and not args.no_cuda 
        else "cpu"
    )
    print(f'Device: {device}')
    
    # Random seed 고정
    if args.seed is not None:
        fix_random_seed(args.seed)
    
    # 2. 결과 저장 디렉토리 생성
    results_dir = os.path.join(config_obj.training.results_path, 'scr_results')
    if openset_enabled:  # 🐋
        results_dir = os.path.join(config_obj.training.results_path, 'scr_openset_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 3. 데이터 스트림 초기화
    print("\n=== Initializing Data Stream ===")
    data_stream = ExperienceStream(
        train_file=config_obj.dataset.train_set_file,
        negative_file=config_obj.dataset.negative_samples_file,
        num_negative_classes=config_obj.dataset.num_negative_classes,
        base_id=config_obj.negative.base_id  # 🌽 base_id 전달
    )
    
    stats = data_stream.get_statistics()
    print(f"Total users: {stats['num_users']}")
    print(f"Samples per user: {stats['samples_per_user']}")
    print(f"Negative samples: {stats['negative_samples']}")
    
    # 4. 모델 및 컴포넌트 초기화
    print("\n=== Initializing Model and Components ===")
    
    # CCNet 모델
    # 🦴 model = ccnet(weight=config_obj.model.competition_weight)
    
    # 🧀 프로젝션 헤드 설정 포함하여 모델 생성
    model = ccnet(
        weight=config_obj.model.competition_weight,
        use_projection=config_obj.model.use_projection,
        projection_dim=config_obj.model.projection_dim
    )

    # 🧀 프로젝션 헤드 설정 출력
    if config_obj.model.use_projection:
        print(f"🧀 Projection Head Configuration:")
        print(f"   Enabled: True")
        print(f"   Dimension: 6144 -> 2048 -> {config_obj.model.projection_dim}")
        print(f"   Structure: 2 layers with LayerNorm")
        print(f"   Training: Uses projection ({config_obj.model.projection_dim}D)")
        print(f"   NCM/Eval: Uses original features (6144D)")
    else:
        print(f"📌 Projection Head: Disabled (using raw 6144D features)")
    
    # 👻 사전훈련 가중치 로드 (device 이동 전에!)
    if hasattr(config_obj.model, 'use_pretrained') and config_obj.model.use_pretrained:  # 👻
        if config_obj.model.pretrained_path and config_obj.model.pretrained_path.exists():  # 👻
            print(f"\n📦 Loading pretrained weights from main script...")  # 👻
            loader = PretrainedLoader()  # 👻
            try:  # 👻
                model = loader.load_ccnet_pretrained(  # 👻
                    model=model,  # 👻
                    checkpoint_path=config_obj.model.pretrained_path,  # 👻
                    device=device,  # 👻
                    verbose=True  # 👻
                )  # 👻
                print("✅ Pretrained weights loaded successfully!")  # 👻
            except Exception as e:  # 👻
                print(f"⚠️  Failed to load pretrained model: {e}")  # 👻
                print("Continuing with random initialization...")  # 👻
        else:  # 👻
            print(f"⚠️  Pretrained path not found or not set")  # 👻
    else:  # 👻
        print("🎲 Starting from random initialization")  # 👻
    
    # 👻 모델을 device로 이동
    model = model.to(device)  # 👻
    
    # NCM Classifier
    ncm_classifier = NCMClassifier(normalize=True).to(device)  # 🐋 코사인 모드로 변경
    
    # Memory Buffer
    memory_buffer = ClassBalancedBuffer(
        max_size=config_obj.training.memory_size,
        min_samples_per_class=config_obj.training.min_samples_per_class
    )
    
    # SCR Trainer
    # 👻 config_obj 전달
    trainer = SCRTrainer(
        model=model,
        ncm_classifier=ncm_classifier,
        memory_buffer=memory_buffer,
        config=config_obj,  # 👻 config → config_obj
        device=device
    )
    
    # 5. Negative 샘플로 초기화 🐣
    print("\n=== Initializing with Negative Samples ===")
    neg_paths, neg_labels = data_stream.get_negative_samples()
    
    # memory_batch_size만큼만 선택
    if len(neg_paths) > config_obj.training.memory_batch_size:
        selected_indices = np.random.choice(
            len(neg_paths), 
            size=config_obj.training.memory_batch_size,
            replace=False
        )
        neg_paths = [neg_paths[i] for i in selected_indices]
        neg_labels = [neg_labels[i] for i in selected_indices]
    
    # 메모리 버퍼 초기화
    memory_buffer.update_from_dataset(neg_paths, neg_labels)
    print(f"Initial buffer size: {len(memory_buffer)}")
    
    # NCM 초기화 🐣
    print("🍄 NCM starts empty - no fake class contamination")  # 🍄
    
    # 6. 평가자 초기화
    evaluator = ContinualLearningEvaluator(num_experiences=config_obj.training.num_experiences)
    
    # 7. 학습 결과 저장용
    training_history = {
        'losses': [],
        'accuracies': [],
        'forgetting_measures': [],
        'memory_sizes': [],
        'negative_removal_history': [],
        'openset_metrics': [],  # 🐋 추가
        # 👻 사전훈련 정보 추가
        'pretrained_used': config_obj.model.use_pretrained,  # 👻
        'pretrained_path': str(config_obj.model.pretrained_path) if config_obj.model.pretrained_path else None,  # 👻
        'openset_enabled': openset_enabled  # 🐋 추가
    }
    
    # 8. Continual Learning 시작
    print("\n=== Starting Continual Learning ===")
    print(f"Total experiences: {config_obj.training.num_experiences}")
    print(f"Evaluation interval: every {config_obj.training.test_interval} users")
    print(f"🔥 Negative warmup: exp0~{config_obj.negative.warmup_experiences-1}")  # 🌽
    # 👻 중요 파라미터 출력
    print(f"Learning rate: {config_obj.training.learning_rate}")  # 👻
    print(f"Memory batch size: {config_obj.training.memory_batch_size}")  # 👻
    print(f"Temperature: {config_obj.training.temperature}")  # 👻
    
    start_time = time.time()
    
    for exp_id, (user_id, image_paths, labels) in enumerate(data_stream):
        
        # Experience 학습
        stats = trainer.train_experience(user_id, image_paths, labels)
        training_history['losses'].append(stats['loss'])
        training_history['memory_sizes'].append(stats['memory_size'])
        
        # 🌽 exp3→exp4 경계에서 네거티브 완전 제거
        if exp_id + 1 == config_obj.negative.warmup_experiences:
            print(f"\n🔥 === Warmup End (exp{exp_id}) → Post-warmup (exp{exp_id+1}) ===")
            
            # 평가 (purge 전)
            acc_pre, _ = evaluate_on_test_set(trainer, config_obj, openset_mode=openset_enabled)
            print(f"[Warmup-End] pre-purge ACC={acc_pre:.2f}%")
            
            # 네거티브 제거
            purge_negatives(memory_buffer, config_obj.negative.base_id)
            trainer._update_ncm()
            
            # 평가 (purge 후)
            acc_post, _ = evaluate_on_test_set(trainer, config_obj, openset_mode=openset_enabled)
            print(f"[Warmup-End] post-purge ACC={acc_post:.2f}%")
            print(f"🔥 ========================================\n")
        
        # 평가 주기 확인
        if (exp_id + 1) % config_obj.training.test_interval == 0 or exp_id == config_obj.training.num_experiences - 1:
            
            print(f"\n=== Evaluation at Experience {exp_id + 1} ===")
            
            # 🎨 t-SNE 시각화 추가
            if (exp_id + 1) % 10 == 0 or exp_id == config_obj.training.num_experiences - 1:
                tsne_dir = os.path.join(results_dir, "tsne")
                os.makedirs(tsne_dir, exist_ok=True)
                
                # 6144D feature space 시각화
                tsne_path = os.path.join(tsne_dir, f"tsne_exp_{exp_id+1:03d}.png")
                plot_tsne_from_memory(
                    trainer=trainer,
                    save_path=tsne_path,
                    per_class=150,
                    space='fe',  # NCM 공간
                    perplexity=30,
                    max_points=5000,
                    seed=42
                )
                
                # Projection space도 시각화 (옵션)
                if config_obj.model.use_projection:
                    tsne_path_z = os.path.join(tsne_dir, f"tsne_z_exp_{exp_id+1:03d}.png")
                    plot_tsne_from_memory(
                        trainer=trainer,
                        save_path=tsne_path_z,
                        per_class=150,
                        space='z',  # 128D projection
                        perplexity=30,
                        max_points=5000,
                        seed=42
                    )
            # 🎨
            
            # 테스트셋으로 평가
            accuracy, openset_metrics = evaluate_on_test_set(  # 🐋
                trainer, config_obj, 
                openset_mode=openset_enabled
            )
            
            # 메트릭 업데이트
            evaluator.update(exp_id, accuracy)
            if openset_metrics:  # 🐋
                evaluator.update_openset(openset_metrics)
            
            # 평균 정확도와 Forgetting 계산
            avg_acc = evaluator.get_average_accuracy()
            forgetting = evaluator.get_forgetting_measure()
            
            # 🐋 기록 저장
            accuracy_record = {
                'experience': exp_id + 1,
                'accuracy': accuracy,
                'average_accuracy': avg_acc,
                'forgetting': forgetting
            }
            
            # 🐋 오픈셋 메트릭 추가
            if openset_metrics:
                accuracy_record.update(openset_metrics)
                training_history['openset_metrics'].append(openset_metrics)
            
            training_history['accuracies'].append(accuracy_record)
            
            print(f"Test Accuracy: {accuracy:.2f}%")
            print(f"Average Accuracy: {avg_acc:.2f}%")
            print(f"Forgetting Measure: {forgetting:.2f}%")
            print(f"Memory Buffer Size: {len(memory_buffer)}")
            
            # 🐋 오픈셋 메트릭 출력
            if openset_metrics:
                print(f"\n🐋 Open-set Metrics:")
                if 'TAR' in openset_metrics:
                    print(f"   TAR: {openset_metrics['TAR']:.3f}, FRR: {openset_metrics['FRR']:.3f}")
                if 'TRR_unknown' in openset_metrics:
                    print(f"   TRR_unknown: {openset_metrics['TRR_unknown']:.3f}, FAR_unknown: {openset_metrics['FAR_unknown']:.3f}")
                print(f"   τ_s: {openset_metrics.get('tau_s', 0):.4f}")
            
            # 🐋 Trainer의 오픈셋 평가 히스토리도 저장
            if hasattr(trainer, 'evaluation_history') and trainer.evaluation_history:
                training_history['trainer_openset_history'] = trainer.evaluation_history
            
            # 체크포인트 저장
            checkpoint_path = os.path.join(
                results_dir, 
                f'checkpoint_exp_{exp_id + 1}.pth'
            )
            trainer.save_checkpoint(checkpoint_path)
            
        # 진행 상황 출력
        if (exp_id + 1) % 10 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (exp_id + 1) * (config_obj.training.num_experiences - exp_id - 1)
            print(f"Progress: {exp_id + 1}/{config_obj.training.num_experiences} "
                  f"({100 * (exp_id + 1) / config_obj.training.num_experiences:.1f}%) "
                  f"| Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")
    
    # 9. 최종 결과 저장
    print("\n=== Saving Final Results ===")
    
    # 학습 기록 저장
    history_path = os.path.join(results_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=4)
    
    # 🎨 t-SNE 시각화 경로 추가
    tsne_images = []
    tsne_dir = os.path.join(results_dir, "tsne")
    if os.path.exists(tsne_dir):
        tsne_images = sorted([f for f in os.listdir(tsne_dir) if f.endswith('.png')])
        if tsne_images:
            print(f"\n🎨 t-SNE visualizations saved: {len(tsne_images)} images")
            print(f"   Location: {tsne_dir}")
    # 🎨
    
    # 최종 통계
    final_result = training_history['accuracies'][-1] if training_history['accuracies'] else {}
    final_acc = final_result.get('accuracy', 0)
    final_avg_acc = final_result.get('average_accuracy', 0)
    final_forget = final_result.get('forgetting', 0)
    
    print(f"\n=== Final Results ===")
    print(f"Final Test Accuracy: {final_acc:.2f}%")
    print(f"Final Average Accuracy: {final_avg_acc:.2f}%")
    print(f"Final Forgetting Measure: {final_forget:.2f}%")
    
    # 🐋 최종 오픈셋 메트릭
    if openset_enabled and 'TAR' in final_result:
        print(f"\n🐋 Final Open-set Performance:")
        print(f"   TAR: {final_result.get('TAR', 0):.3f}")
        print(f"   TRR (Unknown): {final_result.get('TRR_unknown', 0):.3f}")
        print(f"   FAR (Unknown): {final_result.get('FAR_unknown', 0):.3f}")
        print(f"   Final τ_s: {final_result.get('tau_s', 0):.4f}")
    
    print(f"\nTotal Training Time: {(time.time() - start_time)/60:.1f} minutes")
    
    # 결과 요약 저장
    summary = {
        'config': args.config,
        'num_experiences': config_obj.training.num_experiences,
        'memory_size': config_obj.training.memory_size,
        'final_accuracy': final_acc,
        'final_average_accuracy': final_avg_acc,
        'final_forgetting': final_forget,
        'total_time_minutes': (time.time() - start_time) / 60,
        'negative_removal_history': training_history['negative_removal_history'],
        'openset_enabled': openset_enabled  # 🐋
    }
    
    # 🐋 오픈셋 요약 추가
    if openset_enabled and final_result:
        summary['final_openset'] = {
            'TAR': final_result.get('TAR', 0),
            'TRR_unknown': final_result.get('TRR_unknown', 0),
            'FAR_unknown': final_result.get('FAR_unknown', 0),
            'tau_s': final_result.get('tau_s', 0)
        }
    
    summary_path = os.path.join(results_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SCR Training for CCNet with Open-set Support')  # 🐋 설명 수정
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA')
    
    args = parser.parse_args()
    main(args)