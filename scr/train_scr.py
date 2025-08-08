#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Supervised Contrastive Replay (SCR) Training Script
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

class ContinualLearningEvaluator:
    """
    Continual Learning 평가 메트릭 계산
    - Average Accuracy
    - Forgetting Measure
    """
    def __init__(self, num_experiences: int):
        self.num_experiences = num_experiences
        self.accuracy_history = defaultdict(list)  # {exp_id: [acc1, acc2, ...]}
        
    def update(self, experience_id: int, accuracy: float):
        """experience_id 학습 후 정확도 업데이트"""
        self.accuracy_history[experience_id].append(accuracy)
    
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


def evaluate_on_test_set(trainer: SCRTrainer, config) -> float:
    """
    test_Tongji.txt를 사용한 전체 평가
    현재까지 학습한 클래스만 평가
    """
    # 전체 테스트셋 로드
    test_dataset = MyDataset(
        txt=config.Dataset.test_set_file,
        transforms=get_scr_transforms(
            train=False,
            imside=config.Dataset.height,
            channels=config.Dataset.channels
        ),
        train=False,
        imside=config.Dataset.height,
        outchannels=config.Dataset.channels
    )
    
    # 현재까지 학습한 클래스만 필터링
    known_classes = set(trainer.ncm.class_means_dict.keys())
    
    if not known_classes:
        return 0.0
    
    # 필터링된 인덱스 찾기
    filtered_indices = []
    for i in range(len(test_dataset)):
        label = int(test_dataset.images_label[i])
        if label in known_classes:
            filtered_indices.append(i)
    
    if not filtered_indices:
        return 0.0
    
    # Subset 생성
    filtered_test = Subset(test_dataset, filtered_indices)
    
    print(f"Evaluating on {len(filtered_indices)} test samples from {len(known_classes)} classes")
    
    # 평가
    return trainer.evaluate(filtered_test)


def remove_negative_samples_gradually(memory_buffer: ClassBalancedBuffer, 
                                    removal_ratio: float = 0.2):
    """
    메모리 버퍼에서 negative 샘플을 점진적으로 제거
    
    :param removal_ratio: 제거할 비율 (0.2 = 20%)
    """
    # Negative 클래스 찾기 (음수 ID)
    negative_classes = [c for c in memory_buffer.seen_classes if c < 0]
    
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


def initialize_ncm_with_negatives(trainer: SCRTrainer, 
                                negative_paths: List[str], 
                                negative_labels: List[int]):
    """Negative 샘플로 NCM 초기화"""
    print("Initializing NCM with negative samples...")
    
    # 임시 데이터셋 생성
    from scr.scr_trainer import MemoryDataset
    neg_dataset = MemoryDataset(
        paths=negative_paths,
        labels=negative_labels,
        transform=trainer.test_transform,
        train=False
    )
    
    # DataLoader
    dataloader = DataLoader(
        neg_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0
    )
    
    # Feature 추출 및 class mean 계산
    trainer.model.eval()
    class_features = defaultdict(list)
    
    with torch.no_grad():
        for data, labels in dataloader:
            data = data[0].to(trainer.device)  # data1만 사용
            labels = labels.to(trainer.device)
            
            features = trainer.model.getFeatureCode(data)
            
            for i, label in enumerate(labels):
                label_item = label.item()
                class_features[label_item].append(features[i].cpu())
    
    # Class means 계산
    class_means = {}
    for label, features_list in class_features.items():
        if features_list:
            mean_feature = torch.stack(features_list).mean(dim=0)
            mean_feature = mean_feature / mean_feature.norm()
            class_means[label] = mean_feature
    
    # NCM 초기화
    trainer.ncm.replace_class_means_dict(class_means)
    print(f"NCM initialized with {len(class_means)} negative classes")
    
    trainer.model.train()


def main(args):
    """메인 실행 함수"""
    
    # 1. Configuration 로드
    config = ConfigParser(args.config)
    print(f"Using config: {args.config}")
    print(config)
    
    # GPU 설정
    device = torch.device(
        f"cuda:{config.Training.gpu_ids}" 
        if torch.cuda.is_available() and not args.no_cuda 
        else "cpu"
    )
    print(f'Device: {device}')
    
    # Random seed 고정
    if args.seed is not None:
        fix_random_seed(args.seed)
    
    # 2. 결과 저장 디렉토리 생성
    results_dir = os.path.join(config.Training.results_path, 'scr_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 3. 데이터 스트림 초기화
    print("\n=== Initializing Data Stream ===")
    data_stream = ExperienceStream(
        train_file=config.Dataset.train_set_file,
        negative_file=config.Dataset.negative_samples_file,
        num_negative_classes=-1  # 모든 negative 클래스 사용
    )
    
    stats = data_stream.get_statistics()
    print(f"Total users: {stats['num_users']}")
    print(f"Samples per user: {stats['samples_per_user']}")
    print(f"Negative samples: {stats['negative_samples']}")
    
    # 4. 모델 및 컴포넌트 초기화
    print("\n=== Initializing Model and Components ===")
    
    # CCNet 모델
    model = ccnet(weight=config.Model.competition_weight).to(device)
    
    # NCM Classifier
    ncm_classifier = NCMClassifier(normalize=False).to(device)
    
    # Memory Buffer
    memory_buffer = ClassBalancedBuffer(
        max_size=config.Training.memory_size,
        min_samples_per_class=config.Training.min_samples_per_class
    )
    
    # SCR Trainer
    trainer = SCRTrainer(
        model=model,
        ncm_classifier=ncm_classifier,
        memory_buffer=memory_buffer,
        config=config,
        device=device
    )
    
    # 5. Negative 샘플로 초기화 🐣
    print("\n=== Initializing with Negative Samples ===")
    neg_paths, neg_labels = data_stream.get_negative_samples()
    
    # memory_batch_size만큼만 선택
    if len(neg_paths) > config.Training.memory_batch_size:
        selected_indices = np.random.choice(
            len(neg_paths), 
            size=config.Training.memory_batch_size,
            replace=False
        )
        neg_paths = [neg_paths[i] for i in selected_indices]
        neg_labels = [neg_labels[i] for i in selected_indices]
    
    # 메모리 버퍼 초기화
    memory_buffer.update_from_dataset(neg_paths, neg_labels)
    print(f"Initial buffer size: {len(memory_buffer)}")
    
    # NCM 초기화 🐣
    initialize_ncm_with_negatives(trainer, neg_paths, neg_labels)
    
    # 6. 평가자 초기화
    evaluator = ContinualLearningEvaluator(num_experiences=config.Training.num_experiences)
    
    # 7. 학습 결과 저장용
    training_history = {
        'losses': [],
        'accuracies': [],
        'forgetting_measures': [],
        'memory_sizes': [],
        'negative_removal_history': []
    }
    
    # 8. Continual Learning 시작
    print("\n=== Starting Continual Learning ===")
    print(f"Total experiences: {config.Training.num_experiences}")
    print(f"Evaluation interval: every {config.Training.test_interval} users")
    
    start_time = time.time()
    
    for exp_id, (user_id, image_paths, labels) in enumerate(data_stream):
        
        # Experience 학습
        stats = trainer.train_experience(user_id, image_paths, labels)
        training_history['losses'].append(stats['loss'])
        training_history['memory_sizes'].append(stats['memory_size'])
        
        # Negative 샘플 점진적 제거 (50명마다 20%씩) 🐣
        if (exp_id + 1) % 50 == 0 and exp_id > 0:
            removed = remove_negative_samples_gradually(memory_buffer, removal_ratio=0.2)
            training_history['negative_removal_history'].append({
                'experience': exp_id + 1,
                'removed_samples': removed
            })
            
            # NCM 업데이트 (negative 클래스 제거 반영)
            trainer._update_ncm()
        
        # 평가 주기 확인
        if (exp_id + 1) % config.Training.test_interval == 0 or exp_id == config.Training.num_experiences - 1:
            
            print(f"\n=== Evaluation at Experience {exp_id + 1} ===")
            
            # 테스트셋으로 평가
            accuracy = evaluate_on_test_set(trainer, config)
            
            # 메트릭 업데이트
            evaluator.update(exp_id, accuracy)
            
            # 평균 정확도와 Forgetting 계산
            avg_acc = evaluator.get_average_accuracy()
            forgetting = evaluator.get_forgetting_measure()
            
            training_history['accuracies'].append({
                'experience': exp_id + 1,
                'accuracy': accuracy,
                'average_accuracy': avg_acc,
                'forgetting': forgetting
            })
            
            print(f"Test Accuracy: {accuracy:.2f}%")
            print(f"Average Accuracy: {avg_acc:.2f}%")
            print(f"Forgetting Measure: {forgetting:.2f}%")
            print(f"Memory Buffer Size: {len(memory_buffer)}")
            
            # 체크포인트 저장
            checkpoint_path = os.path.join(
                results_dir, 
                f'checkpoint_exp_{exp_id + 1}.pth'
            )
            trainer.save_checkpoint(checkpoint_path)
            
        # 진행 상황 출력
        if (exp_id + 1) % 10 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (exp_id + 1) * (config.Training.num_experiences - exp_id - 1)
            print(f"Progress: {exp_id + 1}/{config.Training.num_experiences} "
                  f"({100 * (exp_id + 1) / config.Training.num_experiences:.1f}%) "
                  f"| Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")
    
    # 9. 최종 결과 저장
    print("\n=== Saving Final Results ===")
    
    # 학습 기록 저장
    history_path = os.path.join(results_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=4)
    
    # 최종 통계
    final_result = training_history['accuracies'][-1] if training_history['accuracies'] else {}
    final_acc = final_result.get('accuracy', 0)
    final_avg_acc = final_result.get('average_accuracy', 0)
    final_forget = final_result.get('forgetting', 0)
    
    print(f"\n=== Final Results ===")
    print(f"Final Test Accuracy: {final_acc:.2f}%")
    print(f"Final Average Accuracy: {final_avg_acc:.2f}%")
    print(f"Final Forgetting Measure: {final_forget:.2f}%")
    print(f"Total Training Time: {(time.time() - start_time)/60:.1f} minutes")
    
    # 결과 요약 저장
    summary = {
        'config': args.config,
        'num_experiences': config.Training.num_experiences,
        'memory_size': config.Training.memory_size,
        'final_accuracy': final_acc,
        'final_average_accuracy': final_avg_acc,
        'final_forgetting': final_forget,
        'total_time_minutes': (time.time() - start_time) / 60,
        'negative_removal_history': training_history['negative_removal_history']
    }
    
    summary_path = os.path.join(results_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SCR Training for CCNet')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA')
    
    args = parser.parse_args()
    main(args)