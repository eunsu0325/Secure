#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Supervised Contrastive Replay (SCR) Training Script with Open-set Support
CCNet + SCR for Continual Learning
ProxyAnchorLoss and TTA Support
"""

import os
import argparse
import time
import numpy as np
from collections import defaultdict
import json
import random

import torch
from torch.utils.data import Subset

# Project imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ConfigParser
from utils.pretrained_loader import PretrainedLoader

# 시각화 관련 (t-SNE 디버깅용)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from PIL import Image
import torch.nn.functional as F

from models import ccnet, MyDataset, get_scr_transforms
from scr import (
    ExperienceStream,
    ClassBalancedBuffer,
    NCMClassifier,
    SCRTrainer
)

from utils.utils_openset import (
    predict_batch,
    predict_batch_tta,
    load_paths_labels_from_txt
)

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

def purge_negatives(memory_buffer, base_id):
    """메모리 버퍼에서 모든 네거티브 클래스 제거"""
    to_del = [int(c) for c in list(memory_buffer.seen_classes) if int(c) >= base_id]
    removed = 0
    for c in to_del:
        if c in memory_buffer.buffer_groups:
            removed += len(memory_buffer.buffer_groups[c].buffer)
            del memory_buffer.buffer_groups[c]
        memory_buffer.seen_classes.discard(c)
    print(f"Purged {len(to_del)} negative classes ({removed} samples)")

# [제거됨] remove_negative_samples_gradually 함수는 사용하지 않음
# 대신 purge_negatives로 워밍업 종료 시 한번에 제거

@torch.no_grad()
def plot_tsne_from_memory(trainer,
                          save_path,
                          per_class=150,
                          space='fe',
                          perplexity=30,
                          max_points=5000,
                          seed=42):
    """메모리 버퍼의 임베딩을 t-SNE로 시각화 (디버깅용)"""
    model = trainer.model
    model.eval()
    device = trainer.device

    paths, labels = trainer.memory_buffer.get_all_data()
    if len(paths) == 0:
        print("[t-SNE] Memory buffer is empty.")
        return

    base_id = getattr(trainer, 'base_id',
                     getattr(trainer.config.negative, 'base_id', 10000))
    real_data = [(p, l) for p, l in zip(paths, labels)
                 if int(l) < base_id]

    if len(real_data) == 0:
        print("[t-SNE] No real users in buffer.")
        return

    from collections import defaultdict
    by_cls = defaultdict(list)
    for p, y in real_data:
        by_cls[int(y)].append(p)

    rng = random.Random(seed)
    sampled = []
    for c, lst in by_cls.items():
        k = min(per_class, len(lst))
        sampled += [(p, c) for p in rng.sample(lst, k)]

    if len(sampled) > max_points:
        sampled = rng.sample(sampled, max_points)

    print(f"[t-SNE] Processing {len(sampled)} samples from {len(by_cls)} classes")

    transform = trainer.test_transform
    ch = trainer.config.dataset.channels

    batch_size = 128
    features_list = []
    labels_list = []

    for i in range(0, len(sampled), batch_size):
        batch = sampled[i:i+batch_size]

        imgs = []
        for path, _ in batch:
            with Image.open(path) as img:
                img = img.convert('L' if ch == 1 else 'RGB')
                imgs.append(transform(img))

        imgs = torch.stack(imgs).to(device)
        features = model(imgs)

        if space == 'z' and hasattr(model, 'projection_head'):
            features = F.normalize(model.projection_head(features), dim=-1)

        features_list.append(features.cpu())
        labels_list.extend([y for _, y in batch])

    X = torch.cat(features_list, dim=0).numpy()
    Y = np.array(labels_list)

    perplexity = min(perplexity, max(5, len(Y)//4))

    print(f"[t-SNE] Running t-SNE with perplexity={perplexity}...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=1500,
        metric='euclidean',
        init='pca',
        learning_rate='auto',
        random_state=seed,
        verbose=0
    )

    Z = tsne.fit_transform(X)

    plt.figure(figsize=(12, 8))

    n_classes = len(set(Y))
    cmap = cm.get_cmap('tab20' if n_classes <= 20 else 'hsv', n_classes)

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

    plt.title(f't-SNE Visualization ({space.upper()}) | {n_classes} classes, {len(Y)} points',
              fontsize=14, fontweight='bold')
    plt.axis('off')

    if n_classes <= 10:
        plt.legend(loc='best', markerscale=2)

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[t-SNE] Saved to {save_path}")

class ContinualLearningEvaluator:
    """
    Continual Learning 평가 메트릭 계산
    - Average Accuracy
    - Forgetting Measure
    - Open-set metrics (TAR, TRR, FAR)
    """
    def __init__(self, num_experiences: int):
        self.num_experiences = num_experiences
        self.accuracy_history = defaultdict(list)
        self.openset_history = []

    def update(self, experience_id: int, accuracy: float):
        """experience_id 학습 후 정확도 업데이트"""
        self.accuracy_history[experience_id].append(accuracy)

    def update_openset(self, metrics: dict):
        """오픈셋 메트릭 업데이트"""
        self.openset_history.append(metrics)

    def get_average_accuracy(self) -> float:
        """현재까지의 평균 정확도"""
        all_accs = []
        for accs in self.accuracy_history.values():
            if accs:
                all_accs.append(accs[-1])
        return np.mean(all_accs) if all_accs else 0.0

    def get_forgetting_measure(self) -> float:
        """Forgetting Measure 계산"""
        forgetting = []
        for exp_id, accs in self.accuracy_history.items():
            if len(accs) > 1:
                max_acc = max(accs[:-1])
                curr_acc = accs[-1]
                forgetting.append(max_acc - curr_acc)

        return np.mean(forgetting) if forgetting else 0.0

    def get_latest_openset_metrics(self) -> dict:
        """최신 오픈셋 메트릭 반환"""
        if self.openset_history:
            return self.openset_history[-1]
        return {}


def evaluate_on_test_set(trainer: SCRTrainer, config, openset_mode=False) -> tuple:
    """
    test_set_file을 사용한 전체 평가
    openset_mode=True면 오픈셋 평가도 수행
    TTA 지원 포함

    Returns:
        (accuracy, openset_metrics) if openset_mode else (accuracy, None)
    """
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

    known_classes = set(trainer.ncm.class_means_dict.keys())

    if not known_classes:
        return (0.0, {}) if openset_mode else (0.0, None)

    # TTA 설정 확인
    use_tta = (hasattr(trainer, 'use_tta') and trainer.use_tta and
               hasattr(config, 'openset') and config.openset.tta_n_views > 1)

    if openset_mode and trainer.openset_enabled:
        known_indices = []
        unknown_indices = []

        for i in range(len(test_dataset)):
            label = int(test_dataset.images_label[i])
            if label in known_classes:
                known_indices.append(i)
            else:
                unknown_indices.append(i)

        if known_indices:
            known_subset = Subset(test_dataset, known_indices)
            accuracy = trainer.evaluate(known_subset)
        else:
            accuracy = 0.0

        openset_metrics = {}

        if known_indices and trainer.ncm.tau_s is not None:
            if len(known_indices) > 500:
                known_indices = np.random.choice(known_indices, 500, replace=False)

            known_paths = [test_dataset.images_path[i] for i in known_indices]
            known_labels = [int(test_dataset.images_label[i]) for i in known_indices]

            # TTA 또는 일반 예측
            if use_tta:
                preds = predict_batch_tta(
                    trainer.model, trainer.ncm,
                    known_paths, trainer.test_transform, trainer.device,
                    n_views=config.openset.tta_n_views,
                    include_original=config.openset.tta_include_original,
                    agree_k=config.openset.tta_agree_k,
                    aug_strength=config.openset.tta_augmentation_strength,
                    img_size=config.dataset.height,
                    channels=config.dataset.channels
                )
            else:
                preds = predict_batch(
                    trainer.model, trainer.ncm,
                    known_paths, trainer.test_transform, trainer.device,
                    channels=config.dataset.channels
                )

            correct = sum(1 for p, l in zip(preds, known_labels) if p == l)
            rejected = sum(1 for p in preds if p == -1)

            openset_metrics['TAR'] = correct / max(1, len(preds))
            openset_metrics['FRR'] = rejected / max(1, len(preds))

        if unknown_indices and trainer.ncm.tau_s is not None:
            if len(unknown_indices) > 500:
                unknown_indices = np.random.choice(unknown_indices, 500, replace=False)

            unknown_paths = [test_dataset.images_path[i] for i in unknown_indices]

            # TTA 또는 일반 예측
            if use_tta:
                preds = predict_batch_tta(
                    trainer.model, trainer.ncm,
                    unknown_paths, trainer.test_transform, trainer.device,
                    n_views=config.openset.tta_n_views,
                    include_original=config.openset.tta_include_original,
                    agree_k=config.openset.tta_agree_k,
                    aug_strength=config.openset.tta_augmentation_strength,
                    img_size=config.dataset.height,
                    channels=config.dataset.channels
                )
            else:
                preds = predict_batch(
                    trainer.model, trainer.ncm,
                    unknown_paths, trainer.test_transform, trainer.device,
                    channels=config.dataset.channels
                )

            openset_metrics['TRR_unknown'] = sum(1 for p in preds if p == -1) / len(preds)
            openset_metrics['FAR_unknown'] = 1 - openset_metrics['TRR_unknown']

        openset_metrics['tau_s'] = trainer.ncm.tau_s

        # TTA 정보 추가
        if use_tta:
            openset_metrics['tta_enabled'] = True
            openset_metrics['tta_views'] = config.openset.tta_n_views
            openset_metrics['tta_repeats_genuine'] = config.openset.tta_n_repeats_genuine
            openset_metrics['tta_repeats_between'] = config.openset.tta_n_repeats_between
            openset_metrics['tta_repeats_negref'] = config.openset.tta_n_repeats_negref
            openset_metrics['tta_repeat_aggregation'] = config.openset.tta_repeat_aggregation
        else:
            openset_metrics['tta_enabled'] = False

        print(f"Evaluating on {len(known_indices)} known + {len(unknown_indices)} unknown samples" +
              (" (TTA enabled)" if use_tta else ""))

        return accuracy, openset_metrics

    else:
        filtered_indices = []
        for i in range(len(test_dataset)):
            label = int(test_dataset.images_label[i])
            if label in known_classes:
                filtered_indices.append(i)

        if not filtered_indices:
            return (0.0, None)

        filtered_test = Subset(test_dataset, filtered_indices)

        print(f"Evaluating on {len(filtered_indices)} test samples from {len(known_classes)} classes")

        accuracy = trainer.evaluate(filtered_test)
        return accuracy, None


def main(args):
    """메인 실행 함수"""

    # 1. Configuration 로드
    config = ConfigParser(args.config)
    config_obj = config.get_config()
    print(f"Using config: {args.config}")
    print(config)

    # config에 seed 설정 추가 (SCRTrainer가 사용)
    if not hasattr(config_obj.training, 'seed'):
        config_obj.training.seed = args.seed

    # BASE_ID 계산 및 설정
    config_obj.negative.base_id = compute_safe_base_id(
        config_obj.dataset.train_set_file,
        config_obj.dataset.test_set_file
    )

    # 오픈셋 모드 확인
    openset_enabled = hasattr(config_obj, 'openset') and config_obj.openset.enabled
    if openset_enabled:
        print("\n========== OPEN-SET MODE ENABLED ==========")
        print(f"   Warmup users: {config_obj.openset.warmup_users}")
        print(f"   Initial tau: {config_obj.openset.initial_tau}")

        # TTA 설정 (시작 시 한 번만 자세히 출력)
        if config_obj.openset.tta_n_views > 1:
            print(f"   TTA Configuration:")
            print(f"      Views: {config_obj.openset.tta_n_views} (+original: {config_obj.openset.tta_include_original})")
            print(f"      Type-specific repeats: G={config_obj.openset.tta_n_repeats_genuine}, "
                  f"B={config_obj.openset.tta_n_repeats_between}, N={config_obj.openset.tta_n_repeats_negref}")
            print(f"      Aggregation: {config_obj.openset.tta_aggregation} (repeat: {config_obj.openset.tta_repeat_aggregation})")

        # Impostor 비율 정보
        print(f"   Impostor Ratios:")
        print(f"      Between: {config_obj.openset.impostor_ratio_between*100:.0f}%")
        print(f"      Unknown: {config_obj.openset.impostor_ratio_unknown*100:.0f}%")
        print(f"      NegRef: {config_obj.openset.impostor_ratio_negref*100:.0f}%")
        print(f"      Total samples: {config_obj.openset.impostor_balance_total}")

        # FAR/EER 모드 표시
        if config_obj.openset.threshold_mode == 'far':
            print(f"   Threshold mode: FAR Target ({config_obj.openset.target_far*100:.1f}%)")
        else:
            print(f"   Threshold mode: EER")

        print("=========================================\n")

    # ProxyAnchorLoss 설정 확인
    if hasattr(config_obj.training, 'use_proxy_anchor') and config_obj.training.use_proxy_anchor:
        print("\n========== PROXY ANCHOR LOSS ENABLED ==========")
        print(f"   Margin (δ): {config_obj.training.proxy_margin}")
        print(f"   Alpha (α): {config_obj.training.proxy_alpha}")
        print(f"   LR Ratio: {config_obj.training.proxy_lr_ratio}x")
        print(f"   Lambda (fixed): {config_obj.training.proxy_lambda}")
        print("===============================================\n")

    # GPU 설정
    device = torch.device(
        f"cuda:{config_obj.training.gpu_ids}"
        if torch.cuda.is_available() and not args.no_cuda
        else "cpu"
    )
    print(f'Device: {device}')

    # 2. 결과 저장 디렉토리 생성
    results_dir = os.path.join(config_obj.training.results_path, 'scr_results')
    os.makedirs(results_dir, exist_ok=True)
    # 3. 데이터 스트림 초기화
    print("\n=== Initializing Data Stream ===")
    data_stream = ExperienceStream(
        train_file=config_obj.dataset.train_set_file,
        negative_file=config_obj.dataset.negative_samples_file,
        num_negative_classes=config_obj.dataset.num_negative_classes,
        base_id=config_obj.negative.base_id
    )

    stats = data_stream.get_statistics()
    print(f"Total users: {stats['num_users']}")
    print(f"Samples per user: {stats['samples_per_user']}")
    print(f"Negative samples: {stats['negative_samples']}")

    # 4. 모델 및 컴포넌트 초기화
    print("\n=== Initializing Model and Components ===")

    # CCNet 모델
    model = ccnet(
        weight=config_obj.model.competition_weight,
        use_projection=config_obj.model.use_projection,
        projection_dim=config_obj.model.projection_dim
    )

    # 프로젝션 헤드 설정 출력
    if config_obj.model.use_projection:
        print(f"Projection Head Configuration:")
        print(f"   Enabled: True")
        print(f"   Dimension: 6144 -> 2048 -> {config_obj.model.projection_dim}")
        print(f"   Structure: 2 layers with LayerNorm")
        print(f"   Training: Uses projection ({config_obj.model.projection_dim}D)")
        print(f"   NCM/Eval: Uses original features (6144D)")
    else:
        print(f"Projection Head: Disabled (using raw 6144D features)")

    # 사전훈련 가중치 로드
    if hasattr(config_obj.model, 'use_pretrained') and config_obj.model.use_pretrained:
        if config_obj.model.pretrained_path and config_obj.model.pretrained_path.exists():
            print(f"\nLoading pretrained weights from main script...")
            loader = PretrainedLoader()
            try:
                model = loader.load_ccnet_pretrained(
                    model=model,
                    checkpoint_path=config_obj.model.pretrained_path,
                    device=device,
                    verbose=True
                )
                print("Pretrained weights loaded successfully!")
            except Exception as e:
                print(f"Failed to load pretrained model: {e}")
                print("Continuing with random initialization...")
        else:
            print(f"Pretrained path not found or not set")
    else:
        print("Starting from random initialization")

    model = model.to(device)

    # NCM Classifier
    ncm_classifier = NCMClassifier(normalize=True).to(device)

    # Memory Buffer
    memory_buffer = ClassBalancedBuffer(
        max_size=config_obj.training.memory_size,
        min_samples_per_class=config_obj.training.min_samples_per_class
    )

    # SCR Trainer
    trainer = SCRTrainer(
        model=model,
        ncm_classifier=ncm_classifier,
        memory_buffer=memory_buffer,
        config=config_obj,
        device=device
    )

    # 5. Negative 샘플로 초기화
    print("\n=== Initializing with Negative Samples ===")
    neg_paths, neg_labels = data_stream.get_negative_samples()

    if len(neg_paths) > config_obj.training.memory_batch_size:
        selected_indices = np.random.choice(
            len(neg_paths),
            size=config_obj.training.memory_batch_size,
            replace=False
        )
        neg_paths = [neg_paths[i] for i in selected_indices]
        neg_labels = [neg_labels[i] for i in selected_indices]

    memory_buffer.update_from_dataset(neg_paths, neg_labels)
    print(f"Initial buffer size: {len(memory_buffer)}")

    print("NCM starts empty - no fake class contamination")

    # 6. 평가자 초기화
    evaluator = ContinualLearningEvaluator(num_experiences=config_obj.training.num_experiences)

    # 7. 학습 결과 저장용
    training_history = {
        'losses': [],
        'accuracies': [],
        'forgetting_measures': [],
        'memory_sizes': [],
        'negative_removal_history': [],
        'openset_metrics': [],
        'pretrained_used': config_obj.model.use_pretrained,
        'pretrained_path': str(config_obj.model.pretrained_path) if config_obj.model.pretrained_path else None,
        'openset_enabled': openset_enabled,
        'seed': config_obj.training.seed,
        'config_path': args.config
    }

    # 8. Continual Learning 시작
    print("\n=== Starting Continual Learning ===")
    print(f"Total experiences: {config_obj.training.num_experiences}")
    print(f"Evaluation interval: every {config_obj.training.test_interval} users")
    print(f"Negative warmup: exp0~{config_obj.negative.warmup_experiences-1}")
    print(f"Learning rate: {config_obj.training.learning_rate}")
    print(f"Memory batch size: {config_obj.training.memory_batch_size}")
    print(f"Temperature: {config_obj.training.temperature}")
    print(f"Random seed: {config_obj.training.seed}")

    start_time = time.time()

    for exp_id, (user_id, image_paths, labels) in enumerate(data_stream):

        # Experience 학습
        stats = trainer.train_experience(user_id, image_paths, labels)
        training_history['losses'].append(stats['loss'])
        training_history['memory_sizes'].append(stats['memory_size'])

        # exp3→exp4 경계에서 네거티브 완전 제거
        if exp_id + 1 == config_obj.negative.warmup_experiences:
            print(f"\n=== Warmup End (exp{exp_id}) → Post-warmup (exp{exp_id+1}) ===")

            acc_pre, _ = evaluate_on_test_set(trainer, config_obj, openset_mode=openset_enabled)
            print(f"[Warmup-End] pre-purge ACC={acc_pre:.2f}%")

            purge_negatives(memory_buffer, config_obj.negative.base_id)
            trainer._update_ncm()

            acc_post, _ = evaluate_on_test_set(trainer, config_obj, openset_mode=openset_enabled)
            print(f"[Warmup-End] post-purge ACC={acc_post:.2f}%")
            print(f"========================================\n")

        # 평가 주기 확인
        if (exp_id + 1) % config_obj.training.test_interval == 0 or exp_id == config_obj.training.num_experiences - 1:

            print(f"\n=== Evaluation at Experience {exp_id + 1} ===")

            # t-SNE 시각화 (디버깅용)
            if (exp_id + 1) % 10 == 0 or exp_id == config_obj.training.num_experiences - 1:
                tsne_dir = os.path.join(results_dir, "tsne")
                os.makedirs(tsne_dir, exist_ok=True)

                tsne_path = os.path.join(tsne_dir, f"tsne_exp_{exp_id+1:03d}.png")
                plot_tsne_from_memory(
                    trainer=trainer,
                    save_path=tsne_path,
                    per_class=150,
                    space='fe',
                    perplexity=30,
                    max_points=5000,
                    seed=42
                )

                if config_obj.model.use_projection:
                    tsne_path_z = os.path.join(tsne_dir, f"tsne_z_exp_{exp_id+1:03d}.png")
                    plot_tsne_from_memory(
                        trainer=trainer,
                        save_path=tsne_path_z,
                        per_class=150,
                        space='z',
                        perplexity=30,
                        max_points=5000,
                        seed=42
                    )

            # 테스트셋으로 평가
            accuracy, openset_metrics = evaluate_on_test_set(
                trainer, config_obj,
                openset_mode=openset_enabled
            )

            # 메트릭 업데이트
            evaluator.update(exp_id, accuracy)
            if openset_metrics:
                evaluator.update_openset(openset_metrics)

            # 평균 정확도와 Forgetting 계산
            avg_acc = evaluator.get_average_accuracy()
            forgetting = evaluator.get_forgetting_measure()

            # 기록 저장
            accuracy_record = {
                'experience': exp_id + 1,
                'accuracy': accuracy,
                'average_accuracy': avg_acc,
                'forgetting': forgetting
            }

            # 오픈셋 메트릭 추가
            if openset_metrics:
                accuracy_record.update(openset_metrics)
                training_history['openset_metrics'].append(openset_metrics)

            training_history['accuracies'].append(accuracy_record)

            print(f"Test Accuracy: {accuracy:.2f}%")
            print(f"Average Accuracy: {avg_acc:.2f}%")
            print(f"Forgetting Measure: {forgetting:.2f}%")
            print(f"Memory Buffer Size: {len(memory_buffer)}")

            # 오픈셋 메트릭 출력
            if openset_metrics:
                print(f"\nOpen-set Metrics:")
                if 'TAR' in openset_metrics:
                    print(f"   TAR: {openset_metrics['TAR']:.3f}, FRR: {openset_metrics['FRR']:.3f}")
                if 'TRR_unknown' in openset_metrics:
                    print(f"   TRR_unknown: {openset_metrics['TRR_unknown']:.3f}, FAR_unknown: {openset_metrics['FAR_unknown']:.3f}")
                print(f"   τ_s: {openset_metrics.get('tau_s', 0):.4f}")

                # TTA 사용 여부만 간단히 표시
                if openset_metrics.get('tta_enabled'):
                    print(f"   TTA: Active")

            # Trainer의 오픈셋 평가 히스토리도 저장
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

    # t-SNE 시각화 경로 추가
    tsne_images = []
    tsne_dir = os.path.join(results_dir, "tsne")
    if os.path.exists(tsne_dir):
        tsne_images = sorted([f for f in os.listdir(tsne_dir) if f.endswith('.png')])
        if tsne_images:
            print(f"\nt-SNE visualizations saved: {len(tsne_images)} images")
            print(f"   Location: {tsne_dir}")

    # 최종 통계
    final_result = training_history['accuracies'][-1] if training_history['accuracies'] else {}
    final_acc = final_result.get('accuracy', 0)
    final_avg_acc = final_result.get('average_accuracy', 0)
    final_forget = final_result.get('forgetting', 0)

    print(f"\n=== Final Results ===")
    print(f"Final Test Accuracy: {final_acc:.2f}%")
    print(f"Final Average Accuracy: {final_avg_acc:.2f}%")
    print(f"Final Forgetting Measure: {final_forget:.2f}%")

    # 최종 오픈셋 메트릭
    if openset_enabled and 'TAR' in final_result:
        print(f"\nFinal Open-set Performance:")
        print(f"   TAR: {final_result.get('TAR', 0):.3f}")
        print(f"   TRR (Unknown): {final_result.get('TRR_unknown', 0):.3f}")
        print(f"   FAR (Unknown): {final_result.get('FAR_unknown', 0):.3f}")
        print(f"   Final τ_s: {final_result.get('tau_s', 0):.4f}")

        # TTA 최종 요약
        if final_result.get('tta_enabled'):
            print(f"   TTA: Enabled (views={final_result.get('tta_views', 1)}, "
                  f"repeats=G{final_result.get('tta_repeats_genuine', 1)}/"
                  f"B{final_result.get('tta_repeats_between', 1)}/"
                  f"N{final_result.get('tta_repeats_negref', 1)})")

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
        'openset_enabled': openset_enabled,
        'seed': config_obj.training.seed
    }

    # 오픈셋 요약 추가
    if openset_enabled and final_result:
        summary['final_openset'] = {
            'TAR': final_result.get('TAR', 0),
            'TRR_unknown': final_result.get('TRR_unknown', 0),
            'FAR_unknown': final_result.get('FAR_unknown', 0),
            'tau_s': final_result.get('tau_s', 0),
            'tta_enabled': final_result.get('tta_enabled', False)
        }

    summary_path = os.path.join(results_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SCR Training with ProxyAnchorLoss, TTA Support')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA')

    args = parser.parse_args()
    main(args)