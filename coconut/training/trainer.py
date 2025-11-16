# coconut/training/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.optim import lr_scheduler
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
import copy
import os
import random

# COCONUT 모듈 import
from coconut.losses import SupConLoss, ProxyAnchorLoss
from coconut.data import MemoryDataset

# 기존 모듈 import (점진적 마이그레이션 예정)
from coconut.data import get_scr_transforms
from .average_meter import AverageMeter
from coconut.models import PretrainedLoader
from coconut.classifiers.threshold import ThresholdCalibrator

# 오픈셋 유틸리티 함수들
try:
    from coconut.openset import (
        split_user_data,
        extract_scores_genuine,
        extract_scores_impostor_between,
        extract_scores_impostor_unknown,
        extract_scores_impostor_negref,
        balance_impostor_scores,
        predict_batch,
        load_paths_labels_from_txt,
        # TTA 관련
        predict_batch_tta,
        extract_scores_genuine_tta,
        extract_scores_impostor_between_tta,
        extract_scores_impostor_negref_tta,
        set_seed
    )
    from coconut.openset.utils import _open_with_channels
    TTA_FUNCTIONS_AVAILABLE = True
except ImportError:
    from coconut.openset import (
        split_user_data,
        extract_scores_genuine,
        extract_scores_impostor_between,
        extract_scores_impostor_unknown,
        extract_scores_impostor_negref,
        balance_impostor_scores,
        predict_batch,
        load_paths_labels_from_txt,
        set_seed
    )
    TTA_FUNCTIONS_AVAILABLE = False
    print("WARNING: TTA functions not available, using max score only")

    def _open_with_channels(path: str, channels: int):
        img = Image.open(path)
        if channels == 1:
            return img.convert('L')
        else:
            return img.convert('RGB')


def worker_init_fn(worker_id):
    """
    Initialize each DataLoader worker with unique seed for reproducibility
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def repeat_and_augment_data(paths, labels, target_size):
    """데이터를 목표 크기에 맞춰 반복 증강"""
    if not paths or target_size <= 0:
        return [], []

    if len(paths) >= target_size:
        # 충분한 데이터가 있으면 랜덤 샘플링
        indices = torch.randperm(len(paths))[:target_size]
        return [paths[i] for i in indices], [labels[i] for i in indices]

    # 부족한 경우 순환 반복해서 목표 크기 달성
    repeated_paths = []
    repeated_labels = []

    for i in range(target_size):
        idx = i % len(paths)  # 순환 인덱스
        repeated_paths.append(paths[idx])
        repeated_labels.append(labels[idx])

    return repeated_paths, repeated_labels

# MemoryDataset moved to coconut.data.datasets


class COCONUTTrainer:
    """
    COCONUT Trainer: CCNet + ProxyAnchor + Supervised Contrastive Replay

    Combines:
    - CCNet backbone
    - ProxyAnchorLoss for metric learning
    - Supervised Contrastive Learning (SupCon)
    - Memory replay for continual learning
    - Open-set recognition with TTA support
    """

    def __init__(self,
                 model: nn.Module,
                 ncm_classifier,
                 memory_buffer,
                 config,
                 device='cuda'):

        # 시드 설정
        seed = getattr(config.training, 'seed', 42) if hasattr(config, 'training') else 42
        set_seed(seed)
        print(f" Random seed set to {seed}")

        # 모델/NCM 디바이스 이동
        self.device = device
        model = model.to(device)
        if hasattr(ncm_classifier, 'to'):
            ncm_classifier = ncm_classifier.to(device)

        # 사전훈련 로딩 (한 번만 실행)
        if hasattr(config.model, 'use_pretrained') and config.model.use_pretrained:
            if config.model.pretrained_path and config.model.pretrained_path.exists():
                print(f"\n🎯 Loading pretrained weights...")
                print(f"   Path: {config.model.pretrained_path}")
                loader = PretrainedLoader()
                try:
                    model = loader.load_ccnet_pretrained(
                        model=model,
                        checkpoint_path=config.model.pretrained_path,
                        device=device,
                        verbose=True
                    )
                    print("✅ Pretrained weights loaded successfully!")
                except Exception as e:
                    print(f"⚠️ WARNING: Failed to load pretrained: {e}")
                    print("   Continuing with current weights...")
            else:
                print(f"⚠️ WARNING: Pretrained path not found: {config.model.pretrained_path}")
        else:
            print("📦 Using random initialization (no pretrained weights)")

        self.model = model
        self.ncm = ncm_classifier
        self.memory_buffer = memory_buffer
        self.config = config

        # Loss function
        self.criterion = SupConLoss(
            temperature=config.training.temperature,
            base_temperature=config.training.temperature
        )

        # ProxyAnchorLoss 초기화
        self.use_proxy_anchor = getattr(config.training, 'use_proxy_anchor', True)

        if self.use_proxy_anchor:
            #  실제 특징 차원에 맞춰 ProxyAnchor 초기화
            if config.model.use_projection:
                embedding_dim = config.model.projection_dim  # 프로젝션 헤드 사용 시
            else:
                embedding_dim = 2048  # getFeatureCode() 출력 차원 (fc1 출력)

            self.proxy_anchor_loss = ProxyAnchorLoss(
                embedding_size=embedding_dim,
                margin=getattr(config.training, 'proxy_margin', 0.1),
                alpha=getattr(config.training, 'proxy_alpha', 32)
            ).to(device)

            print(f"[COCONUT] COCONUT ProxyAnchor initialized with {embedding_dim}D embeddings")

            self.proxy_lambda = getattr(config.training, 'proxy_lambda', 0.3)

            print(f"[COCONUT] ProxyAnchorLoss enabled:")
            print(f"   Margin (δ): {self.proxy_anchor_loss.margin}")
            print(f"   Alpha (α): {self.proxy_anchor_loss.alpha}")
            print(f"   Lambda (fixed): {self.proxy_lambda}")
        else:
            self.proxy_anchor_loss = None
            self.proxy_lambda = 0.0

        # [CORE] 핵심 수정: 옵티마이저 관리 개선
        self.base_lr = config.training.learning_rate
        self.proxy_lr_ratio = getattr(config.training, 'proxy_lr_ratio', 10) if self.use_proxy_anchor else 1

        # 초기 옵티마이저 (그룹별 학습률 적용)
        self.optimizer = self._create_optimizer_with_grouped_params(include_proxies=False)

        # 스케줄러
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.training.scheduler_step_size,
            gamma=config.training.scheduler_gamma
        )

        # 옵티마이저 재생성 추적
        self.last_num_proxies = 0

        # Transform
        self.train_transform = get_scr_transforms(
            train=True,
            imside=config.dataset.height,
            channels=config.dataset.channels
        )

        self.test_transform = get_scr_transforms(
            train=False,
            imside=config.dataset.height,
            channels=config.dataset.channels
        )

        # === 오픈셋 관련 초기화 ===
        self.openset_enabled = hasattr(config, 'openset') and config.openset.enabled

        if self.openset_enabled:
            self.openset_config = config.openset
            self._first_calibration_done = False

            # TTA 설정 확인
            self.use_tta = self.openset_config.tta_n_views > 1
            if self.use_tta:
                if TTA_FUNCTIONS_AVAILABLE:
                    print(f"[TARGET] TTA enabled for evaluation:")
                    print(f"   Views: {self.openset_config.tta_n_views}")
                    print(f"   Include original: {self.openset_config.tta_include_original}")
                    print(f"   Augmentation: {self.openset_config.tta_augmentation_strength}")
                    print(f"   Aggregation: {self.openset_config.tta_aggregation}")

                    # 타입별 반복 설정 출력
                    print(f"   Type-specific repeats:")
                    print(f"     - Genuine: {self.openset_config.tta_n_repeats_genuine}")
                    print(f"     - Between: {self.openset_config.tta_n_repeats_between}")
                else:
                    print("WARNING: TTA requested but functions not available")
                    self.use_tta = False

            # 모드 로그
            mode_str = f"MAX + TTA({self.openset_config.tta_n_views})" if self.use_tta else "MAX"
            print(f" Open-set mode: {mode_str}")

            # ThresholdCalibrator 초기화
            self.threshold_calibrator = ThresholdCalibrator(
                mode="cosine",
                threshold_mode=config.openset.threshold_mode,
                target_far=config.openset.target_far,
                alpha=config.openset.threshold_alpha,
                max_delta=config.openset.threshold_max_delta,
                clip_range=(-1.0, 1.0),
                min_samples=10,
                verbose=config.openset.verbose_calibration
            )

            # Dev/Train 데이터 관리
            self.dev_data = {}
            self.train_data = {}
            self.registered_users = set()

            # 평가 히스토리
            self.evaluation_history = []

            # 초기 임계치 설정
            initial_tau = config.openset.initial_tau
            if hasattr(self.ncm, 'set_thresholds'):
                self.ncm.set_thresholds(tau_s=initial_tau)
            else:
                self.ncm.tau_s = initial_tau

            # 모드별 초기값 조정
            if config.openset.threshold_mode == 'far':
                print(f"[TARGET] FAR Target mode enabled")
                print(f"   Target FAR: {config.openset.target_far*100:.1f}%")
            else:
                print(f"[INFO] EER mode enabled")

            print(f"   Initial τ_s: {initial_tau}")
        else:
            self.registered_users = set()
            self.use_tta = False
            print(" Open-set mode disabled")

        # Statistics
        self.experience_count = 0

        # BASE_ID 저장
        self.base_id = int(config.negative.base_id) if hasattr(config, 'negative') else 10000
        print(f"[COCONUT] COCONUTTrainer using BASE_ID: {self.base_id}")

        # 워밍업 관련 파라미터
        self.warmup_T = int(config.negative.warmup_experiences) if hasattr(config, 'negative') else 4
        self.r0 = float(config.negative.r0) if hasattr(config, 'negative') else 0.5
        self.max_neg_per_class = int(config.negative.max_per_batch) if hasattr(config, 'negative') else 1

        # CuDNN 최적화
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            print("[INIT] CuDNN benchmark enabled (speed priority)")

        # 사전훈련 사용 여부 로그
        if hasattr(config.model, 'use_pretrained') and config.model.use_pretrained:
            print(f"[CORE] COCONUTTrainer initialized with pretrained model")
        else:
            print(f" COCONUTTrainer initialized with random weights")


    def _create_optimizer_with_grouped_params(self, include_proxies=True):
        """ 파라미터 그룹별로 다른 학습률 적용한 옵티마이저 생성"""
        param_groups = []

        # 백본과 프로젝션 헤드 파라미터 분리
        backbone_params = []
        projection_params = []

        for name, param in self.model.named_parameters():
            if 'projection_head' in name:
                projection_params.append(param)
            else:
                backbone_params.append(param)

        # 백본 파라미터 그룹 (사전학습된 CCNet)
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': self.config.training.learning_rate,
                'name': 'backbone'
            })
            print(f"️ Backbone LR: {self.config.training.learning_rate:.6f}")

        # 프로젝션 헤드 파라미터 그룹 (새로 초기화된 레이어)
        if projection_params and self.config.model.use_projection:
            param_groups.append({
                'params': projection_params,
                'lr': self.config.training.projection_learning_rate,
                'name': 'projection'
            })
            print(f" Projection Head LR: {self.config.training.projection_learning_rate:.6f}")

        # 프록시 파라미터 그룹 (새로 초기화된 프록시)
        if include_proxies and self.use_proxy_anchor and hasattr(self, 'proxy_anchor_loss') and self.proxy_anchor_loss.proxies is not None:
            param_groups.append({
                'params': [self.proxy_anchor_loss.proxies],
                'lr': self.base_lr * self.proxy_lr_ratio,
                'name': 'proxies'
            })
            print(f"[COCONUT] Proxies LR: {self.base_lr * self.proxy_lr_ratio:.6f} ({self.proxy_lr_ratio}x)")

        return optim.Adam(param_groups)

    def _recreate_optimizer_with_proxies(self):
        """[CORE] 핵심 수정: 프록시 추가 시 옵티마이저 안전 재생성 (상태 복원 제거)"""
        if not self.use_proxy_anchor or self.proxy_anchor_loss.proxies is None:
            return

        current_num_proxies = self.proxy_anchor_loss.num_classes

        # 프록시 수가 변경된 경우에만 재생성
        if current_num_proxies != self.last_num_proxies:
            # 새 옵티마이저 생성 (그룹별 학습률 적용)
            self.optimizer = self._create_optimizer_with_grouped_params(include_proxies=True)

            # 스케줄러 재생성
            self.scheduler = lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.scheduler_step_size,
                gamma=self.config.training.scheduler_gamma
            )

            self.last_num_proxies = current_num_proxies
            print(f" Optimizer recreated with {current_num_proxies} proxies (fresh state)")

    def _dedup_negative_classes(self, paths, labels, max_per_class=1):
        """네거티브 클래스 중복 제거"""
        out_p, out_l, seen = [], [], set()
        for p, l in zip(paths, labels):
            li = int(l)
            if li >= self.base_id:
                if (li in seen) and (max_per_class == 1):
                    continue
                seen.add(li)
            out_p.append(p)
            out_l.append(l)
        return out_p, out_l

    def _memory_cap_for_ratio(self, num_current, r):
        """r0 비율 제한 계산"""
        if r <= 0:
            return 0
        return int((r / max(1e-8, 1.0 - r)) * num_current)

    @torch.no_grad()
    def _extract_features_with_tta(self, batch_images):
        """
        Extract features with Test-Time Augmentation (TTA).

        Args:
            batch_images: Batch of images (B, C, H, W)

        Returns:
            Features tensor (B, D) where D is the feature dimension
        """
        if not hasattr(self, 'openset_config'):
            # No TTA config, fallback to normal extraction using getFeatureCode
            return self.model.getFeatureCode(batch_images)

        # TTA configuration
        n_views = self.openset_config.tta_n_views
        include_original = self.openset_config.tta_include_original
        aug_strength = self.openset_config.tta_augmentation_strength
        aggregation = self.openset_config.tta_aggregation

        # Get augmentation transform
        from torchvision import transforms as T
        from ..openset.tta_operations import get_light_augmentation

        light_aug = get_light_augmentation(aug_strength, self.config.dataset.height)

        batch_size = batch_images.shape[0]
        all_features = []

        self.model.eval()

        for i in range(batch_size):
            img = batch_images[i]  # Single image tensor
            view_features = []

            # Process multiple views
            for v_idx in range(n_views):
                if v_idx == 0 and include_original:
                    # Use original image for first view
                    view = img
                else:
                    # Apply augmentation for other views
                    # Convert tensor to PIL for augmentation
                    img_pil = T.ToPILImage()(img.cpu())
                    aug_img = light_aug(img_pil) if aug_strength > 0 else img_pil
                    view = T.ToTensor()(aug_img).to(img.device)

                # Extract features using getFeatureCode for NCM compatibility (6144D)
                view = view.unsqueeze(0)  # Add batch dimension
                features = self.model.getFeatureCode(view)  # Always use getFeatureCode for NCM

                view_features.append(features.squeeze(0))

            # Aggregate features from multiple views
            view_features = torch.stack(view_features)
            if aggregation == 'mean':
                aggregated = view_features.mean(dim=0)
            elif aggregation == 'median':
                aggregated = view_features.median(dim=0)[0]
            else:
                aggregated = view_features.mean(dim=0)  # Default to mean

            all_features.append(aggregated)

        return torch.stack(all_features)

    def train_experience(self, user_id: int, image_paths: List[str], labels: List[int]) -> Dict:
        """하나의 experience (한 명의 사용자) 학습 - 오픈셋 지원."""

        print(f"\n=== Training Experience {self.experience_count}: User {user_id} ===")

        # [CORE] 핵심 수정: 프록시 추가 및 옵티마이저 재생성
        if self.use_proxy_anchor:
            unique_labels = set(labels)
            real_classes = [l for l in unique_labels if l < self.base_id]

            if real_classes:
                self.proxy_anchor_loss.add_classes(real_classes)
                self._recreate_optimizer_with_proxies()

        # 원본 labels를 보존
        original_labels = labels.copy()

        # Step 1: Dev/Train 분리 (오픈셋 모드인 경우)
        if self.openset_enabled:
            train_paths, train_labels, dev_paths, dev_labels = split_user_data(
                image_paths, original_labels,
                dev_ratio=self.openset_config.dev_ratio,
                min_dev=2
            )

            # 저장
            self.dev_data[user_id] = (dev_paths, dev_labels)
            self.train_data[user_id] = (train_paths, train_labels)

            print(f"[INFO] Data split: Train={len(train_paths)}, Dev={len(dev_paths)}")
        else:
            train_paths = image_paths
            train_labels = original_labels

        self.registered_users.add(user_id)

        #  현재 사용자 데이터를 experience_batch_size만큼 증강
        augmented_current_paths, augmented_current_labels = repeat_and_augment_data(
            train_paths, train_labels, self.config.training.experience_batch_size
        )

        # 증강된 현재 사용자 데이터셋 생성
        current_dataset = MemoryDataset(
            paths=augmented_current_paths,
            labels=augmented_current_labels,
            transform=self.train_transform,
            train=True,
            channels=self.config.dataset.channels
        )

        # 학습 통계
        loss_avg = AverageMeter('Loss')

        # SCR 논문 방식: epoch당 여러 iteration
        self.model.train()

        for epoch in range(self.config.training.epochs_per_experience):
            epoch_loss = 0

            for iteration in range(self.config.training.iterations_per_epoch):

                #  증강된 현재 데이터를 전체 사용 (이미 experience_batch_size로 맞춰짐)
                current_subset = current_dataset

                # 메모리에서 샘플링
                if len(self.memory_buffer) > 0:
                    memory_paths, memory_labels = self.memory_buffer.sample(
                        self.config.training.memory_batch_size
                    )

                    if torch.is_tensor(memory_labels):
                        memory_labels = memory_labels.cpu().tolist()

                    # 워밍업 규칙 적용
                    if self.experience_count < self.warmup_T:
                        memory_paths, memory_labels = self._dedup_negative_classes(
                            memory_paths, memory_labels, max_per_class=self.max_neg_per_class
                        )

                        cap = self._memory_cap_for_ratio(self.config.training.experience_batch_size, self.r0)
                        if cap >= 0 and len(memory_paths) > cap:
                            idx = np.random.choice(len(memory_paths), size=cap, replace=False)
                            memory_paths = [memory_paths[i] for i in idx]
                            memory_labels = [memory_labels[i] for i in idx]
                    else:
                        kept = [(p, l) for p, l in zip(memory_paths, memory_labels) if int(l) < self.base_id]
                        memory_paths, memory_labels = (list(map(list, zip(*kept))) if kept else ([], []))

                    if memory_paths:
                        #  메모리 데이터를 memory_batch_size만큼 증강
                        augmented_memory_paths, augmented_memory_labels = repeat_and_augment_data(
                            memory_paths, memory_labels, self.config.training.memory_batch_size
                        )

                        memory_dataset = MemoryDataset(
                            paths=augmented_memory_paths,
                            labels=augmented_memory_labels,
                            transform=self.train_transform,
                            train=True,
                            channels=self.config.dataset.channels
                        )

                        combined_dataset = ConcatDataset([current_subset, memory_dataset])
                    else:
                        combined_dataset = current_subset
                else:
                    combined_dataset = current_subset

                # DataLoader로 배치 생성
                batch_loader = DataLoader(
                    combined_dataset,
                    batch_size=len(combined_dataset),
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                    persistent_workers=False
                )

                # 학습
                for data, batch_labels in batch_loader:
                    batch_size = len(batch_labels)

                    view1 = data[0]
                    view2 = data[1]

                    x = torch.cat([view1, view2], dim=0).to(self.device, non_blocking=True)
                    batch_labels = batch_labels.to(self.device, non_blocking=True)

                    self.optimizer.zero_grad()

                    # Forward: CCNet이 알아서 처리 (projection 포함)
                    features_all = self.model(x)

                    f1 = features_all[:batch_size]
                    f2 = features_all[batch_size:]

                    # CCNet 결과 그대로 사용
                    features_paired = torch.stack([f1, f2], dim=1)

                    # SupConLoss
                    loss_supcon = self.criterion(features_paired, batch_labels)

                    # ProxyAnchorLoss 추가
                    if self.use_proxy_anchor and self.proxy_anchor_loss.proxies is not None:
                        all_labels = batch_labels.repeat(2)
                        loss_proxy = self.proxy_anchor_loss(features_all, all_labels)

                        # 고정 가중치로 결합
                        loss = (1 - self.proxy_lambda) * loss_supcon + self.proxy_lambda * loss_proxy

                        if iteration == 0 and epoch == 0:
                            print(f"[COCONUT] Losses - SupCon: {loss_supcon.item():.4f}, ProxyAnchor: {loss_proxy.item():.4f}, λ: {self.proxy_lambda}")
                    else:
                        loss = loss_supcon

                    loss_avg.update(loss.item(), batch_size)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    if self.use_proxy_anchor and self.proxy_anchor_loss.proxies is not None:
                        torch.nn.utils.clip_grad_norm_(self.proxy_anchor_loss.proxies, max_norm=1.0)

                    self.optimizer.step()

                    epoch_loss += loss.item()

            if (epoch + 1) % 1 == 0:
                avg_loss = epoch_loss / self.config.training.iterations_per_epoch
                print(f"  Epoch [{epoch+1}/{self.config.training.epochs_per_experience}] Loss: {avg_loss:.4f}")

                #  에포크마다 코사인 유사도 분포 분석
                if (epoch + 1) % 5 == 0 or epoch == self.config.training.epochs_per_experience - 1:
                    self._analyze_cosine_distribution_epoch()

        # 메모리 버퍼 업데이트
        self.memory_buffer.update_from_dataset(train_paths, train_labels)
        print(f"Memory buffer size after update: {len(self.memory_buffer)}")

        # NCM 업데이트
        self._update_ncm()

        # 디버깅: NCM과 버퍼 동기화 확인
        all_paths, all_labels = self.memory_buffer.get_all_data()
        buffer_classes = set(int(label) for label in all_labels)
        ncm_classes = set(self.ncm.class_means_dict.keys())
        missing = buffer_classes - ncm_classes

        if missing:
            print(f"WARNING:  NCM missing classes: {sorted(list(missing))}")
        else:
            print(f"[OK] NCM synchronized: {len(ncm_classes)} classes")

        # 주기적 캘리브레이션 및 평가 (오픈셋 모드)
        if self.openset_enabled and len(self.registered_users) % self.config.training.test_interval == 0:

            if len(self.registered_users) >= self.openset_config.warmup_users:
                print("\n" + "="*60)
                print(" THRESHOLD CALIBRATION & EVALUATION")
                print("="*60)

                self._calibrate_threshold()
                metrics = self._evaluate_openset()

                self.evaluation_history.append({
                    'num_users': len(self.registered_users),
                    'tau_s': self.ncm.tau_s,
                    'metrics': metrics
                })

                print("="*60 + "\n")
            else:
                print(f" Warmup phase: {len(self.registered_users)}/{self.openset_config.warmup_users}")

        self.experience_count += 1
        self.scheduler.step()

        return {
            'experience': self.experience_count - 1,
            'user_id': user_id,
            'loss': loss_avg.avg,
            'memory_size': len(self.memory_buffer),
            'num_registered': len(self.registered_users)
        }

    @torch.no_grad()
    def _calibrate_threshold(self):
        """임계치 캘리브레이션 (타입별 독립 TTA 반복, 동적 비율)"""

        use_tta = self.use_tta and TTA_FUNCTIONS_AVAILABLE

        # 타입별 독립적인 TTA 반복 설정
        n_repeats_genuine = getattr(self.openset_config, 'tta_n_repeats_genuine', 1)
        n_repeats_between = getattr(self.openset_config, 'tta_n_repeats_between', 1)
        n_repeats_negref = 0  # NegRef not used

        repeat_agg = getattr(self.openset_config, 'tta_repeat_aggregation', 'median')
        tta_verbose = getattr(self.openset_config, 'tta_verbose', False)

        seed = getattr(self.config.training, 'seed', 42)
        img_size = self.config.dataset.height
        channels = self.config.dataset.channels

        # 모드 표시
        mode_str = f"MAX + TTA(views={self.openset_config.tta_n_views})" if use_tta else "MAX"
        print(f" Using {mode_str} scores for calibration")

        # 타입별 반복 설정 출력
        if use_tta:
            print(f" TTA with type-specific repeats:")
            print(f"   Genuine: {self.openset_config.tta_n_views} views × {n_repeats_genuine} repeats")
            print(f"   Between: {self.openset_config.tta_n_views} views × {n_repeats_between} repeats")
            print(f"   NegRef: {self.openset_config.tta_n_views} views × {n_repeats_negref} repeats")

        print(f"\n Extracting scores for {self.openset_config.threshold_mode.upper()} calibration...")

        # Dev 데이터 수집
        all_dev_paths = []
        all_dev_labels = []
        for uid, (paths, labels) in self.dev_data.items():
            all_dev_paths.extend(paths)
            all_dev_labels.extend([uid] * len(paths))

        # TTA 모드에 따른 점수 추출
        if use_tta:
            s_genuine = extract_scores_genuine_tta(
                self.model, self.ncm,
                all_dev_paths, all_dev_labels,
                self.test_transform, self.device,
                n_views=self.openset_config.tta_n_views,
                include_original=self.openset_config.tta_include_original,
                aug_strength=self.openset_config.tta_augmentation_strength,
                aggregation=self.openset_config.tta_aggregation,
                img_size=img_size,
                channels=channels,
                n_repeats=n_repeats_genuine,
                repeat_aggregation=repeat_agg,
                verbose=tta_verbose,
                seed=seed
            )

            s_imp_between = extract_scores_impostor_between_tta(
                self.model, self.ncm,
                all_dev_paths, all_dev_labels,
                self.test_transform, self.device,
                n_views=self.openset_config.tta_n_views,
                include_original=self.openset_config.tta_include_original,
                aug_strength=self.openset_config.tta_augmentation_strength,
                aggregation=self.openset_config.tta_aggregation,
                max_pairs=6000,
                img_size=img_size,
                channels=channels,
                n_repeats=n_repeats_between,
                repeat_aggregation=repeat_agg,
                verbose=tta_verbose,
                seed=seed
            )

            s_imp_negref = extract_scores_impostor_negref_tta(
                self.model, self.ncm,
                self.config.dataset.negative_samples_file,
                self.test_transform, self.device,
                n_views=self.openset_config.tta_n_views,
                include_original=self.openset_config.tta_include_original,
                aug_strength=self.openset_config.tta_augmentation_strength,
                aggregation=self.openset_config.tta_aggregation,
                img_size=img_size,
                channels=channels,
                n_repeats=n_repeats_negref,
                repeat_aggregation=repeat_agg,
                verbose=tta_verbose,
                seed=seed,
                force_memory_save=False
            )
        else:
            # 단일뷰 - 최댓값 버전
            s_genuine = extract_scores_genuine(
                self.model, self.ncm,
                all_dev_paths, all_dev_labels,
                self.test_transform, self.device,
                channels=channels
            )

            s_imp_between = extract_scores_impostor_between(
                self.model, self.ncm,
                all_dev_paths, all_dev_labels,
                self.test_transform, self.device,
                max_pairs=2000,
                channels=channels
            )

            s_imp_negref = extract_scores_impostor_negref(
                self.model, self.ncm,
                self.config.dataset.negative_samples_file,
                self.test_transform, self.device,
                channels=channels
            )

        # 동적 비율로 균형 맞추기
        # Use only between impostor scores (100%)
        impostor_ratio = (
            1.0,  # Between: 100%
            0.0,  # Unknown: 0%
            0.0   # NegRef: 0%
        )

        s_impostor = balance_impostor_scores(
            s_imp_between,
            None,  # Unknown not used
            None,  # NegRef not used
            ratio=impostor_ratio,
            total=self.openset_config.impostor_balance_total
        )

        print(f"   Genuine: {len(s_genuine)} scores")
        print(f"   Impostor: {len(s_impostor)} scores (100% between)")

        # 캘리브레이션
        if len(s_genuine) >= 10 and len(s_impostor) >= 10:
            result = self.threshold_calibrator.calibrate(
                genuine_scores=s_genuine,
                impostor_scores=s_impostor,
                old_tau=None if not self._first_calibration_done else self.ncm.tau_s
            )
            self._first_calibration_done = True

            # NCM에 적용
            new_tau = result['tau_smoothed']
            if hasattr(self.ncm, 'set_thresholds'):
                self.ncm.set_thresholds(tau_s=new_tau)
            else:
                self.ncm.tau_s = new_tau

            # 모드별 출력
            if self.openset_config.threshold_mode == 'eer':
                print(f" EER Mode Results:")
                print(f"   EER: {result.get('eer', 0)*100:.2f}%")
                print(f"   Threshold: {result['tau_smoothed']:.4f}")
            else:
                print(f"[TARGET] FAR Target Results:")
                print(f"   Target FAR: {self.openset_config.target_far*100:.1f}%")
                print(f"   Achieved FAR: {result.get('current_far', 0)*100:.1f}%")
                print(f"   Threshold: {result['tau_smoothed']:.4f}")
        else:
            print("WARNING: Not enough samples for calibration")

    @torch.no_grad()
    def _evaluate_openset(self):
        """오픈셋 평가 (TTA 지원)"""

        #  재현성: 평가 시 시드 재설정
        eval_seed = getattr(self.config.training, 'seed', 42)
        set_seed(eval_seed)

        use_tta = self.use_tta and TTA_FUNCTIONS_AVAILABLE
        img_size = self.config.dataset.height
        channels = self.config.dataset.channels

        if use_tta:
            print(f"\n Open-set Evaluation with TTA (n={self.openset_config.tta_n_views}):")
        else:
            print("\n Open-set Evaluation (Single View):")

        # 변수 초기화
        TAR = FRR = 0.0
        TRR_u = FAR_u = 0.0
        TRR_n = FAR_n = None

        # 1. Known-Dev (TAR/FRR)
        all_dev_paths = []
        all_dev_labels = []
        for uid, (paths, labels) in self.dev_data.items():
            all_dev_paths.extend(paths)
            all_dev_labels.extend([uid] * len(paths))

        if all_dev_paths:
            if use_tta:
                preds, details = predict_batch_tta(
                    self.model, self.ncm,
                    all_dev_paths, self.test_transform, self.device,
                    n_views=self.openset_config.tta_n_views,
                    include_original=self.openset_config.tta_include_original,
                    agree_k=self.openset_config.tta_agree_k,
                    aug_strength=self.openset_config.tta_augmentation_strength,
                    return_details=True,
                    img_size=img_size,
                    channels=channels,
                    seed=eval_seed + 5000  #  재현성: seed 전달
                )

                # TTA 통계 출력
                if details:
                    reject_reasons = [d.get('reject_reason') for d in details if d.get('reject_reason')]
                    if reject_reasons:
                        from collections import Counter
                        reason_counts = Counter(reject_reasons)
                        print(f"   TTA Reject reasons: gate={reason_counts.get('gate', 0)}, "
                              f"class={reason_counts.get('class', 0)}, both={reason_counts.get('both', 0)}")
            else:
                preds = predict_batch(
                    self.model, self.ncm,
                    all_dev_paths, self.test_transform, self.device,
                    channels=channels
                )

            # TAR/FRR 계산
            correct = sum(1 for p, l in zip(preds, all_dev_labels) if p == l)
            rejected = sum(1 for p in preds if p == -1)

            TAR = correct / max(1, len(preds))
            FRR = rejected / max(1, len(preds))

        # 2. Unknown (TRR/FAR)
        unknown_paths, unknown_labels = load_paths_labels_from_txt(
            str(self.config.dataset.train_set_file)
        )

        unknown_filtered = []
        for p, l in zip(unknown_paths, unknown_labels):
            if l not in self.registered_users:
                unknown_filtered.append(p)

        if len(unknown_filtered) > 1000:
            #  재현성: config seed 기반 샘플링
            np.random.seed(eval_seed + 1000)  # deterministic offset
            unknown_filtered = np.random.choice(unknown_filtered, 1000, replace=False).tolist()

        if unknown_filtered:
            if use_tta:
                preds_unk = predict_batch_tta(
                    self.model, self.ncm,
                    unknown_filtered, self.test_transform, self.device,
                    n_views=self.openset_config.tta_n_views,
                    include_original=self.openset_config.tta_include_original,
                    agree_k=self.openset_config.tta_agree_k,
                    aug_strength=self.openset_config.tta_augmentation_strength,
                    img_size=img_size,
                    channels=channels,
                    seed=eval_seed + 3000  #  재현성: seed 전달
                )
            else:
                preds_unk = predict_batch(
                    self.model, self.ncm,
                    unknown_filtered, self.test_transform, self.device,
                    channels=channels
                )
            TRR_u = sum(1 for p in preds_unk if p == -1) / len(preds_unk)
            FAR_u = 1 - TRR_u

        # 3. NegRef
        negref_paths, _ = load_paths_labels_from_txt(
            str(self.config.dataset.negative_samples_file)
        )

        if len(negref_paths) > 1000:
            #  재현성: config seed 기반 샘플링
            np.random.seed(eval_seed + 2000)  # deterministic offset
            negref_paths = np.random.choice(negref_paths, 1000, replace=False).tolist()

        if negref_paths:
            if use_tta:
                preds_neg = predict_batch_tta(
                    self.model, self.ncm,
                    negref_paths, self.test_transform, self.device,
                    n_views=self.openset_config.tta_n_views,
                    include_original=self.openset_config.tta_include_original,
                    agree_k=self.openset_config.tta_agree_k,
                    aug_strength=self.openset_config.tta_augmentation_strength,
                    img_size=img_size,
                    channels=channels,
                    seed=eval_seed + 4000  #  재현성: seed 전달
                )
            else:
                preds_neg = predict_batch(
                    self.model, self.ncm,
                    negref_paths, self.test_transform, self.device,
                    channels=channels
                )
            TRR_n = sum(1 for p in preds_neg if p == -1) / len(preds_neg)
            FAR_n = 1 - TRR_n

        # 결과 출력
        print(f"   Known-Dev: TAR={TAR:.3f}, FRR={FRR:.3f}")
        print(f"   Unknown: TRR={TRR_u:.3f}, FAR={FAR_u:.3f}")
        if TRR_n is not None:
            print(f"   NegRef: TRR={TRR_n:.3f}, FAR={FAR_n:.3f}")
        print(f"   Threshold: τ_s={self.ncm.tau_s:.4f}")

        # 모드 정보 추가
        if self.openset_config.threshold_mode == 'far':
            print(f"   Mode: FAR Target ({self.openset_config.target_far*100:.1f}%)")
        else:
            print(f"   Mode: EER")

        print(f"   Score: Max")

        if use_tta:
            print(f"   [TARGET] TTA: {self.openset_config.tta_n_views} views, agree_k={self.openset_config.tta_agree_k}")

        return {
            'TAR': TAR, 'FRR': FRR,
            'TRR_unknown': TRR_u, 'FAR_unknown': FAR_u,
            'TRR_negref': TRR_n, 'FAR_negref': FAR_n,
            'mode': self.openset_config.threshold_mode,
            'score_type': 'max',
            'tta_enabled': use_tta,
            'tta_views': self.openset_config.tta_n_views if use_tta else 1
        }

    @torch.no_grad()
    def _update_ncm(self):
        """NCM classifier의 class means를 업데이트합니다."""
        if len(self.memory_buffer) == 0:
            return

        self.model.eval()

        # 메모리 버퍼에서 모든 데이터 가져오기
        all_paths, all_labels = self.memory_buffer.get_all_data()

        # 가짜 클래스 필터링
        real_paths = []
        real_labels = []
        fake_count = 0

        for path, label in zip(all_paths, all_labels):
            label_int = int(label) if not isinstance(label, int) else label
            if label_int < self.base_id:
                real_paths.append(path)
                real_labels.append(label)
            else:
                fake_count += 1

        if not real_paths:
            print("WARNING: No real users for NCM update (NCM remains empty)")
            return

        if fake_count > 0:
            print(f" NCM update: {len(real_paths)} real samples ({fake_count} fake samples filtered out)")

        # 데이터셋 생성
        dataset = MemoryDataset(
            paths=real_paths,
            labels=real_labels,
            transform=self.test_transform,
            train=False,
            channels=self.config.dataset.channels
        )

        dataloader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            worker_init_fn=worker_init_fn if self.config.training.num_workers > 0 else None,
            pin_memory=True,
            persistent_workers=False
        )

        # 클래스별로 features 수집
        class_features = {}

        for data, labels in dataloader:
            data = data.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            features = self.model.getFeatureCode(data)

            for i, label in enumerate(labels):
                label_item = label.item()
                if label_item not in class_features:
                    class_features[label_item] = []
                class_features[label_item].append(features[i].cpu())

        # 클래스별 평균 계산
        class_means = {}
        for label, features_list in class_features.items():
            if len(features_list) > 0:
                mean_feature = torch.stack(features_list).mean(dim=0)
                mean_feature = mean_feature / (mean_feature.norm(p=2) + 1e-12)
                class_means[label] = mean_feature

        # NCM 업데이트
        self.ncm.replace_class_means_dict(class_means)

        real_classes = [k for k in class_means.keys() if k < self.base_id]
        print(f" Updated NCM with {len(real_classes)} real classes (no fake contamination)")

        self.model.train()

    @torch.no_grad()
    def _analyze_cosine_distribution_epoch(self):
        """ 에포크마다 코사인 유사도 분포 분석"""
        if len(self.memory_buffer) < 20:  # 최소 샘플 수 확인
            return

        # 현재 모델 모드 저장
        was_training = self.model.training
        self.model.eval()

        # 메모리 버퍼에서 모든 데이터 가져오기 (실제 사용자만)
        all_paths, all_labels = self.memory_buffer.get_all_data()

        # 가짜 클래스 필터링
        real_paths = []
        real_labels = []
        for path, label in zip(all_paths, all_labels):
            label_int = int(label) if not isinstance(label, int) else label
            if label_int < self.base_id:
                real_paths.append(path)
                real_labels.append(label_int)

        if len(real_paths) < 10:
            # 모드 복원 후 리턴
            if was_training:
                self.model.train()
            return

        # 샘플링 (너무 많으면 일부만)
        if len(real_paths) > 200:
            indices = np.random.choice(len(real_paths), 200, replace=False)
            real_paths = [real_paths[i] for i in indices]
            real_labels = [real_labels[i] for i in indices]

        # [TARGET] 기존 방식과 동일한 NCM 기반 스코어 계산
        genuine_scores = []
        impostor_scores = []

        # 클래스별로 데이터 분류
        from collections import defaultdict
        by_class = defaultdict(list)
        for path, label in zip(real_paths, real_labels):
            by_class[int(label)].append(path)

        # Genuine 스코어 계산 (같은 클래스 내)
        for cls_id, cls_paths in by_class.items():
            if len(cls_paths) < 2:
                continue

            # 클래스 내에서 샘플링
            sample_paths = cls_paths[:min(5, len(cls_paths))]

            for path in sample_paths:
                # 특징 추출
                img = _open_with_channels(path, self.config.dataset.channels)
                img_tensor = self.test_transform(img).unsqueeze(0).to(self.device)

                # NCM 점수 계산 (6144D 특징 사용)
                feat = self.model.getFeatureCode(img_tensor)
                ncm_scores = self.ncm.forward(feat)

                if ncm_scores.numel() > 0:
                    #  수정: 클래스 ID를 직접 인덱스로 사용
                    if cls_id in self.ncm.class_means_dict and cls_id < ncm_scores.shape[1]:
                        genuine_score = ncm_scores[0, cls_id].item()
                        genuine_scores.append(genuine_score)

        # Impostor 스코어 계산 (다른 클래스들)
        max_impostor_samples = 50
        impostor_count = 0

        for cls_id, cls_paths in by_class.items():
            if impostor_count >= max_impostor_samples:
                break

            # 클래스당 최대 3개 샘플
            sample_paths = cls_paths[:min(3, len(cls_paths))]

            for path in sample_paths:
                if impostor_count >= max_impostor_samples:
                    break

                # 특징 추출
                img = _open_with_channels(path, self.config.dataset.channels)
                img_tensor = self.test_transform(img).unsqueeze(0).to(self.device)

                # NCM 점수 계산
                feat = self.model.getFeatureCode(img_tensor)
                ncm_scores = self.ncm.forward(feat)

                if ncm_scores.numel() > 0:
                    # 유효한 클래스만으로 Impostor 점수 계산
                    valid_ids = sorted(self.ncm.class_means_dict.keys())
                    if len(valid_ids) > 1:  # 최소 2개 클래스 필요
                        scores_valid = ncm_scores[:, valid_ids].clone()
                        own_col = valid_ids.index(cls_id) if cls_id in valid_ids else None
                        if own_col is not None:
                            scores_valid[0, own_col] = -1e9

                        max_score = scores_valid.max(dim=1).values.item()
                        if max_score > -1e9:
                            impostor_scores.append(max_score)
                            impostor_count += 1

        #  디버깅: NCM과 메모리 버퍼 동기화 확인
        print(f"\n NCM Classes: {sorted(list(self.ncm.class_means_dict.keys()))}")
        print(f" Memory Classes: {sorted(list(set(real_labels)))}")
        print(f" Sample count - Genuine: {len(genuine_scores)}, Impostor: {len(impostor_scores)}")

        # 통계 계산 및 출력
        if genuine_scores and impostor_scores:
            genuine_mean = np.mean(genuine_scores)
            genuine_std = np.std(genuine_scores)
            impostor_mean = np.mean(impostor_scores)
            impostor_std = np.std(impostor_scores)
            separation = genuine_mean - impostor_mean

            print(f"     NCM Score Distribution (like EER Calculation):")
            print(f"       Genuine:  {genuine_mean:.3f} ± {genuine_std:.3f} (n={len(genuine_scores)})")
            print(f"       Impostor: {impostor_mean:.3f} ± {impostor_std:.3f} (n={len(impostor_scores)})")
            print(f"       Separation: {separation:.3f}")

            # NCM 스코어 기준 성능 평가 (코사인 유사도와 다른 기준)
            if separation > 0.3:
                print(f"       Status:  Excellent separation")
            elif separation > 0.2:
                print(f"       Status:  Good separation")
            elif separation > 0.1:
                print(f"       Status:  Moderate separation")
            else:
                print(f"       Status:  Poor separation")

        # 원래 모델 모드 복원
        if was_training:
            self.model.train()
        else:
            self.model.eval()

    def evaluate(self, test_dataset: Dataset) -> float:
        """NCM을 사용하여 정확도를 평가합니다."""
        self.model.eval()

        dataloader = DataLoader(
            test_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            worker_init_fn=worker_init_fn if self.config.training.num_workers > 0 else None,
            pin_memory=True,
            persistent_workers=False
        )

        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels in dataloader:
                data = data.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                features = self.model.getFeatureCode(data)
                predictions = self.ncm.predict(features)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = 100.0 * correct / total
        return accuracy

    def save_checkpoint(self, path: str):
        """[CORE] 안전한 체크포인트 저장 (디렉토리 생성 추가)"""
        # 디렉토리 생성
        save_dir = os.path.dirname(path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        checkpoint_dict = {
            'model_state_dict': self.model.state_dict(),
            'ncm_state_dict': self.ncm.state_dict(),
            'experience_count': self.experience_count,
            'memory_buffer_size': len(self.memory_buffer)
        }

        # 옵티마이저 상태 안전 저장
        try:
            checkpoint_dict['optimizer_state_dict'] = self.optimizer.state_dict()
            checkpoint_dict['scheduler_state_dict'] = self.scheduler.state_dict()
        except Exception as e:
            print(f"Warning: Could not save optimizer/scheduler state: {e}")

        # ProxyAnchorLoss 관련 저장
        if self.use_proxy_anchor and self.proxy_anchor_loss.proxies is not None:
            try:
                checkpoint_dict['proxy_anchor_data'] = {
                    'proxies': self.proxy_anchor_loss.proxies.detach().cpu(),
                    'class_to_idx': self.proxy_anchor_loss.class_to_idx.copy(),
                    'num_classes': self.proxy_anchor_loss.num_classes
                }
            except Exception as e:
                print(f"Warning: Could not save proxy anchor data: {e}")

        # 오픈셋 관련 추가 저장
        if self.openset_enabled:
            try:
                checkpoint_dict['openset_data'] = {
                    'tau_s': self.ncm.tau_s,
                    'registered_users': list(self.registered_users),
                    'evaluation_history': self.evaluation_history
                }
            except Exception as e:
                print(f"Warning: Could not save openset data: {e}")

        torch.save(checkpoint_dict, path)
        print(f"[OK] Checkpoint saved to: {path}")
