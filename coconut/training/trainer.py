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

        # verbose 설정
        self.verbose = getattr(config.training, 'verbose', False)

        # 시드 설정
        seed = getattr(config.training, 'seed', 42) if hasattr(config, 'training') else 42
        set_seed(seed)
        if self.verbose:
            print(f" Random seed set to {seed}")

        # 모델/NCM 디바이스 이동
        self.device = device
        model = model.to(device)
        if hasattr(ncm_classifier, 'to'):
            ncm_classifier = ncm_classifier.to(device)

        # 사전훈련 로딩 (한 번만 실행)
        if hasattr(config.model, 'use_pretrained') and config.model.use_pretrained:
            if config.model.pretrained_path and config.model.pretrained_path.exists():
                if self.verbose:
                    print(f"\n🎯 Loading pretrained weights...")
                    print(f"   Path: {config.model.pretrained_path}")
                loader = PretrainedLoader()
                try:
                    model = loader.load_ccnet_pretrained(
                        model=model,
                        checkpoint_path=config.model.pretrained_path,
                        device=device,
                        verbose=self.verbose
                    )
                    if self.verbose:
                        print("✅ Pretrained weights loaded successfully!")
                except Exception as e:
                    print(f"⚠️ WARNING: Failed to load pretrained: {e}")
                    print("   Continuing with current weights...")
            else:
                print(f"⚠️ WARNING: Pretrained path not found: {config.model.pretrained_path}")
        else:
            if self.verbose:
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
                embedding_dim = 6144  # projection 미사용 시 forward()가 반환하는 6144D 특징

            self.proxy_anchor_loss = ProxyAnchorLoss(
                embedding_size=embedding_dim,
                margin=getattr(config.training, 'proxy_margin', 0.1),
                alpha=getattr(config.training, 'proxy_alpha', 32)
            ).to(device)

            if self.verbose:
                print(f"[COCONUT] COCONUT ProxyAnchor initialized with {embedding_dim}D embeddings")

            self.proxy_lambda = getattr(config.training, 'proxy_lambda', 0.3)

            if self.verbose:
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
                    if self.verbose:
                        print(f"[TARGET] TTA enabled for evaluation:")
                        print(f"   Views: {self.openset_config.tta_n_views}")
                        print(f"   Include original: {self.openset_config.tta_include_original}")
                        print(f"   Augmentation: {self.openset_config.tta_augmentation_strength}")
                        print(f"   Aggregation: {self.openset_config.tta_aggregation}")
                        print(f"   Type-specific repeats:")
                        print(f"     - Genuine: {self.openset_config.tta_n_repeats_genuine}")
                        print(f"     - Between: {self.openset_config.tta_n_repeats_between}")
                else:
                    print("WARNING: TTA requested but functions not available")
                    self.use_tta = False

            if self.verbose:
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
                verbose=self.verbose and config.openset.verbose_calibration
            )

            # Dev/Train 데이터 관리
            self.probe_data = {}
            self.train_data = {}
            self.registered_users = set()

            # 평가 히스토리
            self.evaluation_history = []

            # DET curve용 마지막 스코어 캐시
            self._last_genuine_scores = np.array([])
            self._last_impostor_scores = np.array([])

            # 초기 임계치 설정
            initial_tau = config.openset.initial_tau
            if hasattr(self.ncm, 'set_thresholds'):
                self.ncm.set_thresholds(tau_s=initial_tau)
            else:
                self.ncm.tau_s = initial_tau

            if self.verbose:
                print(f"[TARGET] FAR Target mode enabled")
                print(f"   Target FAR: {config.openset.target_far*100:.1f}%")
                print(f"   Initial τ_s: {initial_tau}")
        else:
            self.registered_users = set()
            self.use_tta = False
            if self.verbose:
                print(" Open-set mode disabled")

        # Statistics
        self.experience_count = 0

        # CuDNN 최적화
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            if self.verbose:
                print("[INIT] CuDNN benchmark enabled (speed priority)")

        if self.verbose:
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
            if self.verbose:
                print(f"️ Backbone LR: {self.config.training.learning_rate:.6f}")

        # 프로젝션 헤드 파라미터 그룹 (새로 초기화된 레이어)
        if projection_params and self.config.model.use_projection:
            param_groups.append({
                'params': projection_params,
                'lr': self.config.training.projection_learning_rate,
                'name': 'projection'
            })
            if self.verbose:
                print(f" Projection Head LR: {self.config.training.projection_learning_rate:.6f}")

        # 프록시 파라미터 그룹 (새로 초기화된 프록시)
        if include_proxies and self.use_proxy_anchor and hasattr(self, 'proxy_anchor_loss') and self.proxy_anchor_loss.proxies is not None:
            param_groups.append({
                'params': [self.proxy_anchor_loss.proxies],
                'lr': self.base_lr * self.proxy_lr_ratio,
                'name': 'proxies'
            })
            if self.verbose:
                print(f"[COCONUT] Proxies LR: {self.base_lr * self.proxy_lr_ratio:.6f} ({self.proxy_lr_ratio}x)")

        return optim.Adam(param_groups)

    def _recreate_optimizer_with_proxies(self, old_proxy_state=None, n_old=0):
        """프록시 추가 시 옵티마이저 재생성 + 기존 state 보존"""
        if not self.use_proxy_anchor or self.proxy_anchor_loss.proxies is None:
            return

        current_num_proxies = self.proxy_anchor_loss.num_classes

        if current_num_proxies != self.last_num_proxies:
            # 백본/프로젝션 헤드 state 저장 (파라미터 객체가 동일하므로 복원 가능)
            old_model_states = {}
            for param in self.model.parameters():
                if param in self.optimizer.state and self.optimizer.state[param]:
                    old_model_states[param] = {
                        k: v.clone() if torch.is_tensor(v) else v
                        for k, v in self.optimizer.state[param].items()
                    }

            # 스케줄러 step 위치 저장
            old_scheduler_state = self.scheduler.state_dict() if hasattr(self, 'scheduler') else None

            # 새 옵티마이저 생성
            self.optimizer = self._create_optimizer_with_grouped_params(include_proxies=True)

            # 백본/프로젝션 state 복원
            for param, state in old_model_states.items():
                self.optimizer.state[param] = state

        # 프록시 state 이전: 기존 N개 복원 + 새 슬롯은 0으로 초기화
        if old_proxy_state and n_old > 0:
            # 차원이 바뀌었으면 기존 state는 폐기하고 새로 초기화한다
            old_dim = old_proxy_state.get('exp_avg', torch.empty(0)).shape[1] if 'exp_avg' in old_proxy_state else None
            D = self.proxy_anchor_loss.embedding_size
            if old_dim is not None and old_dim != D:
                old_proxy_state = {}

        if n_old > 0 and old_proxy_state:
            n_new = current_num_proxies - n_old
            D = self.proxy_anchor_loss.embedding_size
            device = self.proxy_anchor_loss.proxies.device

            new_proxy_state = {}
            if 'exp_avg' in old_proxy_state:
                new_proxy_state['exp_avg'] = torch.cat([
                    old_proxy_state['exp_avg'].to(device),
                    torch.zeros(n_new, D, device=device)
                ], dim=0)
            if 'exp_avg_sq' in old_proxy_state:
                new_proxy_state['exp_avg_sq'] = torch.cat([
                    old_proxy_state['exp_avg_sq'].to(device),
                    torch.zeros(n_new, D, device=device)
                ], dim=0)
            if 'step' in old_proxy_state:
                new_proxy_state['step'] = old_proxy_state['step']

            self.optimizer.state[self.proxy_anchor_loss.proxies] = new_proxy_state

            # 스케줄러 재생성 + step 위치 복원
            self.scheduler = lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.scheduler_step_size,
                gamma=self.config.training.scheduler_gamma
            )
            if old_scheduler_state is not None:
                self.scheduler.load_state_dict(old_scheduler_state)

            self.last_num_proxies = current_num_proxies
            if self.verbose:
                print(f" Optimizer recreated: {current_num_proxies} proxies "
                      f"(preserved {n_old} existing, fresh {current_num_proxies - n_old} new)")

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
            # 🍑 config에 따라 projection 사용
            use_projection = getattr(self.config.model, 'use_projection_for_ncm', False)
            return self.model.getFeatureCode(batch_images, use_projection=use_projection)

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

                # Extract features using getFeatureCode for NCM compatibility
                view = view.unsqueeze(0)  # Add batch dimension
                # 🍑 config에 따라 projection 사용
                use_projection = getattr(self.config.model, 'use_projection_for_ncm', False)
                features = self.model.getFeatureCode(view, use_projection=use_projection)  # 6144D 또는 512D

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

        if self.verbose:
            print(f"\n=== Training Experience {self.experience_count}: User {user_id} ===")

        # 프록시 추가 및 옵티마이저 재생성 (기존 state 보존)
        if self.use_proxy_anchor:
            unique_labels = set(labels)
            real_classes = list(unique_labels)

            if real_classes:
                # 새 클래스 피처 평균 계산 — proxy 초기화용 (512D projection 공간)
                new_class_ids = [c for c in real_classes
                                 if c not in self.proxy_anchor_loss.class_to_idx]
                feature_means = {}
                if new_class_ids:
                    self.model.eval()
                    with torch.no_grad():
                        for cid in new_class_ids:
                            cls_paths = [p for p, l in zip(image_paths, labels) if l == cid]
                            feats = []
                            for p in cls_paths:
                                img = _open_with_channels(p, self.config.dataset.channels)
                                img = self.test_transform(img).unsqueeze(0).to(self.device)
                                feat = self.model.getFeatureCode(img, use_projection=True)
                                feats.append(feat.squeeze(0))
                            feature_means[cid] = torch.stack(feats).mean(dim=0)
                    self.model.train()

                # add_classes() 전에 현재 프록시 참조와 optimizer state 저장
                old_proxy_param = self.proxy_anchor_loss.proxies
                n_old = old_proxy_param.shape[0] if old_proxy_param is not None else 0
                old_proxy_state = {}
                if old_proxy_param is not None and old_proxy_param in self.optimizer.state:
                    old_proxy_state = {
                        k: v.clone() if torch.is_tensor(v) else v
                        for k, v in self.optimizer.state[old_proxy_param].items()
                    }

                self.proxy_anchor_loss.add_classes(real_classes, feature_means=feature_means, verbose=self.verbose)
                self._recreate_optimizer_with_proxies(
                    old_proxy_state=old_proxy_state,
                    n_old=n_old
                )

        # 원본 labels를 보존
        original_labels = labels.copy()

        # Step 1: Dev/Train 분리 (오픈셋 모드인 경우)
        if self.openset_enabled:
            train_paths, train_labels, dev_paths, dev_labels = split_user_data(
                image_paths, original_labels,
                dev_ratio=self.openset_config.dev_ratio,
                min_dev=1 if len(image_paths) <= 5 else 2
            )

            # 저장
            self.probe_data[user_id] = (dev_paths, dev_labels)
            self.train_data[user_id] = (train_paths, train_labels)

            if self.verbose:
                print(f"[INFO] Data split: Train={len(train_paths)}, Probe={len(dev_paths)}")
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
                    # Clipping: 실제 저장 수를 초과하지 않도록 제한 (초반 과적합 방지)
                    effective_memory_batch = min(
                        self.config.training.memory_batch_size,
                        len(self.memory_buffer)
                    )
                    memory_paths, memory_labels = self.memory_buffer.sample(
                        effective_memory_batch
                    )

                    if torch.is_tensor(memory_labels):
                        memory_labels = memory_labels.cpu().tolist()

                    if memory_paths:
                        #  메모리 데이터를 effective_memory_batch만큼 증강 (clipping 적용)
                        augmented_memory_paths, augmented_memory_labels = repeat_and_augment_data(
                            memory_paths, memory_labels, effective_memory_batch
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

                    # Curriculum Loss Schedule + Batch Gating
                    ramp_users = getattr(self.config.training, 'curriculum_ramp_users', 12)
                    num_users = len(self.registered_users)
                    w_supcon = min(0.5, num_users / ramp_users)
                    w_proxy = 1.0 - w_supcon

                    # Batch Gating: 배치 내 클래스 < 2이면 SupCon 비활성화
                    unique_in_batch = len(set(batch_labels.tolist()))
                    if unique_in_batch < 2:
                        w_supcon = 0.0
                        w_proxy = 1.0

                    if w_supcon > 0:
                        loss_supcon = self.criterion(features_paired, batch_labels)
                    else:
                        loss_supcon = torch.tensor(0.0, device=self.device)

                    # ProxyAnchorLoss
                    if self.use_proxy_anchor and self.proxy_anchor_loss.proxies is not None:
                        all_labels = batch_labels.repeat(2)
                        loss_proxy = self.proxy_anchor_loss(features_all, all_labels)

                        # proxy_lambda는 ProxyAnchor 비중을 조절하는 명시적 가중치
                        proxy_weight = w_proxy * self.proxy_lambda
                        supcon_weight = w_supcon
                        loss = proxy_weight * loss_proxy + supcon_weight * loss_supcon

                        if iteration == 0 and epoch == 0:
                            # compact 모드용: curriculum weights 저장
                            self._last_curriculum = {
                                'w_proxy': proxy_weight, 'w_supcon': supcon_weight,
                                'loss_supcon': loss_supcon.item(), 'loss_proxy': loss_proxy.item()
                            }
                            if self.verbose:
                                print(f"[Curriculum] users={num_users}, w_proxy={w_proxy:.2f}, w_supcon={w_supcon:.2f}, gate={'ON' if unique_in_batch >= 2 else 'OFF'}")
                                print(f"   Weights → proxy:{proxy_weight:.2f}, supcon:{supcon_weight:.2f}")
                                print(f"   SupCon: {loss_supcon.item():.4f}, ProxyAnchor: {loss_proxy.item():.4f}")
                    else:
                        loss = loss_supcon

                    loss_avg.update(loss.item(), batch_size)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    if self.use_proxy_anchor and self.proxy_anchor_loss.proxies is not None:
                        torch.nn.utils.clip_grad_norm_(self.proxy_anchor_loss.proxies, max_norm=1.0)

                    self.optimizer.step()

                    epoch_loss += loss.item()

            avg_loss = epoch_loss / self.config.training.iterations_per_epoch
            if epoch == 0:
                self._first_epoch_loss = avg_loss
            self._last_epoch_loss = avg_loss

            if self.verbose:
                print(f"  Epoch [{epoch+1}/{self.config.training.epochs_per_experience}] Loss: {avg_loss:.4f}")

        # 메모리 버퍼 업데이트
        self.memory_buffer.update_from_dataset(train_paths, train_labels)
        if self.verbose:
            print(f"Memory buffer size after update: {len(self.memory_buffer)}")

        # NCM 업데이트
        self._update_ncm()

        # NCM 갱신 후 코사인 유사도 분포 분석 (현재 백본 + 현재 프로토타입 기준으로 정확한 측정)
        self._analyze_cosine_distribution_epoch()

        # 디버깅: NCM과 버퍼 동기화 확인
        all_paths, all_labels = self.memory_buffer.get_all_data()
        buffer_classes = set(int(label) for label in all_labels)
        ncm_classes = set(self.ncm.class_means_dict.keys())
        missing = buffer_classes - ncm_classes

        if missing:
            print(f"  [WARN] NCM missing classes: {sorted(list(missing))}")
        elif self.verbose:
            print(f"[OK] NCM synchronized: {len(ncm_classes)} classes")

        # 주기적 캘리브레이션 및 평가 (오픈셋 모드)
        if self.openset_enabled and len(self.registered_users) % self.config.training.test_interval == 0:

            if len(self.registered_users) >= self.openset_config.warmup_users:
                if self.verbose:
                    print("\n" + "="*60)
                    print(" THRESHOLD CALIBRATION & EVALUATION")
                    print("="*60)

                self._calibrate_threshold()
                metrics = self._evaluate_openset()

                self.evaluation_history.append({
                    'experience': self.experience_count,
                    'num_users': len(self.registered_users),
                    'tau_s': self.ncm.tau_s,
                    'metrics': metrics
                })

                if self.verbose:
                    print("="*60 + "\n")

                # Compact 출력: 한국어 설명 포함 핵심 지표
                if not self.verbose:
                    n_train = len(self.train_data.get(user_id, ([], []))[0]) if self.openset_enabled else len(image_paths)
                    n_probe = len(self.probe_data.get(user_id, ([], []))[0]) if self.openset_enabled else 0
                    n_proxies = self.proxy_anchor_loss.num_classes if self.use_proxy_anchor else 0

                    # Line 1: Experience header
                    print(f"\n[Exp {self.experience_count:03d}] User {user_id} | Train={n_train}, Probe={n_probe} | Proxies: {n_proxies}")

                    # Line 2: Loss + curriculum
                    curriculum = getattr(self, '_last_curriculum', {})
                    w_proxy = curriculum.get('w_proxy', 0)
                    w_supcon = curriculum.get('w_supcon', 0)
                    first_loss = getattr(self, '_first_epoch_loss', 0)
                    last_loss = getattr(self, '_last_epoch_loss', 0)
                    epochs = self.config.training.epochs_per_experience
                    print(f"  Loss: {first_loss:.4f} -> {last_loss:.4f} ({epochs}ep) | w_proxy={w_proxy:.2f}, w_supcon={w_supcon:.2f}")

                    # Line 3: Threshold + open-set metrics (한국어 설명)
                    tau = self.ncm.tau_s
                    achieved_fpir = metrics.get('FPIR_in', 0)
                    rank1 = metrics.get('Rank1', 0)
                    fpir_in = metrics.get('FPIR_in', 0)
                    trr = metrics.get('TRR_unknown', 0)
                    print(f"  tau={tau:.4f} (FPIR={achieved_fpir*100:.1f}%) | "
                          f"Rank-1={rank1:.3f} (최고유사도 사용자가 정답인 비율) | "
                          f"FPIR_in={fpir_in:.3f} (미등록자를 등록자로 잘못 수락한 비율) | "
                          f"TRR={trr:.3f} (미등록자를 정상 거부한 비율)")

            else:
                print(f"  [WARN] Warmup phase: {len(self.registered_users)}/{self.openset_config.warmup_users}")

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

        if self.verbose:
            mode_str = f"MAX + TTA(views={self.openset_config.tta_n_views})" if use_tta else "MAX"
            print(f" Using {mode_str} scores for calibration")

            if use_tta:
                print(f" TTA with type-specific repeats:")
                print(f"   Genuine: {self.openset_config.tta_n_views} views × {n_repeats_genuine} repeats")
                print(f"   Between: {self.openset_config.tta_n_views} views × {n_repeats_between} repeats")
                print(f"   NegRef: {self.openset_config.tta_n_views} views × {n_repeats_negref} repeats")

            print(f"\nExtracting Unknown Dev scores for FPIR calibration...")

        unknown_dev_file = getattr(self.config.dataset, 'unknown_dev_file', None)
        s_impostor = np.array([])
        use_projection = getattr(self.config.model, 'use_projection_for_ncm', False)

        if unknown_dev_file and str(unknown_dev_file) != 'None':
            s_impostor = extract_scores_impostor_unknown(
                self.model, self.ncm,
                str(unknown_dev_file),
                self.registered_users,
                self.test_transform, self.device,
                max_eval=3000,
                channels=channels,
                use_projection=use_projection
            )
            if self.verbose:
                print(f"   Unknown Dev: {len(s_impostor)} scores from {unknown_dev_file}")
        else:
            print("  [WARN] unknown_dev_file not set. τ calibration skipped.")

        # probe_data에서 genuine 점수 추출 (참고용)
        all_probe_paths = []
        all_probe_labels = []
        for uid, (paths, labels) in self.probe_data.items():
            all_probe_paths.extend(paths)
            all_probe_labels.extend([uid] * len(paths))

        s_genuine = extract_scores_genuine(
            self.model, self.ncm,
            all_probe_paths, all_probe_labels,
            self.test_transform, self.device,
            channels=channels,
            use_projection=use_projection
        )

        if self.verbose:
            print(f"   Genuine (probe, 참고용): {len(s_genuine)} scores")
            print(f"   Impostor (unknown_dev): {len(s_impostor)} scores")

        # DET curve용 스코어 캐시
        self._last_genuine_scores = s_genuine.copy() if len(s_genuine) > 0 else np.array([])
        self._last_impostor_scores = s_impostor.copy() if len(s_impostor) > 0 else np.array([])

        if len(s_impostor) >= 10:
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

            if self.verbose:
                print(f"FPIR Target Results:")
                print(f"   Target FPIR: {self.openset_config.target_far*100:.1f}%")
                print(f"   Achieved FPIR: {result.get('current_far', 0)*100:.1f}%")
                print(f"   Threshold τ: {result['tau_smoothed']:.4f}")
        else:
            print("  [WARN] Not enough unknown_dev samples for calibration")

    @torch.no_grad()
    def _evaluate_openset(self):
        """오픈셋 평가 (TTA 지원)"""

        #  재현성: 평가 시 시드 재설정
        eval_seed = getattr(self.config.training, 'seed', 42)
        set_seed(eval_seed)

        use_tta = self.use_tta and TTA_FUNCTIONS_AVAILABLE
        img_size = self.config.dataset.height
        channels = self.config.dataset.channels
        use_projection = getattr(self.config.model, 'use_projection_for_ncm', False)

        if self.verbose:
            if use_tta:
                print(f"\n Open-set Evaluation with TTA (n={self.openset_config.tta_n_views}):")
            else:
                print("\n Open-set Evaluation (Single View):")

        # 변수 초기화
        TAR = FRR = 0.0
        MisID_rate = FNIR = 0.0
        TRR_u = FAR_u = 0.0
        TRR_n = FAR_n = None

        # 1. Known-Probe (TAR/FRR)
        all_dev_paths = []
        all_dev_labels = []
        for uid, (paths, labels) in self.probe_data.items():
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
                    seed=eval_seed + 5000,  #  재현성: seed 전달
                    use_projection=use_projection
                )

                # TTA 통계 출력
                if details:
                    reject_reasons = [d.get('reject_reason') for d in details if d.get('reject_reason')]
                    if reject_reasons and self.verbose:
                        from collections import Counter
                        reason_counts = Counter(reject_reasons)
                        print(f"   TTA Reject reasons: gate={reason_counts.get('gate', 0)}, "
                              f"class={reason_counts.get('class', 0)}, both={reason_counts.get('both', 0)}")
            else:
                preds = predict_batch(
                    self.model, self.ncm,
                    all_dev_paths, self.test_transform, self.device,
                    channels=channels,
                    use_projection=use_projection
                )

            n = max(1, len(preds))
            correct  = sum(1 for p, l in zip(preds, all_dev_labels) if p == l)
            rejected = sum(1 for p in preds if p == -1)
            misid    = sum(1 for p, l in zip(preds, all_dev_labels) if p != l and p != -1)

            TAR          = correct  / n
            FRR          = rejected / n
            MisID_rate   = misid    / n
            FNIR         = 1.0 - TAR

        # 2. Unknown FPIR_in — unknown_test_file 고정 파일 로드 (매 experience 동일한 pool)
        unknown_test_file = getattr(self.config.dataset, 'unknown_test_file', None)

        if unknown_test_file and str(unknown_test_file) != 'None':
            # unknown_test_file: enroll_file 미등록 ID 뒤 50% 권장
            unknown_filtered_paths, _ = load_paths_labels_from_txt(str(unknown_test_file))
            unknown_filtered = unknown_filtered_paths
            if self.verbose:
                print(f"   Unknown Test: {len(unknown_filtered)} samples from {unknown_test_file}")
        else:
            print("  [WARN] unknown_test_file not set. Using enroll_file fallback (non-fixed pool).")
            unknown_paths, unknown_labels = load_paths_labels_from_txt(
                str(self.config.dataset.enroll_file)
            )
            unknown_filtered = [p for p, l in zip(unknown_paths, unknown_labels)
                                if l not in self.registered_users]

        if len(unknown_filtered) > 1000:
            np.random.seed(eval_seed + 1000)
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
                    seed=eval_seed + 3000,  #  재현성: seed 전달
                    use_projection=use_projection
                )
            else:
                preds_unk = predict_batch(
                    self.model, self.ncm,
                    unknown_filtered, self.test_transform, self.device,
                    channels=channels,
                    use_projection=use_projection
                )
            TRR_u = sum(1 for p in preds_unk if p == -1) / len(preds_unk)
            FAR_u = 1 - TRR_u

        # 3. NegRef FPIR_xdom — xdomain_file (크로스도메인 평가 전용, 학습 사용 안 함)
        _negref_source = str(self.config.dataset.xdomain_file)
        negref_paths, _ = load_paths_labels_from_txt(_negref_source)

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
                    seed=eval_seed + 4000,  #  재현성: seed 전달
                    use_projection=use_projection
                )
            else:
                preds_neg = predict_batch(
                    self.model, self.ncm,
                    negref_paths, self.test_transform, self.device,
                    channels=channels,
                    use_projection=use_projection
                )
            TRR_n = sum(1 for p in preds_neg if p == -1) / len(preds_neg)
            FAR_n = 1 - TRR_n

        # 결과 출력
        if self.verbose:
            print(f"   Known-Probe: Rank-1={TAR:.3f}, FNIR={FNIR:.3f} (FRR={FRR:.3f}, MisID={MisID_rate:.3f})")
            print(f"   Unknown: TRR={TRR_u:.3f}, FPIR_in={FAR_u:.3f}")
            if TRR_n is not None:
                print(f"   NegRef: TRR={TRR_n:.3f}, FPIR_xdom={FAR_n:.3f}")
            print(f"   Threshold: τ_s={self.ncm.tau_s:.4f}")
            print(f"   Mode: FPIR Target ({self.openset_config.target_far*100:.1f}%)")
            print(f"   Score: Max")
            if use_tta:
                print(f"   [TARGET] TTA: {self.openset_config.tta_n_views} views, agree_k={self.openset_config.tta_agree_k}")

        return {
            'Rank1': TAR, 'FNIR': FNIR,
            'FRR': FRR, 'MisID': MisID_rate,
            'TRR_unknown': TRR_u, 'FPIR_in': FAR_u,
            'TRR_negref': TRR_n, 'FPIR_xdom': FAR_n,
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
        real_paths = all_paths
        real_labels = all_labels


        if not real_paths:
            print("WARNING: No real users for NCM update (NCM remains empty)")
            return

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

            # 🍑 config에 따라 6144D 또는 512D 사용
            use_projection = getattr(self.config.model, 'use_projection_for_ncm', False)
            features = self.model.getFeatureCode(data, use_projection=use_projection)

            for i, label in enumerate(labels):
                label_item = label.item()
                if label_item not in class_features:
                    class_features[label_item] = []
                class_features[label_item].append(features[i].cpu())

        # 클래스별 평균 계산 (정규화는 replace_class_means_dict 내부에서 처리)
        class_means = {}
        for label, features_list in class_features.items():
            if len(features_list) > 0:
                mean_feature = torch.stack(features_list).mean(dim=0)
                class_means[label] = mean_feature

        # NCM 업데이트
        self.ncm.replace_class_means_dict(class_means)

        if self.verbose:
            print(f" Updated NCM with {len(class_means)} classes")

        self.model.train()

    @torch.no_grad()
    def _analyze_cosine_distribution_epoch(self):
        """ 에포크마다 코사인 유사도 분포 분석"""
        if len(self.memory_buffer) < 20:  # 최소 샘플 수 확인
            return

        # 현재 모델 모드 저장
        was_training = self.model.training
        self.model.eval()

        # 메모리 버퍼에서 모든 데이터 가져오기
        all_paths, all_labels = self.memory_buffer.get_all_data()
        real_paths = all_paths
        real_labels = [int(l) if not isinstance(l, int) else l for l in all_labels]

        if len(real_paths) < 10:
            # 모드 복원 후 리턴
            if was_training:
                self.model.train()
            return

        # 샘플링 (너무 많으면 일부만) — seed 고정으로 재현성 보장
        if len(real_paths) > 200:
            monitor_seed = getattr(self.config.training, 'seed', 42) + self.experience_count
            rng = np.random.RandomState(monitor_seed)
            indices = rng.choice(len(real_paths), 200, replace=False)
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

                # NCM 점수 계산 (config에 따라 6144D 또는 512D 특징 사용)
                # 🍑 use_projection_for_ncm 옵션 추가
                use_projection = getattr(self.config.model, 'use_projection_for_ncm', False)
                feat = self.model.getFeatureCode(img_tensor, use_projection=use_projection)
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
                # 🍑 use_projection_for_ncm 옵션 추가
                use_projection = getattr(self.config.model, 'use_projection_for_ncm', False)
                feat = self.model.getFeatureCode(img_tensor, use_projection=use_projection)
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

        if self.verbose:
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

            if self.verbose:
                print(f"     NCM Score Distribution (like EER Calculation):")
                print(f"       Genuine:  {genuine_mean:.3f} ± {genuine_std:.3f} (n={len(genuine_scores)})")
                print(f"       Impostor: {impostor_mean:.3f} ± {impostor_std:.3f} (n={len(impostor_scores)})")
                print(f"       Separation: {separation:.3f}")

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

                # 🍑 use_projection_for_ncm 옵션 추가
                use_projection = getattr(self.config.model, 'use_projection_for_ncm', False)
                features = self.model.getFeatureCode(data, use_projection=use_projection)
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

        # memory_buffer 전체 데이터 저장 (학습 재개 시 복원용)
        try:
            buf_paths, buf_labels = self.memory_buffer.get_all_data()
            checkpoint_dict['memory_buffer_data'] = {
                'paths': buf_paths,
                'labels': [int(l) for l in buf_labels]
            }
        except Exception as e:
            print(f"Warning: Could not save memory buffer data: {e}")

        # probe_data 저장 (평가 재개 시 복원용)
        if self.openset_enabled and self.probe_data:
            try:
                checkpoint_dict['probe_data'] = {
                    uid: (paths, [int(l) for l in labels])
                    for uid, (paths, labels) in self.probe_data.items()
                }
            except Exception as e:
                print(f"Warning: Could not save probe data: {e}")

        torch.save(checkpoint_dict, path)
        if self.verbose:
            print(f"[OK] Checkpoint saved to: {path}")

    def load_checkpoint(self, path: str):
        """체크포인트 로드 및 trainer 상태 복원 (학습 재개용)"""
        checkpoint = torch.load(path, map_location=self.device)

        # 모델 가중치 복원
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.verbose:
            print(f"[OK] Model weights restored")

        # NCM 상태 복원
        self.ncm.load_state_dict(checkpoint['ncm_state_dict'])
        if self.verbose:
            print(f"[OK] NCM state restored ({self.ncm.get_num_classes()} classes)")

        # experience 카운터 복원
        self.experience_count = checkpoint.get('experience_count', 0)

        # 옵티마이저/스케줄러 복원
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if self.verbose:
                    print(f"[OK] Optimizer state restored")
            except Exception as e:
                print(f"Warning: Could not restore optimizer state: {e}")

        if 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if self.verbose:
                    print(f"[OK] Scheduler state restored")
            except Exception as e:
                print(f"Warning: Could not restore scheduler state: {e}")

        # ProxyAnchorLoss 복원
        if 'proxy_anchor_data' in checkpoint and self.use_proxy_anchor:
            pa_data = checkpoint['proxy_anchor_data']
            self.proxy_anchor_loss.proxies = nn.Parameter(
                pa_data['proxies'].to(self.device)
            )
            self.proxy_anchor_loss.class_to_idx = pa_data['class_to_idx']
            self.proxy_anchor_loss.num_classes = pa_data['num_classes']
            self.last_num_proxies = pa_data['num_classes']
            self._recreate_optimizer_with_proxies()
            if self.verbose:
                print(f"[OK] ProxyAnchor restored ({pa_data['num_classes']} proxies)")

        # 오픈셋 데이터 복원
        if 'openset_data' in checkpoint and self.openset_enabled:
            od = checkpoint['openset_data']
            self.registered_users = set(od.get('registered_users', []))
            self.evaluation_history = od.get('evaluation_history', [])
            tau_s = od.get('tau_s')
            if tau_s is not None:
                if hasattr(self.ncm, 'set_thresholds'):
                    self.ncm.set_thresholds(tau_s=tau_s)
                else:
                    self.ncm.tau_s = tau_s
            if self.verbose:
                print(f"[OK] Openset data restored ({len(self.registered_users)} registered users, τ={tau_s})")

        # memory_buffer 데이터 복원
        if 'memory_buffer_data' in checkpoint:
            buf_data = checkpoint['memory_buffer_data']
            buf_paths = buf_data['paths']
            buf_labels = buf_data['labels']
            if buf_paths:
                self.memory_buffer.update_from_dataset(buf_paths, buf_labels)
                if self.verbose:
                    print(f"[OK] Memory buffer restored ({len(buf_paths)} samples)")

        # probe_data 복원
        if 'probe_data' in checkpoint and self.openset_enabled:
            self.probe_data = {
                uid: (paths, labels)
                for uid, (paths, labels) in checkpoint['probe_data'].items()
            }
            if self.verbose:
                print(f"[OK] Probe data restored ({len(self.probe_data)} users)")

        if self.verbose:
            print(f"[OK] Checkpoint loaded from: {path} (experience={self.experience_count})")

    def save_eval_curve(self, path: str):
        """
        evaluation_history를 flat CSV로 저장 (논문 Figure용)
        컬럼: experience, num_users, tau_s, Rank1, FNIR, FRR, MisID, FPIR_in, FPIR_xdom
        """
        if not self.evaluation_history:
            print("WARNING: evaluation_history is empty. Nothing to save.")
            return

        import csv
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        fieldnames = [
            'experience', 'num_users', 'tau_s',
            'Rank1', 'FNIR', 'FRR', 'MisID',
            'FPIR_in', 'FPIR_xdom',
            'BWT', 'mean_forgetting'
        ]

        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in self.evaluation_history:
                m = entry.get('metrics', {})
                bwt_val = entry.get('bwt')
                fgt_val = entry.get('mean_forgetting')
                row = {
                    'experience':      entry.get('experience', ''),
                    'num_users':       entry.get('num_users', ''),
                    'tau_s':           f"{entry.get('tau_s', 0):.6f}",
                    'Rank1':           f"{m.get('Rank1', 0):.6f}",
                    'FNIR':            f"{m.get('FNIR', 0):.6f}",
                    'FRR':             f"{m.get('FRR', 0):.6f}",
                    'MisID':           f"{m.get('MisID', 0):.6f}",
                    'FPIR_in':         f"{m.get('FPIR_in', 0):.6f}",
                    'FPIR_xdom':       f"{m.get('FPIR_xdom', '') if m.get('FPIR_xdom') is not None else ''}",
                    'BWT':             f"{bwt_val:.6f}" if bwt_val is not None else '',
                    'mean_forgetting': f"{fgt_val:.6f}" if fgt_val is not None else '',
                }
                writer.writerow(row)

        if self.verbose:
            print(f"[eval_curve] saved: {path} ({len(self.evaluation_history)} rows)")

    def save_eval_curve_plot(self, path: str):
        """
        eval_curve.csv 데이터를 바탕으로 Performance vs Users 그래프 PNG 생성.
        서브플롯 3개: (1) TAR/FNIR vs Users  (2) FPIR vs Users  (3) τ vs Users
        """
        if not self.evaluation_history:
            print("WARNING: evaluation_history is empty. Cannot plot.")
            return

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("WARNING: matplotlib not available. Skipping plot.")
            return

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        num_users_list = [e['num_users'] for e in self.evaluation_history]
        tau_list       = [e.get('tau_s', 0) for e in self.evaluation_history]
        tar_list   = [e['metrics'].get('Rank1', 0)   for e in self.evaluation_history]
        fnir_list  = [e['metrics'].get('FNIR', 0)    for e in self.evaluation_history]
        fpir_list  = [e['metrics'].get('FPIR_in', 0) for e in self.evaluation_history]
        fpirx_list = [e['metrics'].get('FPIR_xdom')  for e in self.evaluation_history]
        bwt_list   = [e.get('bwt') for e in self.evaluation_history]
        fgt_list   = [e.get('mean_forgetting') for e in self.evaluation_history]

        has_bwt = any(v is not None for v in bwt_list)
        n_cols = 4 if has_bwt else 3

        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
        fig.suptitle('Performance vs Registered Users', fontsize=14, fontweight='bold')

        # 서브플롯 1: TAR & FNIR
        ax = axes[0]
        ax.plot(num_users_list, tar_list,  'b-o', markersize=4, label='TAR (Rank-1)')
        ax.plot(num_users_list, fnir_list, 'r-s', markersize=4, label='FNIR')
        ax.set_xlabel('Registered Users')
        ax.set_ylabel('Rate')
        ax.set_title('TAR & FNIR vs Users')
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 서브플롯 2: FPIR
        ax = axes[1]
        ax.plot(num_users_list, fpir_list, 'g-^', markersize=4, label='FPIR_in')
        has_xdom = any(v is not None for v in fpirx_list)
        if has_xdom:
            xdom_vals = [v if v is not None else float('nan') for v in fpirx_list]
            ax.plot(num_users_list, xdom_vals, 'm-v', markersize=4, label='FPIR_xdom')
        ax.set_xlabel('Registered Users')
        ax.set_ylabel('FPIR')
        ax.set_title('FPIR vs Users')
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 서브플롯 3: τ
        ax = axes[2]
        ax.plot(num_users_list, tau_list, 'k-o', markersize=4, label='τ_s')
        ax.set_xlabel('Registered Users')
        ax.set_ylabel('Threshold τ')
        ax.set_title('Threshold τ vs Users')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 서브플롯 4: BWT & Forgetting (데이터 있을 때만)
        if has_bwt:
            ax = axes[3]
            bwt_vals = [v if v is not None else float('nan') for v in bwt_list]
            ax.plot(num_users_list, bwt_vals, 'c-D', markersize=4, label='BWT')
            ax.axhline(0, color='gray', linestyle='--', linewidth=1)
            has_fgt = any(v is not None for v in fgt_list)
            if has_fgt:
                fgt_vals = [v if v is not None else float('nan') for v in fgt_list]
                ax.plot(num_users_list, fgt_vals, 'orange', linestyle='-',
                        marker='x', markersize=4, label='Mean Forgetting')
            ax.set_xlabel('Registered Users')
            ax.set_ylabel('BWT / Forgetting')
            ax.set_title('BWT & Forgetting vs Users')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        if self.verbose:
            print(f"[Plot] Performance vs Users saved: {path}")

    def save_det_curve(self, path: str, n_points: int = 200):
        """
        τ를 sweep하여 FNIR vs FPIR 곡선 데이터를 CSV로 저장 (논문 DET curve용)

        FNIR(τ) = P(genuine_max_score < τ)  ← 거부율 (reject at threshold τ)
        FPIR(τ) = P(unknown_max_score >= τ) ← 오수락율

        주의: FNIR은 rejection만 반영 (misidentification은 τ와 무관하게 발생하므로
              별도 고정값으로 더해야 정확하나, τ sweep 목적상 근사값으로 사용)
        특정 FPIR 기준점(0.1%, 1%, 5%)에서의 FNIR도 출력
        """
        if len(self._last_genuine_scores) == 0 or len(self._last_impostor_scores) == 0:
            print("WARNING: No cached scores for DET curve. Run calibration first.")
            return

        import csv
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        g = self._last_genuine_scores
        u = self._last_impostor_scores

        # τ 범위: 전체 스코어 분포에 맞게 자동 설정
        tau_min = float(min(g.min(), u.min())) - 0.05
        tau_max = float(max(g.max(), u.max())) + 0.05
        tau_range = np.linspace(tau_min, tau_max, n_points)

        rows = []
        for tau in tau_range:
            fnir = float((g < tau).mean())   # rejection rate at τ (FNIR 근사)
            fpir = float((u >= tau).mean())  # false positive rate at τ
            rows.append({'tau': f"{tau:.6f}", 'FNIR': f"{fnir:.6f}", 'FPIR': f"{fpir:.6f}"})

        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['tau', 'FNIR', 'FPIR'])
            writer.writeheader()
            writer.writerows(rows)

        if self.verbose:
            print(f"[det_curve] saved: {path} ({n_points} τ points)")
            print(f"   DET curve operating points:")
            for target_fpir in [0.001, 0.01, 0.05]:
                fpir_arr = np.array([(u >= tau).mean() for tau in tau_range])
                idx = np.argmin(np.abs(fpir_arr - target_fpir))
                tau_at = tau_range[idx]
                fnir_at = (g < tau_at).mean()
                fpir_at = fpir_arr[idx]
                print(f"   FPIR={target_fpir*100:.1f}%: τ={tau_at:.4f}, FNIR={fnir_at*100:.2f}%  (actual FPIR={fpir_at*100:.2f}%)")
