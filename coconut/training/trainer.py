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
        extract_features,
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
        extract_features,
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
        self.seed = getattr(config.training, 'seed', 42) if hasattr(config, 'training') else 42
        set_seed(self.seed)
        if self.verbose:
            print(f" Random seed set to {self.seed}")

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

        # IDL+RTM Loss 초기화 (Su et al., "Open-Set Biometrics")
        self.use_idl_rtm = getattr(config.training, 'use_idl_rtm', False)
        if self.use_idl_rtm:
            from coconut.losses import IDL_RTMLoss
            self.idl_rtm_loss = IDL_RTMLoss(
                alpha=getattr(config.training, 'idl_alpha_det', 6.0),
                beta=getattr(config.training, 'idl_beta_id', 0.2),
                gamma=getattr(config.training, 'idl_gamma_rank', 6.0),
                rtm_lambda=getattr(config.training, 'idl_rtm_weight', 4.0),
                min_gallery_classes=getattr(config.training, 'idl_min_gallery_classes', 2),
                gallery_fraction=getattr(config.training, 'idl_gallery_fraction', 0.7),
            )
            self.idl_rtm_lambda = getattr(config.training, 'idl_rtm_lambda', 0.3)
            self.idl_rtm_warmup_users = getattr(config.training, 'idl_rtm_warmup_users', 5)
            if self.verbose:
                print(f"[COCONUT] IDL+RTM Loss enabled: ext_λ={self.idl_rtm_lambda}, "
                      f"α={self.idl_rtm_loss.alpha}, β={self.idl_rtm_loss.beta}, "
                      f"γ={self.idl_rtm_loss.gamma}, RTM_λ={self.idl_rtm_loss.rtm_lambda}")
        else:
            self.idl_rtm_loss = None
            self.idl_rtm_lambda = 0.0
            self.idl_rtm_warmup_users = 0

        # DER++ (Dark Experience Replay) — 6144D feature distillation
        self.der_alpha = getattr(config.training, 'der_alpha', 0.0)
        self.der_batch_size = getattr(config.training, 'der_batch_size', 32)
        self.der_warmup_users = getattr(config.training, 'der_warmup_users', 3)
        if self.der_alpha > 0 and self.verbose:
            print(f"[COCONUT] DER++ enabled: α={self.der_alpha}, "
                  f"batch_size={self.der_batch_size}, warmup={self.der_warmup_users}")

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

            # GHOST 초기화
            self.use_ghost = getattr(config.openset, 'use_ghost', False)
            self.ghost_n_augment = getattr(config.openset, 'ghost_n_augment', 10)
            if self.use_ghost:
                self.ncm.ghost_enabled = True
                self.ncm.ghost_shrinkage_min_n = getattr(config.openset, 'ghost_shrinkage_min_n', 10)
                if self.verbose:
                    print(f" GHOST rejection enabled (n_augment={self.ghost_n_augment}, shrinkage_min_n={self.ncm.ghost_shrinkage_min_n})")

            # ThresholdCalibrator 초기화
            score_mode = getattr(config.openset, 'score_mode', 'cosine')
            if score_mode == 'mahalanobis':
                cal_clip_range = None
            elif self.use_ghost:
                cal_clip_range = None  # GHOST score 범위 사전 불명
            else:
                cal_clip_range = (-1.0, 1.0)  # cosine 기본 범위

            self.threshold_calibrator = ThresholdCalibrator(
                mode=score_mode,
                threshold_mode=config.openset.threshold_mode,
                target_far=config.openset.target_far,
                alpha=config.openset.threshold_alpha,
                max_delta=config.openset.threshold_max_delta,
                clip_range=cal_clip_range,
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

            # 진단 로그 저장용
            self._diag_history = []

            # 초기 임계치 설정
            initial_tau = config.openset.initial_tau
            score_mode_init = getattr(config.openset, 'score_mode', 'cosine')
            if score_mode_init == 'mahalanobis':
                initial_tau = -float('inf')  # warmup 중 모두 수락, calibration이 적절한 값 설정
            elif self.use_ghost:
                initial_tau = 0.0  # GHOST score는 매우 작은 양수 → warmup 중 모두 수락

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
            self.use_ghost = False
            if self.verbose:
                print(" Open-set mode disabled")

        # Statistics
        self.experience_count = 0

        # CuDNN 설정: train_coconut.py의 재현성 설정을 유지
        # (128x128 고정 크기에서는 benchmark=True의 속도 이점 없음)

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
                min_dev=1 if len(image_paths) <= 5 else 2,
                seed=self.seed + user_id  # user별 고유 시드로 재현성 보장
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
                    memory_paths, memory_labels, _ = self.memory_buffer.sample(
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

                    # IDL+RTM Loss 계산 (배치 내 episode splitting)
                    if self.use_idl_rtm and len(set(batch_labels.tolist())) >= 2:
                        # 랜덤 view 선택으로 augmentation 편향 감소
                        idl_features = f1 if random.random() < 0.5 else f2
                        idl_rtm_loss_val, idl_rtm_info = self.idl_rtm_loss(idl_features, batch_labels)
                    else:
                        idl_rtm_loss_val = torch.tensor(0.0, device=self.device)
                        idl_rtm_info = {}

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

                        # IDL+RTM warmup: 사용자 수에 따라 가중치 점진 증가
                        if self.use_idl_rtm:
                            idl_rtm_weight = self.idl_rtm_lambda * min(1.0, num_users / max(1, self.idl_rtm_warmup_users))
                        else:
                            idl_rtm_weight = 0.0

                        loss = proxy_weight * loss_proxy + supcon_weight * loss_supcon + idl_rtm_weight * idl_rtm_loss_val

                        if iteration == 0 and epoch == 0:
                            # compact 모드용: curriculum weights 저장
                            self._last_curriculum = {
                                'w_proxy': proxy_weight, 'w_supcon': supcon_weight,
                                'loss_supcon': loss_supcon.item(), 'loss_proxy': loss_proxy.item()
                            }
                            if self.use_idl_rtm:
                                self._last_curriculum['w_idl_rtm'] = idl_rtm_weight
                                self._last_curriculum['loss_idl_rtm'] = idl_rtm_loss_val.item()
                                self._last_idl_rtm_info = idl_rtm_info
                            if self.verbose:
                                print(f"[Curriculum] users={num_users}, w_proxy={w_proxy:.2f}, w_supcon={w_supcon:.2f}, gate={'ON' if unique_in_batch >= 2 else 'OFF'}")
                                print(f"   Weights → proxy:{proxy_weight:.2f}, supcon:{supcon_weight:.2f}")
                                print(f"   SupCon: {loss_supcon.item():.4f}, ProxyAnchor: {loss_proxy.item():.4f}")
                                if self.use_idl_rtm and not idl_rtm_info.get('skipped', True):
                                    print(f"   IDL+RTM: w={idl_rtm_weight:.2f}, L_IDL={idl_rtm_info['L_IDL']:.4f}, L_RTM={idl_rtm_info['L_RTM']:.4f}")
                                    print(f"   S_det={idl_rtm_info['S_det_mean']:.4f}, S_id={idl_rtm_info['S_id_mean']:.4f} | G={idl_rtm_info['n_gallery']}, NM={idl_rtm_info['n_nonmated']}")
                    else:
                        # IDL+RTM도 포함
                        if self.use_idl_rtm:
                            idl_rtm_weight = self.idl_rtm_lambda * min(1.0, num_users / max(1, self.idl_rtm_warmup_users))
                            loss = loss_supcon + idl_rtm_weight * idl_rtm_loss_val
                        else:
                            loss = loss_supcon

                    # === DER++ Feature Distillation ===
                    if self.der_alpha > 0 and len(self.memory_buffer) > 0:
                        num_users_der = len(self.registered_users)
                        if num_users_der >= self.der_warmup_users:
                            # 재현성: DER 샘플링을 위한 결정적 시드 (메인 학습 RNG에 영향 주지 않음)
                            _rng_state = torch.random.get_rng_state()
                            der_seed = self.seed + self.experience_count * 10000 + epoch * 100 + iteration
                            torch.manual_seed(der_seed)

                            der_sample_size = min(self.der_batch_size, len(self.memory_buffer))
                            der_paths, _, der_stored_feats = self.memory_buffer.sample(der_sample_size)
                            torch.random.set_rng_state(_rng_state)  # RNG 복원

                            # None이 아닌 유효한 feature만 필터링
                            valid = [(p, f) for p, f in zip(der_paths, der_stored_feats)
                                     if f is not None]

                            if valid:
                                der_images = []
                                stored_tensors = []
                                for path, feat in valid:
                                    img = _open_with_channels(path, self.config.dataset.channels)
                                    der_images.append(self.test_transform(img).unsqueeze(0))
                                    stored_tensors.append(feat)

                                der_batch = torch.cat(der_images, dim=0).to(self.device)
                                stored_batch = torch.stack(stored_tensors).to(self.device)

                                # eval mode: 저장 시점과 동일한 BatchNorm (running stats)
                                # gradient는 여전히 흐름
                                self.model.eval()
                                current_feats = self.model.getFeatureCode(
                                    der_batch, use_projection=False
                                )
                                current_feats = F.normalize(current_feats, dim=-1)
                                self.model.train()

                                der_loss = F.mse_loss(current_feats, stored_batch.detach())
                                effective_alpha = self.der_alpha * min(
                                    1.0, num_users_der / max(1, self.der_warmup_users)
                                )
                                loss = loss + effective_alpha * der_loss

                                # 로깅용 저장
                                if iteration == 0 and epoch == 0:
                                    self._last_der_info = {
                                        'loss': der_loss.item(),
                                        'alpha_eff': effective_alpha,
                                        'n_valid': len(valid)
                                    }

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

            # 에포크 진행률과 평균 손실은 항상 표시해 학습 상황을 바로 확인 가능하게 함
            print(f"  [에포크 {epoch+1}/{self.config.training.epochs_per_experience}] 평균 손실: {avg_loss:.4f}")

        # 메모리 버퍼 업데이트 (DER++: 6144D feature도 함께 저장)
        if self.der_alpha > 0:
            self.model.eval()
            stored_features = []
            with torch.no_grad():
                for path in train_paths:
                    img = _open_with_channels(path, self.config.dataset.channels)
                    img_tensor = self.test_transform(img).unsqueeze(0).to(self.device)
                    feat = self.model.getFeatureCode(img_tensor, use_projection=False)
                    feat = F.normalize(feat, dim=-1)
                    stored_features.append(feat.squeeze(0).cpu())
            self.model.train()
        else:
            stored_features = None

        self.memory_buffer.update_from_dataset(train_paths, train_labels, stored_features)
        if self.verbose:
            print(f"Memory buffer size after update: {len(self.memory_buffer)}")

        # NCM 업데이트
        self._update_ncm()

        # NCM 갱신 후 코사인 유사도 분포 분석 (현재 백본 + 현재 프로토타입 기준으로 정확한 측정)
        self._analyze_cosine_distribution_epoch()

        # 디버깅: NCM과 버퍼 동기화 확인
        all_paths, all_labels, _ = self.memory_buffer.get_all_data()
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
                    w_idl_rtm = curriculum.get('w_idl_rtm', 0)
                    loss_line = f"  Loss: {first_loss:.4f} -> {last_loss:.4f} ({epochs}ep) | w_proxy={w_proxy:.2f}, w_supcon={w_supcon:.2f}"
                    if self.use_idl_rtm and w_idl_rtm > 0:
                        loss_line += f", w_idl_rtm={w_idl_rtm:.2f}"
                    der_info = getattr(self, '_last_der_info', {})
                    if der_info:
                        loss_line += f", DER={der_info['loss']:.4f}(α={der_info['alpha_eff']:.2f})"
                    print(loss_line)

                    # IDL+RTM 손실 상세 (활성화 시)
                    idl_info = getattr(self, '_last_idl_rtm_info', {})
                    if self.use_idl_rtm and not idl_info.get('skipped', True):
                        print(f"  IDL+RTM: L_IDL={idl_info['L_IDL']:.4f}, L_RTM={idl_info['L_RTM']:.4f} | S_det={idl_info['S_det_mean']:.4f}, S_id={idl_info['S_id_mean']:.4f} | G={idl_info['n_gallery']}, NM={idl_info['n_nonmated']}")

                    # Line 3: Su et al. FNIR@FPIR 지표
                    tau = self.ncm.tau_s
                    fnir1 = metrics.get('FNIR@1%FPIR', 0)
                    tar1 = metrics.get('TAR@1%FPIR', 0)
                    rank1 = metrics.get('Rank1', 0)
                    det_fail = metrics.get('det_fail@1%', 0)
                    id_fail = metrics.get('id_fail@1%', 0)
                    print(f"  tau={tau:.4f} | Rank-1={rank1:.3f} | "
                          f"FNIR@1%FPIR={fnir1:.3f} (TAR={tar1:.3f}) | "
                          f"Det실패={det_fail:.3f}, Id실패={id_fail:.3f}")

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
            if getattr(self, 'use_ghost', False):
                # GHOST: feature 추출 → ghost max score
                from coconut.openset.utils import load_paths_labels_excluding
                unk_paths, _ = load_paths_labels_excluding(str(unknown_dev_file), self.registered_users)
                if len(unk_paths) > 3000:
                    rng = np.random.RandomState(seed)
                    idx = rng.choice(len(unk_paths), 3000, replace=False)
                    unk_paths = [unk_paths[i] for i in idx]
                if unk_paths:
                    unk_feats = extract_features(
                        self.model, unk_paths, self.test_transform, self.device,
                        channels=channels, use_projection=use_projection
                    )
                    if len(unk_feats) > 0:
                        unk_tensor = torch.from_numpy(unk_feats).to(self.device)
                        s_impostor = self.ncm.compute_ghost_max_scores(unk_tensor).cpu().numpy()
            else:
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
                print(f"   Unknown Dev: {len(s_impostor)} scores from {unknown_dev_file}"
                      + (" (GHOST)" if getattr(self, 'use_ghost', False) else ""))
        else:
            print("  [WARN] unknown_dev_file not set. τ calibration skipped.")

        # probe_data에서 genuine 점수 추출 (참고용)
        all_probe_paths = []
        all_probe_labels = []
        for uid, (paths, labels) in self.probe_data.items():
            all_probe_paths.extend(paths)
            all_probe_labels.extend([uid] * len(paths))

        if getattr(self, 'use_ghost', False):
            # GHOST: feature 추출 → ghost max score
            if all_probe_paths:
                gen_feats = extract_features(
                    self.model, all_probe_paths, self.test_transform, self.device,
                    channels=channels, use_projection=use_projection
                )
                if len(gen_feats) > 0:
                    gen_tensor = torch.from_numpy(gen_feats).to(self.device)
                    s_genuine = self.ncm.compute_ghost_max_scores(gen_tensor).cpu().numpy()
                else:
                    s_genuine = np.array([])
            else:
                s_genuine = np.array([])
        else:
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
        """
        Su et al. FNIR@FPIR 프로토콜에 따른 오픈셋 평가

        Gallery: NCM의 class_means_dict (등록된 사용자 프로토타입)
        Mated Probe: eval_probe_file (test_left.txt)에서 등록된 사용자의 테스트 샘플
        Non-mated Probe: unknown_test_file에서 미등록 사용자 샘플
        """
        eval_seed = getattr(self.config.training, 'seed', 42)
        set_seed(eval_seed)
        channels = self.config.dataset.channels
        use_projection = getattr(self.config.model, 'use_projection_for_ncm', False)

        self.model.eval()

        if self.verbose:
            print("\n [Su et al.] Open-set Evaluation (FNIR@FPIR protocol):")

        # ========================================
        # Step 1: Mated Probe 로드 (eval_probe_file 사용)
        # ========================================
        eval_probe_file = getattr(self.config.dataset, 'eval_probe_file', None)

        if eval_probe_file and str(eval_probe_file) != 'None':
            all_test_paths, all_test_labels = load_paths_labels_from_txt(str(eval_probe_file))

            # 등록된 사용자에 해당하는 샘플만 필터링 = mated probe
            mated_paths = []
            mated_labels = []
            for path, label in zip(all_test_paths, all_test_labels):
                if label in self.registered_users:
                    mated_paths.append(path)
                    mated_labels.append(label)
        else:
            # fallback: 기존 probe_data 사용 (비권장)
            print("  [WARN] eval_probe_file not set. Falling back to probe_data (1~2 samples/user)")
            mated_paths = []
            mated_labels = []
            for uid, (paths, labels) in self.probe_data.items():
                mated_paths.extend(paths)
                mated_labels.extend([uid] * len(paths))

        # ========================================
        # Step 2: Non-mated Probe 로드 (unknown_test_file 사용)
        # ========================================
        unknown_test_file = getattr(self.config.dataset, 'unknown_test_file', None)

        if unknown_test_file and str(unknown_test_file) != 'None':
            nonmated_paths, _ = load_paths_labels_from_txt(str(unknown_test_file))
        else:
            print("  [WARN] unknown_test_file not set. Using enroll_file fallback.")
            enroll_paths, enroll_labels = load_paths_labels_from_txt(str(self.config.dataset.enroll_file))
            nonmated_paths = [p for p, l in zip(enroll_paths, enroll_labels) if l not in self.registered_users]

        if len(nonmated_paths) > 1000:
            rng = np.random.RandomState(eval_seed + 1000)
            nonmated_paths = rng.choice(nonmated_paths, 1000, replace=False).tolist()

        # ========================================
        # Step 3: Feature 추출
        # ========================================
        mated_feats = extract_features(
            self.model, mated_paths, self.test_transform, self.device,
            batch_size=64, channels=channels, use_projection=use_projection
        )

        nonmated_feats = extract_features(
            self.model, nonmated_paths, self.test_transform, self.device,
            batch_size=64, channels=channels, use_projection=use_projection
        )

        # ========================================
        # Step 4: NCM 스코어 계산
        # ========================================
        # 등록된 클래스 인덱스 마스크 (미등록 인덱스의 0점 제외용)
        registered_ids = sorted(self.ncm.class_means_dict.keys())

        # 4a. Non-mated probe → max score per probe (FPIR 임계치 결정용)
        nonmated_max_scores = np.array([])
        if len(nonmated_feats) > 0:
            nonmated_tensor = torch.from_numpy(nonmated_feats).to(self.device)
            if getattr(self, 'use_ghost', False):
                # GHOST score 사용
                nonmated_max_scores = self.ncm.compute_ghost_max_scores(nonmated_tensor).cpu().numpy()
            else:
                nonmated_ncm_scores = self.ncm.forward(nonmated_tensor)  # (N, C)
                if nonmated_ncm_scores.numel() > 0:
                    registered_scores = nonmated_ncm_scores[:, registered_ids]
                    nonmated_max_scores = registered_scores.max(dim=1).values.cpu().numpy()

        # 4b. Mated probe → genuine score & rank-1 prediction
        mated_genuine_scores = []
        mated_rank1_correct = []
        mated_max_scores = []

        if len(mated_feats) > 0:
            mated_tensor = torch.from_numpy(mated_feats).to(self.device)
            mated_ncm_scores = self.ncm.forward(mated_tensor)  # (N, C)

            # 등록된 클래스만의 스코어
            mated_registered_scores = mated_ncm_scores[:, registered_ids]
            # registered_ids 내에서의 인덱스 → 실제 class_id 매핑
            id_to_reg_idx = {cid: idx for idx, cid in enumerate(registered_ids)}

            # GHOST: detection용 max score를 미리 계산
            if getattr(self, 'use_ghost', False):
                mated_ghost_scores = self.ncm.compute_ghost_max_scores(mated_tensor).cpu().numpy()

            for i in range(len(mated_labels)):
                true_id = mated_labels[i]
                scores_reg = mated_registered_scores[i]  # (num_registered,)

                # Genuine score: 정답 클래스와의 유사도 (cosine, 분석용)
                if true_id in id_to_reg_idx:
                    genuine_score = scores_reg[id_to_reg_idx[true_id]].item()
                else:
                    genuine_score = -1.0

                # Rank-1 (cosine 기준, 변경 없음)
                pred_reg_idx = scores_reg.argmax().item()
                pred_id = registered_ids[pred_reg_idx]

                # Max score: GHOST면 ghost score, 아니면 cosine max
                if getattr(self, 'use_ghost', False):
                    max_score = mated_ghost_scores[i]
                else:
                    max_score = scores_reg.max().item()

                mated_genuine_scores.append(genuine_score)
                mated_max_scores.append(max_score)
                mated_rank1_correct.append(pred_id == true_id)

        mated_genuine_scores = np.array(mated_genuine_scores)
        mated_max_scores_arr = np.array(mated_max_scores)
        mated_rank1_correct = np.array(mated_rank1_correct)

        # ========================================
        # Step 5: FNIR@FPIR 계산 (Su et al. 정의)
        # ========================================
        target_fpirs = [0.01, 0.05, 0.10]
        results = {}

        for target_fpir in target_fpirs:
            if len(nonmated_max_scores) > 0:
                # τ = non-mated max score 분포의 (1 - target_fpir) quantile
                tau = np.quantile(nonmated_max_scores, 1 - target_fpir)
                achieved_fpir = np.mean(nonmated_max_scores >= tau)
            else:
                tau = self.ncm.tau_s if self.ncm.tau_s is not None else 0.5
                achieved_fpir = 0.0

            if len(mated_genuine_scores) > 0:
                # FN 조건 (논문 정의): Detection 실패 OR Identification 실패
                detection_fail = mated_max_scores_arr < tau
                identification_fail = ~mated_rank1_correct
                is_fn = detection_fail | identification_fail

                fnir = np.mean(is_fn)
                tar = 1.0 - fnir

                # Detection/Identification 실패 분해 (분석용)
                det_only = np.mean(detection_fail & ~identification_fail)
                id_only = np.mean(~detection_fail & identification_fail)
                both = np.mean(detection_fail & identification_fail)
            else:
                fnir = 1.0
                tar = 0.0
                det_only = id_only = both = 0.0

            fpir_key = f"{int(target_fpir*100):d}"
            results[f'FNIR@{fpir_key}%FPIR'] = fnir
            results[f'TAR@{fpir_key}%FPIR'] = tar
            results[f'tau@{fpir_key}%FPIR'] = tau
            results[f'achieved_FPIR@{fpir_key}%'] = achieved_fpir
            results[f'det_fail@{fpir_key}%'] = det_only
            results[f'id_fail@{fpir_key}%'] = id_only
            results[f'both_fail@{fpir_key}%'] = both

        # ========================================
        # Step 6: Closed-set Rank-1 + xdomain FPIR
        # ========================================
        rank1 = np.mean(mated_rank1_correct) if len(mated_rank1_correct) > 0 else 0.0
        results['Rank1'] = rank1

        # FPIR_xdom (크로스도메인)
        TRR_n = FAR_n = None
        _negref_source = str(self.config.dataset.xdomain_file)
        negref_paths, _ = load_paths_labels_from_txt(_negref_source)

        if len(negref_paths) > 1000:
            rng = np.random.RandomState(eval_seed + 2000)
            negref_paths = rng.choice(negref_paths, 1000, replace=False).tolist()

        if negref_paths:
            preds_neg = predict_batch(
                self.model, self.ncm, negref_paths, self.test_transform, self.device,
                channels=channels, use_projection=use_projection
            )
            TRR_n = sum(1 for p in preds_neg if p == -1) / len(preds_neg)
            FAR_n = 1 - TRR_n

        results['TRR_negref'] = TRR_n
        results['FPIR_xdom'] = FAR_n

        # ========================================
        # Step 6.5: 종합 진단 로그
        # ========================================
        if len(mated_max_scores_arr) >= 10:
            _ms = mated_max_scores_arr
            _gs = mated_genuine_scores
            _rc = mated_rank1_correct
            _labels = np.array(mated_labels)

            # --- 1. Genuine score 분포 상세 ---
            print(f"\n{'='*70}")
            print(f"[DIAG] Genuine Score Distribution (n={len(_ms)})")
            print(f"  max_score:    \u03bc={_ms.mean():.4f} \u03c3={_ms.std():.4f} "
                  f"min={_ms.min():.4f} p5={np.percentile(_ms,5):.4f} "
                  f"p25={np.percentile(_ms,25):.4f} median={np.median(_ms):.4f}")
            print(f"  genuine_score: \u03bc={_gs.mean():.4f} \u03c3={_gs.std():.4f} "
                  f"min={_gs.min():.4f} p5={np.percentile(_gs,5):.4f}")

            # --- 2. Top1-Top2 Margin 분석 ---
            _margin = None
            _nm_top1 = None
            _nm_margin = None
            _fn_cos = None
            _fn_margin = None
            _tau_cos = None
            _tau_margin = None

            if len(mated_feats) > 0:
                _mt = torch.from_numpy(mated_feats).to(self.device)
                _all_scores = self.ncm.forward(_mt)
                _reg_scores = _all_scores[:, registered_ids]

                if _reg_scores.shape[1] >= 2:
                    _topk = _reg_scores.topk(2, dim=1)
                    _top1 = _topk.values[:, 0].cpu().numpy()
                    _top2 = _topk.values[:, 1].cpu().numpy()
                    _margin = _top1 - _top2

                    print(f"\n[DIAG] Top1-Top2 Margin (genuine)")
                    print(f"  margin: \u03bc={_margin.mean():.4f} \u03c3={_margin.std():.4f} "
                          f"min={_margin.min():.4f} p5={np.percentile(_margin,5):.4f} "
                          f"median={np.median(_margin):.4f}")

                    # tail vs normal
                    _tail_mask = _ms < np.percentile(_ms, 20)
                    _normal_mask = _ms >= np.percentile(_ms, 50)

                    if _tail_mask.sum() > 0 and _normal_mask.sum() > 0:
                        print(f"\n[DIAG] Tail(\ud558\uc704 20%) vs Normal(\uc0c1\uc704 50%)")
                        print(f"  Tail   (n={_tail_mask.sum()}): "
                              f"cos={_ms[_tail_mask].mean():.4f} "
                              f"margin={_margin[_tail_mask].mean():.4f} "
                              f"rank1_acc={_rc[_tail_mask].mean():.3f}")
                        print(f"  Normal (n={_normal_mask.sum()}): "
                              f"cos={_ms[_normal_mask].mean():.4f} "
                              f"margin={_margin[_normal_mask].mean():.4f} "
                              f"rank1_acc={_rc[_normal_mask].mean():.3f}")

            # --- 3. Per-class Worst/Best ---
            print(f"\n[DIAG] Per-class Worst/Best 5")
            _class_scores = {}
            for i, lbl in enumerate(_labels):
                if lbl not in _class_scores:
                    _class_scores[lbl] = []
                _class_scores[lbl].append(_ms[i])

            _class_means = {k: np.mean(v) for k, v in _class_scores.items()}
            _worst5 = sorted(_class_means.items(), key=lambda x: x[1])[:5]
            _best5 = sorted(_class_means.items(), key=lambda x: x[1], reverse=True)[:5]

            for uid, score in _worst5:
                print(f"  Worst: User {uid} -> mean_score={score:.4f} (n={len(_class_scores[uid])})")
            for uid, score in _best5:
                print(f"  Best:  User {uid} -> mean_score={score:.4f} (n={len(_class_scores[uid])})")

            # --- 4. GHOST 진단 ---
            _corr = None
            _s_values = None
            _gammas = None
            if getattr(self, 'use_ghost', False) and hasattr(self.ncm, 'ghost_class_stds_raw') and self.ncm.ghost_class_stds_raw:
                _cosines_g = []
                _s_values_list = []
                _gammas_list = []
                _is_correct = []

                _mt2 = torch.from_numpy(mated_feats).to(self.device)
                _scores2 = self.ncm.forward(_mt2)
                _top1_vals = _scores2.max(dim=1).values.cpu().numpy()
                _top1_ids = _scores2.argmax(dim=1).cpu().numpy()

                for i in range(len(_labels)):
                    k = int(_top1_ids[i])
                    if k not in self.ncm.ghost_class_stds_raw:
                        continue

                    mu_k = self.ncm.ghost_class_means_raw[k].to(self.device)
                    std_k = self.ncm.ghost_class_stds_raw[k].to(self.device)

                    n_k = self.ncm.ghost_class_counts.get(k, 1)
                    if self.ncm.ghost_global_std_raw is not None:
                        lam = min(n_k / self.ncm.ghost_shrinkage_min_n, 1.0)
                        g_std = self.ncm.ghost_global_std_raw.to(self.device)
                        std_k = lam * std_k + (1 - lam) * g_std

                    s = (torch.abs(_mt2[i] - mu_k) / (std_k + 1e-12)).sum().item()
                    gamma = float(_top1_vals[i]) / (s + 1e-12)

                    _cosines_g.append(float(_top1_vals[i]))
                    _s_values_list.append(s)
                    _gammas_list.append(gamma)
                    _is_correct.append(int(_top1_ids[i]) == _labels[i])

                if len(_cosines_g) >= 5:
                    _cosines_g = np.array(_cosines_g)
                    _s_values = np.array(_s_values_list)
                    _gammas = np.array(_gammas_list)

                    _corr = np.corrcoef(_cosines_g, _s_values)[0, 1]

                    print(f"\n[DIAG] GHOST Analysis (n={len(_cosines_g)})")
                    print(f"  s values: \u03bc={_s_values.mean():.0f} \u03c3={_s_values.std():.0f} "
                          f"min={_s_values.min():.0f} max={_s_values.max():.0f}")
                    print(f"  \u03b3 values: \u03bc={_gammas.mean():.6f} \u03c3={_gammas.std():.6f} "
                          f"min={_gammas.min():.6f} max={_gammas.max():.6f}")
                    print(f"  corr(cosine, s) = {_corr:.3f}  "
                          f"{'<- High: GHOST limited' if abs(_corr) > 0.5 else '<- Low: GHOST promising'}")

                    _tail_idx = np.argsort(_cosines_g)[:5]
                    print(f"\n  [Genuine Tail Top-5]")
                    for idx in _tail_idx:
                        correct = "O" if _is_correct[idx] else "X"
                        print(f"    [{correct}] cos={_cosines_g[idx]:.3f} s={_s_values[idx]:.0f} "
                              f"\u03b3={_gammas[idx]:.6f}")

            # --- 5. Impostor vs Genuine margin 비교 ---
            if len(nonmated_feats) > 0 and _margin is not None:
                _nmt = torch.from_numpy(nonmated_feats).to(self.device)
                _nm_scores = self.ncm.forward(_nmt)
                _nm_reg = _nm_scores[:, registered_ids]

                if _nm_reg.shape[1] >= 2:
                    _nm_topk = _nm_reg.topk(2, dim=1)
                    _nm_top1 = _nm_topk.values[:, 0].cpu().numpy()
                    _nm_top2 = _nm_topk.values[:, 1].cpu().numpy()
                    _nm_margin = _nm_top1 - _nm_top2

                    print(f"\n[DIAG] Impostor vs Genuine")
                    print(f"  Genuine  max_cos: \u03bc={_ms.mean():.4f}  margin: \u03bc={_margin.mean():.4f}")
                    print(f"  Impostor max_cos: \u03bc={_nm_top1.mean():.4f}  margin: \u03bc={_nm_margin.mean():.4f}")
                    print(f"  separation(cosine):  {_ms.mean() - _nm_top1.mean():.4f}")
                    print(f"  separation(margin):  {_margin.mean() - _nm_margin.mean():.4f}")

                    _tau_cos = np.percentile(_nm_top1, 99)
                    _tau_margin = np.percentile(_nm_margin, 99)
                    _fn_cos = np.mean(_ms < _tau_cos)
                    _fn_margin = np.mean(_margin < _tau_margin)

                    print(f"\n  @1%FPIR simulation:")
                    print(f"    cosine  tau={_tau_cos:.4f} -> genuine reject={_fn_cos*100:.1f}%")
                    print(f"    margin  tau={_tau_margin:.4f} -> genuine reject={_fn_margin*100:.1f}%")
                    print(f"    {'-> margin BETTER!' if _fn_margin < _fn_cos else '-> cosine better'}")

            # --- 6. 진단 데이터 저장 ---
            _diag_entry = {
                'exp': len(self.registered_users) - 1,
                'n_classes': self.ncm.get_num_classes(),
                'n_mated': len(_ms),
                'rank1': float(_rc.mean()),
                'genuine_max_mean': float(_ms.mean()),
                'genuine_max_std': float(_ms.std()),
                'genuine_max_min': float(_ms.min()),
                'genuine_max_p5': float(np.percentile(_ms, 5)),
                'genuine_max_p25': float(np.percentile(_ms, 25)),
                'genuine_max_median': float(np.median(_ms)),
                'genuine_score_mean': float(_gs.mean()),
                'genuine_score_std': float(_gs.std()),
            }

            if _margin is not None:
                _diag_entry.update({
                    'margin_mean': float(_margin.mean()),
                    'margin_std': float(_margin.std()),
                    'margin_min': float(_margin.min()),
                    'margin_p5': float(np.percentile(_margin, 5)),
                    'margin_median': float(np.median(_margin)),
                })
                if _tail_mask.sum() > 0 and _normal_mask.sum() > 0:
                    _diag_entry.update({
                        'tail_n': int(_tail_mask.sum()),
                        'tail_cos_mean': float(_ms[_tail_mask].mean()),
                        'tail_margin_mean': float(_margin[_tail_mask].mean()),
                        'tail_rank1': float(_rc[_tail_mask].mean()),
                        'normal_n': int(_normal_mask.sum()),
                        'normal_cos_mean': float(_ms[_normal_mask].mean()),
                        'normal_margin_mean': float(_margin[_normal_mask].mean()),
                        'normal_rank1': float(_rc[_normal_mask].mean()),
                    })

            if _nm_top1 is not None:
                _diag_entry.update({
                    'impostor_max_mean': float(_nm_top1.mean()),
                    'impostor_margin_mean': float(_nm_margin.mean()),
                    'separation_cosine': float(_ms.mean() - _nm_top1.mean()),
                    'separation_margin': float(_margin.mean() - _nm_margin.mean()),
                })
                if _fn_cos is not None:
                    _diag_entry.update({
                        'sim_tau_cosine': float(_tau_cos),
                        'sim_fnir_cosine': float(_fn_cos),
                        'sim_tau_margin': float(_tau_margin),
                        'sim_fnir_margin': float(_fn_margin),
                    })

            if _corr is not None:
                _diag_entry.update({
                    'ghost_s_mean': float(_s_values.mean()),
                    'ghost_s_std': float(_s_values.std()),
                    'ghost_gamma_mean': float(_gammas.mean()),
                    'ghost_gamma_std': float(_gammas.std()),
                    'ghost_corr_cos_s': float(_corr),
                })

            if '_worst5' in dir():
                _diag_entry['worst_users'] = [(int(uid), float(sc)) for uid, sc in _worst5]
                _diag_entry['best_users'] = [(int(uid), float(sc)) for uid, sc in _best5]

            self._diag_history.append(_diag_entry)

            # === 10 경험마다 리포트 파일 저장 ===
            _exp_num = _diag_entry['exp']
            if (_exp_num + 1) % 10 == 0 or _exp_num == 0:
                import json, os
                _save_dir = str(self.config.training.results_path)
                os.makedirs(_save_dir, exist_ok=True)

                _json_path = os.path.join(_save_dir, 'diag_history.json')
                with open(_json_path, 'w') as f:
                    json.dump(self._diag_history, f, indent=2, ensure_ascii=False)

                _txt_path = os.path.join(_save_dir, f'diag_report_exp{_exp_num:03d}.txt')
                with open(_txt_path, 'w') as f:
                    f.write(f"{'='*80}\n")
                    f.write(f"COCONUT Diagnostic Report - Experience {_exp_num}\n")
                    f.write(f"{'='*80}\n\n")

                    f.write(f"{'Exp':>4} {'Cls':>4} {'R1':>5} "
                            f"{'cos_u':>6} {'cos_s':>6} {'cos_mn':>6} {'cos_p5':>6} "
                            f"{'mrg_u':>6} {'mrg_p5':>6} "
                            f"{'imp_c':>6} {'imp_m':>6} "
                            f"{'sp_c':>6} {'sp_m':>6} "
                            f"{'fn_c':>6} {'fn_m':>6}\n")
                    f.write(f"{'-'*110}\n")

                    for d in self._diag_history:
                        f.write(f"{d['exp']:4d} {d['n_classes']:4d} {d['rank1']:5.3f} "
                                f"{d['genuine_max_mean']:6.3f} {d['genuine_max_std']:6.3f} "
                                f"{d['genuine_max_min']:6.3f} {d['genuine_max_p5']:6.3f} ")
                        if 'margin_mean' in d:
                            f.write(f"{d['margin_mean']:6.3f} {d['margin_p5']:6.3f} ")
                        else:
                            f.write(f"{'N/A':>6} {'N/A':>6} ")
                        if 'impostor_max_mean' in d:
                            f.write(f"{d['impostor_max_mean']:6.3f} {d.get('impostor_margin_mean',0):6.3f} "
                                    f"{d.get('separation_cosine',0):6.3f} {d.get('separation_margin',0):6.3f} ")
                        else:
                            f.write(f"{'N/A':>6} {'N/A':>6} {'N/A':>6} {'N/A':>6} ")
                        if 'sim_fnir_cosine' in d:
                            f.write(f"{d['sim_fnir_cosine']:6.3f} {d['sim_fnir_margin']:6.3f}")
                        else:
                            f.write(f"{'N/A':>6} {'N/A':>6}")
                        f.write("\n")

                    # GHOST 추이
                    if any('ghost_corr_cos_s' in d for d in self._diag_history):
                        f.write(f"\n\nGHOST corr(cosine, s) trend\n{'='*60}\n")
                        for d in self._diag_history:
                            if 'ghost_corr_cos_s' in d:
                                f.write(f"  Exp {d['exp']:3d}: corr={d['ghost_corr_cos_s']:.3f}  "
                                        f"s_u={d['ghost_s_mean']:.0f}  "
                                        f"g_u={d['ghost_gamma_mean']:.6f}\n")

                    # Tail vs Normal 추이
                    if any('tail_cos_mean' in d for d in self._diag_history):
                        f.write(f"\n\nTail(bot20%) vs Normal(top50%) trend\n{'='*60}\n")
                        for d in self._diag_history:
                            if 'tail_cos_mean' in d:
                                f.write(f"  Exp {d['exp']:3d}: "
                                        f"tail cos={d['tail_cos_mean']:.3f} mrg={d['tail_margin_mean']:.3f} r1={d['tail_rank1']:.3f} | "
                                        f"norm cos={d['normal_cos_mean']:.3f} mrg={d['normal_margin_mean']:.3f} r1={d['normal_rank1']:.3f}\n")

                    # Per-class worst 추이
                    f.write(f"\n\nPer-class Worst Users trend\n{'='*60}\n")
                    for d in self._diag_history:
                        if 'worst_users' in d:
                            worst_str = ', '.join([f"U{uid}({sc:.3f})" for uid, sc in d['worst_users']])
                            f.write(f"  Exp {d['exp']:3d}: {worst_str}\n")

                    # 핵심 결론
                    f.write(f"\n\nKey Summary\n{'='*60}\n")
                    latest = self._diag_history[-1]
                    if 'sim_fnir_cosine' in latest and 'sim_fnir_margin' in latest:
                        f.write(f"  @1%FPIR genuine reject rate:\n")
                        f.write(f"    cosine: {latest['sim_fnir_cosine']*100:.1f}%\n")
                        f.write(f"    margin: {latest['sim_fnir_margin']*100:.1f}%\n")
                        if latest['sim_fnir_margin'] < latest['sim_fnir_cosine']:
                            diff = (latest['sim_fnir_cosine'] - latest['sim_fnir_margin']) * 100
                            f.write(f"    -> margin is {diff:.1f}%p better\n")
                        else:
                            f.write(f"    -> cosine is better\n")
                    if 'ghost_corr_cos_s' in latest:
                        f.write(f"  GHOST corr(cos,s) = {latest['ghost_corr_cos_s']:.3f}\n")
                        if abs(latest['ghost_corr_cos_s']) > 0.5:
                            f.write(f"    -> High correlation: GHOST effect limited\n")
                        else:
                            f.write(f"    -> Low correlation: GHOST promising\n")

                print(f"  [DIAG] Report saved: {_txt_path}")
                print(f"  [DIAG] JSON saved: {_json_path}")

            print(f"{'='*70}\n")

        # ========================================
        # Step 7: DET curve용 스코어 캐시
        # ========================================
        # DET curve용: detection 실패 판단에 max score 사용 (genuine score 아님)
        self._last_genuine_scores = mated_max_scores_arr.copy() if len(mated_max_scores_arr) > 0 else np.array([])
        self._last_impostor_scores = nonmated_max_scores.copy() if len(nonmated_max_scores) > 0 else np.array([])

        # τ_s 참고값 저장
        results['tau_s_current'] = self.ncm.tau_s

        # ========================================
        # Step 8: 결과 출력
        # ========================================
        if self.verbose:
            print(f"   Gallery: {self.ncm.get_num_classes()} users")
            print(f"   Mated probes: {len(mated_paths)} samples")
            print(f"   Non-mated probes: {len(nonmated_paths)} samples")
            print(f"   Rank-1 (closed-set): {rank1:.4f}")
            for fpir_pct in [1, 5, 10]:
                fnir_v = results[f'FNIR@{fpir_pct}%FPIR']
                tar_v = results[f'TAR@{fpir_pct}%FPIR']
                tau_v = results[f'tau@{fpir_pct}%FPIR']
                det_v = results[f'det_fail@{fpir_pct}%']
                idf_v = results[f'id_fail@{fpir_pct}%']
                both_v = results[f'both_fail@{fpir_pct}%']
                print(f"   FNIR@{fpir_pct}%FPIR = {fnir_v:.4f} (TAR={tar_v:.4f}, τ={tau_v:.4f})")
                print(f"     └ Det fail: {det_v:.4f}, Id fail: {idf_v:.4f}, Both: {both_v:.4f}")
            if TRR_n is not None:
                print(f"   FPIR_xdom: {FAR_n:.4f}")
            print(f"   Threshold (calibrated): τ_s={self.ncm.tau_s:.4f}")

        # Compact 출력
        if not self.verbose:
            fnir1 = results.get('FNIR@1%FPIR', 0)
            tar1 = results.get('TAR@1%FPIR', 0)
            print(f"  [Eval] Rank-1={rank1:.3f} | FNIR@1%FPIR={fnir1:.3f} (TAR={tar1:.3f}) | "
                  f"mated={len(mated_paths)}, nonmated={len(nonmated_paths)}")

        # 하위 호환: 기존 키도 포함
        results['FNIR'] = results.get('FNIR@1%FPIR', 0)
        results['FPIR_in'] = results.get('achieved_FPIR@1%', 0)
        results['FRR'] = results.get('det_fail@1%', 0)
        results['MisID'] = results.get('id_fail@1%', 0)
        results['TRR_unknown'] = 1.0 - results.get('achieved_FPIR@1%', 0)
        results['mode'] = 'fnir_at_fpir'
        results['score_type'] = 'max'

        return results

    @torch.no_grad()
    def _update_ncm(self):
        """NCM classifier의 class means를 업데이트합니다."""
        if len(self.memory_buffer) == 0:
            return

        self.model.eval()

        # 메모리 버퍼에서 모든 데이터 가져오기
        all_paths, all_labels, _ = self.memory_buffer.get_all_data()

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

        # Global diagonal variance 계산 (Mahalanobis 모드)
        ncm_score_mode = getattr(self.config.openset, 'score_mode', 'cosine')
        if ncm_score_mode == 'mahalanobis':
            all_features = []
            for features_list in class_features.values():
                all_features.extend(features_list)
            if len(all_features) >= 2:
                all_feat_tensor = torch.stack(all_features)
                all_feat_norm = F.normalize(all_feat_tensor, p=2, dim=1, eps=1e-12)
                global_var = all_feat_norm.var(dim=0, unbiased=True)
                self.ncm.set_global_var(global_var)

        # GHOST: augmented raw features로 per-class μ_raw, σ_raw 계산
        if getattr(self, 'use_ghost', False):
            use_projection = getattr(self.config.model, 'use_projection_for_ncm', False)
            channels = self.config.dataset.channels

            # class별 경로 그룹핑
            from collections import defaultdict
            class_paths_dict = defaultdict(list)
            for path, label in zip(real_paths, real_labels):
                lbl = int(label) if not isinstance(label, int) else label
                class_paths_dict[lbl].append(path)

            ghost_means_raw = {}
            ghost_stds_raw = {}
            ghost_counts = {}
            all_raw_feats_for_global = []

            for label, paths_for_class in class_paths_dict.items():
                raw_feats = []
                for path in paths_for_class:
                    img = _open_with_channels(path, channels)
                    for _ in range(self.ghost_n_augment):
                        aug_tensor = self.train_transform(img).unsqueeze(0).to(self.device)
                        feat = self.model.getFeatureCode(aug_tensor, use_projection=use_projection)
                        # raw feature (L2 norm 안 함!)
                        raw_feats.append(feat.squeeze(0).cpu())

                ghost_counts[label] = len(raw_feats)

                if len(raw_feats) >= 2:
                    feat_tensor = torch.stack(raw_feats)
                    ghost_means_raw[label] = feat_tensor.mean(dim=0)
                    ghost_stds_raw[label] = feat_tensor.std(dim=0).clamp_min(1e-6)
                else:
                    ghost_means_raw[label] = raw_feats[0] if raw_feats else torch.zeros(1)
                    ghost_stds_raw[label] = torch.ones_like(ghost_means_raw[label])

                all_raw_feats_for_global.extend(raw_feats)

            # Global std (shrinkage target)
            ghost_global_std = None
            if len(all_raw_feats_for_global) >= 2:
                all_tensor = torch.stack(all_raw_feats_for_global)
                ghost_global_std = all_tensor.std(dim=0).clamp_min(1e-6)

            self.ncm.set_ghost_stats(ghost_means_raw, ghost_stds_raw, ghost_global_std, ghost_counts)

            if self.verbose:
                avg_n = np.mean(list(ghost_counts.values())) if ghost_counts else 0
                print(f" GHOST stats updated: {len(ghost_stds_raw)} classes, "
                      f"avg {avg_n:.0f} augmented features/class")

        if self.verbose:
            print(f" Updated NCM with {len(class_means)} classes"
                  + (f" (score_mode={ncm_score_mode})" if ncm_score_mode != 'cosine' else "")
                  + (f" + GHOST" if getattr(self, 'use_ghost', False) else ""))

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
        all_paths, all_labels, _ = self.memory_buffer.get_all_data()
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
            buf_paths, buf_labels, buf_logits = self.memory_buffer.get_all_data()
            checkpoint_dict['memory_buffer_data'] = {
                'paths': buf_paths,
                'labels': [int(l) for l in buf_labels],
                'logits': [l.cpu() if l is not None else None for l in buf_logits]
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
            buf_logits = buf_data.get('logits', None)  # 이전 체크포인트 호환
            if buf_paths:
                self.memory_buffer.update_from_dataset(buf_paths, buf_labels, buf_logits)
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
        Su et al. FNIR@FPIR 프로토콜 기준 지표 포함
        """
        if not self.evaluation_history:
            print("WARNING: evaluation_history is empty. Nothing to save.")
            return

        import csv
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        fieldnames = [
            'experience', 'num_users', 'tau_s',
            'Rank1',
            'FNIR@1%FPIR', 'FNIR@5%FPIR', 'FNIR@10%FPIR',
            'TAR@1%FPIR', 'TAR@5%FPIR', 'TAR@10%FPIR',
            'det_fail@1%', 'id_fail@1%', 'both_fail@1%',
            'FPIR_xdom',
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
                    'FNIR@1%FPIR':     f"{m.get('FNIR@1%FPIR', 0):.6f}",
                    'FNIR@5%FPIR':     f"{m.get('FNIR@5%FPIR', 0):.6f}",
                    'FNIR@10%FPIR':    f"{m.get('FNIR@10%FPIR', 0):.6f}",
                    'TAR@1%FPIR':      f"{m.get('TAR@1%FPIR', 0):.6f}",
                    'TAR@5%FPIR':      f"{m.get('TAR@5%FPIR', 0):.6f}",
                    'TAR@10%FPIR':     f"{m.get('TAR@10%FPIR', 0):.6f}",
                    'det_fail@1%':     f"{m.get('det_fail@1%', 0):.6f}",
                    'id_fail@1%':      f"{m.get('id_fail@1%', 0):.6f}",
                    'both_fail@1%':    f"{m.get('both_fail@1%', 0):.6f}",
                    'FPIR_xdom':       f"{m.get('FPIR_xdom', '')}" if m.get('FPIR_xdom') is not None else '',
                    'BWT':             f"{bwt_val:.6f}" if bwt_val is not None else '',
                    'mean_forgetting': f"{fgt_val:.6f}" if fgt_val is not None else '',
                }
                writer.writerow(row)

        if self.verbose:
            print(f"[eval_curve] saved: {path} ({len(self.evaluation_history)} rows)")

    def save_eval_curve_plot(self, path: str):
        """
        Su et al. FNIR@FPIR 프로토콜 기준 Performance vs Users 그래프 PNG 생성.
        서브플롯: (1) Rank-1 & FNIR@1%FPIR  (2) FNIR 분해  (3) τ@1%FPIR  (4) BWT/Forgetting
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

        # Su et al. 지표
        rank1_list = [e['metrics'].get('Rank1', 0) for e in self.evaluation_history]
        fnir1_list = [e['metrics'].get('FNIR@1%FPIR', 0) for e in self.evaluation_history]
        tar1_list  = [e['metrics'].get('TAR@1%FPIR', 0) for e in self.evaluation_history]
        tau1_list  = [e['metrics'].get('tau@1%FPIR', 0) for e in self.evaluation_history]

        # FNIR 분해
        det_list  = [e['metrics'].get('det_fail@1%', 0) for e in self.evaluation_history]
        id_list   = [e['metrics'].get('id_fail@1%', 0) for e in self.evaluation_history]
        both_list = [e['metrics'].get('both_fail@1%', 0) for e in self.evaluation_history]

        bwt_list = [e.get('bwt') for e in self.evaluation_history]
        fgt_list = [e.get('mean_forgetting') for e in self.evaluation_history]

        has_bwt = any(v is not None for v in bwt_list)
        n_cols = 4 if has_bwt else 3

        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
        fig.suptitle('Su et al. FNIR@FPIR — Performance vs Registered Users', fontsize=13, fontweight='bold')

        # 서브플롯 1: Rank-1 & FNIR@1%FPIR
        ax = axes[0]
        ax.plot(num_users_list, rank1_list, 'b-o', markersize=4, label='Rank-1')
        ax.plot(num_users_list, fnir1_list, 'r-s', markersize=4, label='FNIR@1%FPIR')
        ax.plot(num_users_list, tar1_list,  'g--^', markersize=3, alpha=0.7, label='TAR@1%FPIR')
        ax.set_xlabel('Registered Users')
        ax.set_ylabel('Rate')
        ax.set_title('Rank-1 & FNIR@1%FPIR')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 서브플롯 2: FNIR 분해 (Det fail, Id fail, Both)
        ax = axes[1]
        ax.plot(num_users_list, det_list,  'c-o', markersize=4, label='Det fail only')
        ax.plot(num_users_list, id_list,   'm-s', markersize=4, label='Id fail only')
        ax.plot(num_users_list, both_list, 'k-^', markersize=4, label='Both fail')
        ax.set_xlabel('Registered Users')
        ax.set_ylabel('Rate')
        ax.set_title('FNIR Decomposition @1%FPIR')
        ax.set_ylim(0, max(0.5, max(max(det_list), max(id_list), max(both_list)) * 1.2) if det_list else 0.5)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 서브플롯 3: τ@1%FPIR vs calibrated τ_s
        ax = axes[2]
        ax.plot(num_users_list, tau1_list, 'r-o', markersize=4, label='τ@1%FPIR')
        ax.plot(num_users_list, tau_list,  'k--s', markersize=3, alpha=0.7, label='τ_s (calibrated)')
        ax.set_xlabel('Registered Users')
        ax.set_ylabel('Threshold τ')
        ax.set_title('Threshold τ vs Users')
        ax.legend(fontsize=8)
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
            ax.legend(fontsize=8)
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
