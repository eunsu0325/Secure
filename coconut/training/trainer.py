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

from coconut.losses import ProxyAnchorLoss
from coconut.data import MemoryDataset, get_scr_transforms
from .average_meter import AverageMeter
from coconut.models import PretrainedLoader, ProjectionHead, ProjectionWrappedModel
from coconut.classifiers.threshold import ThresholdCalibrator

# 오픈셋 유틸리티 함수들
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
    set_seed,
    _open_with_channels
)


class _HistTracker:
    """DecayPredictor(at_risk)가 기대하는 tracker 인터페이스를
    _qar_score_history(dict: {uid: [raw-cosine score,...]})로 채우는 어댑터."""
    def __init__(self, hist):
        self.performance_matrix = hist

    def get_performance_trajectory(self, uid, metric=None):
        return list(self.performance_matrix.get(uid, []))


# B6: Diagnostic / calibration subsample caps — lifted from buried inline magic
# numbers so reviewers can audit "no cherry-pick" without grep-ing for ints.
# Full data is always used for the final paper metrics; these only cap how
# many samples are drawn for τ calibration (paper text: §3.3) or for
# diagnostic logging (PCA spectrum, xdomain FPIR).
MAX_UNK_CALIB_SAMPLES = 3000   # unknown_dev cap for τ calibration
MAX_NEGREF_EVAL_SAMPLES = 1000  # xdomain (negref) cap for FPIR_xdom diagnostic


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
    COCONUT Trainer: CCNet + ProxyAnchor + Memory Replay + QAR

    Combines:
    - CCNet backbone
    - ProxyAnchorLoss for class-level metric learning
    - Class-balanced memory replay + QAR tail-user rehab
    - Open-set recognition support (cosine NCM + S-norm)
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

        self.ncm = ncm_classifier
        self.memory_buffer = memory_buffer
        self.config = config

        # ProjectionHead 초기화 (선택적: CCNet 2048D → projection_dim)
        # PCA 가중치 초기화는 train_coconut.py에서 학습 시작 전에 별도 수행.
        # NOTE: self.model assignment happens AFTER projection setup so that
        # when projection is on, self.model points to the wrapper (not the raw
        # ccnet). This is critical for _create_optimizer_with_grouped_params
        # which reads self.model.ccnet / self.model.projection.
        self.use_projection_head = bool(getattr(config.training, 'use_projection_head', False))
        if self.use_projection_head:
            self.projection_dim = int(getattr(config.training, 'projection_dim', 512))
            self.projection_lr_ratio = float(getattr(config.training, 'projection_lr_ratio', 1.0))
            self.projection = ProjectionHead(in_dim=2048, out_dim=self.projection_dim).to(device)

            # Wrap the model so every downstream caller — including evaluator
            # and openset.score_extraction — automatically receives projected
            # features.
            self.model = ProjectionWrappedModel(
                ccnet=model, projection=self.projection
            ).to(device)

            if self.verbose:
                print(f"[COCONUT] ProjectionHead enabled: 2048 -> {self.projection_dim} "
                      f"(lr_ratio={self.projection_lr_ratio}x backbone)")
                print(f"[COCONUT] Model wrapped with ProjectionWrappedModel; every "
                      f"forward/getFeatureCode now passes through projection.")
        else:
            self.projection = None
            self.projection_dim = 2048
            self.projection_lr_ratio = 0.0
            self.model = model

        # ProxyAnchorLoss 초기화
        self.use_proxy_anchor = getattr(config.training, 'use_proxy_anchor', True)

        if self.use_proxy_anchor:
            # ProxyAnchor 의 embedding_size 는 projection 출력 차원에 맞춤
            embedding_dim = self.projection_dim

            use_canonical = getattr(config.training, 'use_canonical_proxy_loss', False)
            self.proxy_anchor_loss = ProxyAnchorLoss(
                embedding_size=embedding_dim,
                margin=getattr(config.training, 'proxy_margin', 0.1),
                alpha=getattr(config.training, 'proxy_alpha', 32),
                use_canonical=use_canonical,
            ).to(device)

            if self.verbose:
                print(f"[COCONUT] COCONUT ProxyAnchor initialized with {embedding_dim}D embeddings")

            self.proxy_lambda = getattr(config.training, 'proxy_lambda', 0.3)

            if self.verbose:
                print(f"[COCONUT] ProxyAnchorLoss enabled:")
                print(f"   Margin (δ): {self.proxy_anchor_loss.margin}")
                print(f"   Alpha (α): {self.proxy_anchor_loss.alpha}")
                print(f"   Lambda (fixed): {self.proxy_lambda}")
                print(f"   Negative term: {'canonical (paper Eq. 4)' if use_canonical else 'legacy (P+ only)'}")
        else:
            self.proxy_anchor_loss = None
            self.proxy_lambda = 0.0

        # QAR (Quality-Aware Replay) — tail user 선별 재학습
        self.use_qar = getattr(config.training, 'use_qar', False)
        self.rehab_margin = getattr(config.training, 'rehab_margin', 0.05)
        self.rehab_samples_per_user = getattr(config.training, 'rehab_samples_per_user', 4)
        self.qar_warmup_users = getattr(config.training, 'qar_warmup_users', 10)
        self._qar_class_scores = {}  # {user_id: mean_genuine_cosine} — 평가 시 갱신
        self._qar_raw_floor = None   # QAR tail 판정용 raw-cosine floor (tau_cos는 S-norm 시 z-score라 못 씀)
        self._qar_rehab_log = []     # 진단용: [(exp, tail_ids, n_rehab_samples)]
        self._qar_score_history = {} # {uid: [raw-cosine genuine score per eval]} — MRS ⓒ 궤적용

        # MRS (replay scheduling: ⓐ SSL + ⓑ cohort 간격 + ⓒ 위험 override)
        # ★ 전부 default OFF = 학습/평가 경로 byte-identical.
        from coconut.memory import CohortScheduler, DecayPredictor
        from coconut.losses import SSLConsistencyLoss
        self.use_mrs = getattr(config.training, 'use_mrs', False)
        self.w_ssl = getattr(config.training, 'w_ssl', 0.0)
        self.cohort_scheduler = CohortScheduler(
            recall_interval=getattr(config.training, 'mrs_recall_interval', 1),
            warmup_classes=getattr(config.training, 'mrs_warmup_users', 10),
        )
        self.decay_predictor = DecayPredictor(
            enabled=self.use_mrs,
            higher_is_better=True,   # raw-cosine genuine score: 클수록 좋음
            floor=0.0,               # 호출 시 self._qar_raw_floor(raw cosine)로 갱신
            horizon=1, min_history=2,
            cap=(getattr(config.training, 'mrs_override_cap', 0) or None),  # 0=무제한
        )
        self.ssl_loss = SSLConsistencyLoss(mode='cosine') if self.w_ssl > 0 else None

        if self.use_qar and self.verbose:
            print(f"[COCONUT] QAR enabled: margin={self.rehab_margin}, "
                  f"samples/user={self.rehab_samples_per_user}, warmup={self.qar_warmup_users}")

        # ── Loss-head ablation 대조군 (ProxyAnchor vs softmax) ──────────────
        # 'proxy'(기본) = 현행 byte-identical. softmax 모드면 ProxyAnchor를 끄고
        # SoftmaxHead로 대체(같은 백본·데이터·NCM eval, 손실만 교체 = loss-only 귀속).
        self.loss_head = getattr(config.training, 'loss_head', 'proxy')
        self.softmax_head = None
        self.softmax_lr_ratio = float(getattr(config.training, 'softmax_lr_ratio', 50.0))
        # softmax 손실 weight λ. 기본 = proxy_lambda → proxy와 *matched*(0.5-vs-1.0 비대칭 제거).
        # config.yaml의 proxy_lambda는 여기서 안 건드려져서(=원본) fallback으로 안전하게 읽힘.
        self.softmax_lambda = float(getattr(config.training, 'softmax_lambda',
                                            getattr(config.training, 'proxy_lambda', 0.5)))
        if self.loss_head != 'proxy':
            from coconut.losses import SoftmaxHead
            self.use_proxy_anchor = False          # softmax가 ProxyAnchor를 대체
            self.proxy_anchor_loss = None
            self.proxy_lambda = 0.0
            self.softmax_head = SoftmaxHead(
                embedding_size=self.projection_dim,   # proxy와 동일 D
                max_classes=int(config.training.num_experiences),
                mode=self.loss_head,
                scale=float(getattr(config.training, 'softmax_scale', 16.0)),
            ).to(device)
            if self.verbose:
                print(f"[COCONUT] Loss head = {self.loss_head} (D={self.projection_dim}, "
                      f"LR ratio={self.softmax_lr_ratio}x) — ProxyAnchor OFF")

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

        # ── AAVB (Activation-Adaptive View-Batch Replay) — plan §L. OFF=byte-identical ──
        self.use_aavb = bool(getattr(config.training, 'use_aavb', False))
        self.view_batch_V = int(getattr(config.training, 'view_batch_V', 3))
        self.aavb_ssl = bool(getattr(config.training, 'aavb_ssl', False))
        self.aavb_adaptive = bool(getattr(config.training, 'aavb_adaptive', False))
        self.aavb_peak_window = int(getattr(config.training, 'aavb_peak_window', 10))
        self.aavb_samples_per_user_target = int(getattr(config.training, 'aavb_samples_per_user_target', 5))
        # 상호배타 (D13): use_qar / use_mrs / use_aavb 중 최대 1개만 True
        if sum([bool(self.use_qar), bool(self.use_mrs), bool(self.use_aavb)]) > 1:
            raise ValueError("use_qar / use_mrs / use_aavb 중 최대 1개만 True 가능 (plan §L D13)")
        # AAVB 비대칭 증강 (weak anchor + strong); OFF면 None
        self.aavb_weak_transform = None
        self.aavb_strong_transform = None
        if self.use_aavb:
            if self.view_batch_V < 2:
                raise ValueError(f"view_batch_V must be >= 2, got {self.view_batch_V}")
            from coconut.data.transforms import get_aavb_weak_transform, get_aavb_strong_transform
            self.aavb_weak_transform = get_aavb_weak_transform(
                imside=config.dataset.height, channels=config.dataset.channels)
            self.aavb_strong_transform = get_aavb_strong_transform(
                imside=config.dataset.height, channels=config.dataset.channels)
            if self.verbose:
                print(f"[COCONUT] AAVB ON: V={self.view_batch_V}, ssl={self.aavb_ssl}, "
                      f"adaptive={self.aavb_adaptive}, peak_W={self.aavb_peak_window}")

        # === 오픈셋 관련 초기화 ===
        self.openset_enabled = hasattr(config, 'openset') and config.openset.enabled

        if self.openset_enabled:
            self.openset_config = config.openset
            self._first_calibration_done = False

            if self.verbose:
                print(f" Open-set mode: MAX")

            # Rejection gate 모드 설정
            self.rejection_gate = getattr(config.openset, 'rejection_gate', 'top1_only')
            self.ncm.rejection_gate = self.rejection_gate
            self.use_snorm = getattr(config.openset, 'use_snorm', False)
            if self.verbose:
                print(f" Rejection gate: {self.rejection_gate}")

            # ThresholdCalibrator 초기화 (cosine-only)
            _cal_verbose = self.verbose and config.openset.verbose_calibration

            if self.rejection_gate == 'top1_margin':
                # 이중 게이트: cosine용 + margin용 calibrator 각각 생성
                self.threshold_calibrator_cos = ThresholdCalibrator(
                    mode='cosine',
                    threshold_mode=config.openset.threshold_mode,
                    target_far=config.openset.target_far,
                    clip_range=(-1.0, 1.0),
                    min_samples=10,
                    verbose=_cal_verbose
                )
                self.threshold_calibrator_margin = ThresholdCalibrator(
                    mode='cosine',
                    threshold_mode=config.openset.threshold_mode,
                    target_far=config.openset.target_far,
                    clip_range=(0.0, 2.0),
                    min_samples=10,
                    verbose=_cal_verbose
                )
                # 레거시 호환: 단일 calibrator도 cosine용으로 alias
                self.threshold_calibrator = self.threshold_calibrator_cos
            else:
                # 단일 calibrator (top1_only). S-norm 켜져 있으면 z-score 범위가
                # [-1,1]을 벗어날 수 있으므로 clip 비활성화.
                cal_clip_range = None if self.use_snorm else (-1.0, 1.0)

                self.threshold_calibrator = ThresholdCalibrator(
                    mode='cosine',
                    threshold_mode=config.openset.threshold_mode,
                    target_far=config.openset.target_far,
                    clip_range=cal_clip_range,
                    min_samples=10,
                    verbose=_cal_verbose
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
            if self.rejection_gate == 'top1_margin':
                self.ncm.set_thresholds(tau_cos=initial_tau, tau_margin=0.0)
                self.ncm.tau_s = initial_tau  # 레거시 호환
            else:
                # top1_only
                self.ncm.set_thresholds(tau_s=initial_tau, tau_cos=initial_tau)

            if self.verbose:
                print(f"[TARGET] FAR Target mode enabled")
                print(f"   Target FAR: {config.openset.target_far*100:.1f}%")
                print(f"   Initial τ_s: {initial_tau}")
        else:
            self.registered_users = set()
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


    # ────────────────────────────────────────────────────────────────────
    # Projection helper (now a thin pass-through)
    # ────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _extract_features_projected(self, paths, channels, batch_size=64):
        """Pass-through wrapper for openset.extract_features.

        After Commit 4 hotfix, projection is applied inside model.getFeatureCode
        (monkey-patched in __init__ when use_projection_head=True). So
        extract_features() already returns projected features automatically.
        This helper exists only to keep the call sites in this file
        consistent and to centralise any future feature-side processing.
        """
        return extract_features(
            self.model, paths, self.test_transform, self.device,
            batch_size=batch_size, channels=channels,
        )

    def _create_optimizer_with_grouped_params(self, include_proxies=True):
        """ 파라미터 그룹별로 다른 학습률 적용한 옵티마이저 생성

        When ProjectionWrappedModel is used, self.model.ccnet and
        self.model.projection are accessed explicitly so the backbone and
        projection groups stay disjoint. When projection is off,
        self.model is the raw ccnet and self.model.parameters() is the
        backbone group.
        """
        param_groups = []

        # 백본 파라미터 그룹 (사전학습된 CCNet)
        if self.projection is not None:
            # Wrapper: backbone = wrapper.ccnet.parameters() only
            backbone_params = list(self.model.ccnet.parameters())
        else:
            # No wrapper: backbone = full model
            backbone_params = list(self.model.parameters())

        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': self.config.training.learning_rate,
                'name': 'backbone'
            })
            if self.verbose:
                print(f"️ Backbone LR: {self.config.training.learning_rate:.6f}")

        # Projection 파라미터 그룹 (선택적: PCA-init linear projection)
        if self.projection is not None:
            proj_lr = self.base_lr * self.projection_lr_ratio
            # Use wrapper's projection sub-module for parameter list — same
            # object as self.projection since wrapper stores it as a submodule.
            param_groups.append({
                'params': list(self.model.projection.parameters()),
                'lr': proj_lr,
                'name': 'projection'
            })
            if self.verbose:
                print(f"[COCONUT] Projection LR: {proj_lr:.6f} ({self.projection_lr_ratio}x)")

        # 프록시 파라미터 그룹 (새로 초기화된 프록시)
        if include_proxies and self.use_proxy_anchor and hasattr(self, 'proxy_anchor_loss') and self.proxy_anchor_loss.proxies is not None:
            param_groups.append({
                'params': [self.proxy_anchor_loss.proxies],
                'lr': self.base_lr * self.proxy_lr_ratio,
                'name': 'proxies'
            })
            if self.verbose:
                print(f"[COCONUT] Proxies LR: {self.base_lr * self.proxy_lr_ratio:.6f} ({self.proxy_lr_ratio}x)")

        # Softmax head 그룹 (loss-head ablation; proxy와 동등하게 별도 LR로 학습).
        # W가 pre-allocated이라 include_proxies와 무관하게 항상 포함 → optimizer state 보존.
        if getattr(self, 'softmax_head', None) is not None:
            param_groups.append({
                'params': list(self.softmax_head.parameters()),
                'lr': self.base_lr * self.softmax_lr_ratio,
                'name': 'softmax_head'
            })
            if self.verbose:
                print(f"[COCONUT] Softmax head LR: {self.base_lr * self.softmax_lr_ratio:.6f} "
                      f"({self.softmax_lr_ratio}x)")

        return optim.Adam(param_groups)

    def _recreate_optimizer_with_proxies(self, old_proxy_state=None, n_old=0):
        """프록시 추가 시 옵티마이저 재생성 + 기존 state 보존"""
        if not self.use_proxy_anchor or self.proxy_anchor_loss.proxies is None:
            return

        current_num_proxies = self.proxy_anchor_loss.num_classes
        old_scheduler_state = None  # scope 누수 방지: 블록 밖에서 초기화

        if current_num_proxies != self.last_num_proxies:
            # 백본 state 저장 (파라미터 객체가 동일하므로 복원 가능)
            # When wrapped, iterate ccnet's params explicitly to skip projection.
            backbone_iter = (self.model.ccnet.parameters()
                             if self.projection is not None
                             else self.model.parameters())
            old_model_states = {}
            for param in backbone_iter:
                if param in self.optimizer.state and self.optimizer.state[param]:
                    old_model_states[param] = {
                        k: v.clone() if torch.is_tensor(v) else v
                        for k, v in self.optimizer.state[param].items()
                    }

            # Projection state 저장 (있는 경우)
            old_projection_states = {}
            if self.projection is not None:
                for param in self.projection.parameters():
                    if param in self.optimizer.state and self.optimizer.state[param]:
                        old_projection_states[param] = {
                            k: v.clone() if torch.is_tensor(v) else v
                            for k, v in self.optimizer.state[param].items()
                        }

            # 스케줄러 step 위치 저장
            old_scheduler_state = self.scheduler.state_dict() if hasattr(self, 'scheduler') else None

            # 새 옵티마이저 생성
            self.optimizer = self._create_optimizer_with_grouped_params(include_proxies=True)

            # 백본 state 복원
            for param, state in old_model_states.items():
                self.optimizer.state[param] = state

            # Projection state 복원
            for param, state in old_projection_states.items():
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

            # scheduler 재생성: optimizer 교체 시 항상 수행 (n_old 무관)
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

    def train_experience(self, user_id: int, image_paths: List[str], labels: List[int]) -> Dict:
        """하나의 experience (한 명의 사용자) 학습 - 오픈셋 지원."""

        if self.verbose:
            print(f"\n=== Training Experience {self.experience_count}: User {user_id} ===")

        # 프록시 추가 및 옵티마이저 재생성 (기존 state 보존)
        if self.use_proxy_anchor:
            unique_labels = set(labels)
            real_classes = list(unique_labels)

            if real_classes:
                # 새 클래스 피처 평균 계산 — proxy 초기화용 (2048D backbone 공간)
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
                                # model.getFeatureCode is monkey-patched to apply
                                # projection automatically when use_projection_head=True
                                feat = self.model.getFeatureCode(img)
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

        # Softmax head 클래스 등록 (loss-head ablation). cosine 모드면 proxy와 *동일하게*
        # feature-mean으로 컬럼 init(기하·init 동일, 손실만 다름). W는 pre-allocated이라
        # 텐서 재생성·optimizer state 보존 머신이 불필요(구조적으로 보존됨).
        if self.softmax_head is not None:
            real_classes = list(set(labels))
            new_class_ids = [c for c in real_classes
                             if c not in self.softmax_head.class_to_idx]
            feature_means = {}
            if new_class_ids and self.loss_head == 'cosine_softmax':
                self.model.eval()
                with torch.no_grad():
                    for cid in new_class_ids:
                        cls_paths = [p for p, l in zip(image_paths, labels) if l == cid]
                        feats = []
                        for p in cls_paths:
                            img = _open_with_channels(p, self.config.dataset.channels)
                            img = self.test_transform(img).unsqueeze(0).to(self.device)
                            feat = self.model.getFeatureCode(img)
                            feats.append(feat.squeeze(0))
                        feature_means[cid] = torch.stack(feats).mean(dim=0)
                self.model.train()
            self.softmax_head.add_classes(real_classes, feature_means=feature_means)

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
        self.cohort_scheduler.register(user_id)  # MRS ⓑ: 등록 순서 기록 (OFF면 eligible()=None이라 미사용)

        # AAVB(plan §L C1/D1): 같은 forward 예산(≈2·(eb+mb))을 distinct↓·V뷰↑로 재분배.
        #   OFF면 V=2·현행 크기 = byte-identical(current=eb, memory=mb).
        _eb = self.config.training.experience_batch_size
        _mb = self.config.training.memory_batch_size
        if self.use_aavb:
            V = self.view_batch_V
            _total_distinct = max(V, round(2 * (_eb + _mb) / V))
            _current_distinct = max(1, round(_eb * _total_distinct / (_eb + _mb)))
            _memory_distinct = max(1, _total_distinct - _current_distinct)
        else:
            _current_distinct = _eb
            _memory_distinct = _mb

        #  현재 사용자 데이터를 current_distinct만큼 증강
        augmented_current_paths, augmented_current_labels = repeat_and_augment_data(
            train_paths, train_labels, _current_distinct
        )

        # 증강된 현재 사용자 데이터셋 생성 (AAVB면 비대칭 V뷰, OFF면 dual-view)
        current_dataset = MemoryDataset(
            paths=augmented_current_paths,
            labels=augmented_current_labels,
            transform=self.train_transform,
            train=True,
            channels=self.config.dataset.channels,
            use_aavb_views=self.use_aavb,
            n_views=self.view_batch_V,
            weak_transform=self.aavb_weak_transform,
            strong_transform=self.aavb_strong_transform,
        )

        # 학습 통계
        loss_avg = AverageMeter('Loss')

        # QAR: tail user rehab 샘플링 (experience당 1회)
        rehab_dataset = None
        if self.use_qar:
            rehab_paths, rehab_labels = self._qar_sample_rehab_batch()
            if rehab_paths:
                rehab_dataset = MemoryDataset(
                    paths=rehab_paths,
                    labels=rehab_labels,
                    transform=self.train_transform,
                    train=True,
                    channels=self.config.dataset.channels
                )
                if self.verbose:
                    n_tail = self._qar_rehab_log[-1]['n_tail'] if self._qar_rehab_log else 0
                    print(f"[QAR] Rehab: {n_tail} tail users, "
                          f"{len(rehab_paths)} samples injected")

        # MRS: 이번 experience의 replay 대상 제한 (ⓑ cohort 간격 ∪ ⓒ 위험 override).
        # OFF면 None → memory_buffer.sample(n, None) = 현행 균등 (byte-identical).
        mrs_eligible = self._mrs_eligible_classes() if self.use_mrs else None

        # SCR 논문 방식: epoch당 여러 iteration
        self.model.train()

        for epoch in range(self.config.training.epochs_per_experience):
            epoch_loss = 0

            for iteration in range(self.config.training.iterations_per_epoch):

                #  증강된 현재 데이터를 전체 사용 (이미 experience_batch_size로 맞춰짐)
                current_subset = current_dataset

                # 메모리에서 샘플링 (AAVB면 memory_distinct, OFF면 memory_batch_size)
                if len(self.memory_buffer) > 0:
                    # Clipping: 실제 저장 수를 초과하지 않도록 제한 (초반 과적합 방지)
                    effective_memory_batch = min(
                        _memory_distinct,
                        len(self.memory_buffer)
                    )
                    memory_paths, memory_labels, _ = self.memory_buffer.sample(
                        effective_memory_batch,
                        eligible_class_ids=mrs_eligible,  # MRS: OFF면 None=현행 균등
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
                            channels=self.config.dataset.channels,
                            use_aavb_views=self.use_aavb,
                            n_views=self.view_batch_V,
                            weak_transform=self.aavb_weak_transform,
                            strong_transform=self.aavb_strong_transform,
                        )

                        datasets = [current_subset, memory_dataset]
                        if rehab_dataset is not None:
                            datasets.append(rehab_dataset)
                        combined_dataset = ConcatDataset(datasets)
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

                    # 뷰 일반화: dual-view면 [view1,view2](n_views=2=현행),
                    #   AAVB면 [weak, strong, ...](n_views=V). 단일뷰면 [data].
                    views = list(data) if isinstance(data, (list, tuple)) else [data]
                    n_views = len(views)

                    x = torch.cat(views, dim=0).to(self.device, non_blocking=True)
                    batch_labels = batch_labels.to(self.device, non_blocking=True)

                    self.optimizer.zero_grad()

                    # Forward: CCNet feature extraction (V뷰 augmentation 효과)
                    # self.model.forward is monkey-patched to apply projection automatically
                    # when use_projection_head=True.
                    features_all = self.model(x)

                    # ProxyAnchorLoss
                    if self.use_proxy_anchor and self.proxy_anchor_loss.proxies is not None:
                        all_labels = batch_labels.repeat(n_views)
                        loss_proxy = self.proxy_anchor_loss(features_all, all_labels)
                        loss = self.proxy_lambda * loss_proxy

                        # MRS ⓐ: dual-view consistency (features_all[:B]=view1, [B:]=view2).
                        # w_ssl==0이면 ssl_loss=None → 미추가 = byte-identical.
                        # (AAVB one-to-many KL은 Phase 3에서 별도 V뷰 경로로 추가 — 여기선 2뷰만)
                        if self.ssl_loss is not None and self.w_ssl > 0 and n_views == 2:
                            loss = loss + self.w_ssl * self.ssl_loss(
                                features_all[:batch_size], features_all[batch_size:]
                            )

                        if iteration == 0 and epoch == 0:
                            # compact 모드용: loss 저장
                            self._last_curriculum = {
                                'loss_proxy': loss_proxy.item()
                            }
                            if self.verbose:
                                num_users = len(self.registered_users)
                                print(f"[Loss] users={num_users}, ProxyAnchor={loss_proxy.item():.4f}")
                    elif self.softmax_head is not None:
                        # Loss-head ablation: ProxyAnchor 대신 softmax CE (cosine/vanilla).
                        # features_all=[2B,D] dual-view → labels.repeat(2)로 두 뷰 모두 분류.
                        # softmax_lambda는 기본 proxy_lambda와 동일 → matched (λ confound 제거).
                        loss = self.softmax_lambda * self.softmax_head(features_all, batch_labels.repeat(n_views))
                        if iteration == 0 and epoch == 0:
                            self._last_curriculum = {'loss_softmax': loss.item()}
                            if self.verbose:
                                print(f"[Loss] users={len(self.registered_users)}, "
                                      f"{self.loss_head}(λ={self.softmax_lambda})={loss.item():.4f}")
                    else:
                        # ProxyAnchor 비활성화 (L_naive, L_replay, no_proxy variants):
                        # grad-connected zero loss로 loss.backward()가 작동하도록.
                        loss = features_all.sum() * 0.0

                    loss_avg.update(loss.item(), batch_size)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    if self.use_proxy_anchor and self.proxy_anchor_loss.proxies is not None:
                        torch.nn.utils.clip_grad_norm_(self.proxy_anchor_loss.proxies, max_norm=1.0)
                    elif self.softmax_head is not None:
                        torch.nn.utils.clip_grad_norm_(self.softmax_head.parameters(), max_norm=1.0)

                    self.optimizer.step()

                    epoch_loss += loss.item()

            avg_loss = epoch_loss / self.config.training.iterations_per_epoch
            if epoch == 0:
                self._first_epoch_loss = avg_loss
            self._last_epoch_loss = avg_loss

            # 에포크 진행률과 평균 손실은 항상 표시해 학습 상황을 바로 확인 가능하게 함
            print(f"  [에포크 {epoch+1}/{self.config.training.epochs_per_experience}] 평균 손실: {avg_loss:.4f}")

        # 메모리 버퍼 업데이트
        self.memory_buffer.update_from_dataset(train_paths, train_labels)
        if self.verbose:
            print(f"Memory buffer size after update: {len(self.memory_buffer)}")

        # NCM 업데이트
        self._update_ncm()

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

                _eval_entry = {
                    'experience': self.experience_count,
                    'num_users': len(self.registered_users),
                    'tau_s': self.ncm.tau_s,
                    'metrics': metrics
                }
                if self.rejection_gate == 'top1_margin':
                    _eval_entry['tau_cos'] = self.ncm.tau_cos
                    _eval_entry['tau_margin'] = self.ncm.tau_margin
                self.evaluation_history.append(_eval_entry)

                if self.verbose:
                    print("="*60 + "\n")

                # Compact 출력: 한국어 설명 포함 핵심 지표
                if not self.verbose:
                    n_train = len(self.train_data.get(user_id, ([], []))[0]) if self.openset_enabled else len(image_paths)
                    n_probe = len(self.probe_data.get(user_id, ([], []))[0]) if self.openset_enabled else 0
                    n_proxies = self.proxy_anchor_loss.num_classes if self.use_proxy_anchor else 0

                    # Line 1: Experience header
                    print(f"\n[Exp {self.experience_count:03d}] User {user_id} | Train={n_train}, Probe={n_probe} | Proxies: {n_proxies}")

                    # Line 2: Loss
                    curriculum = getattr(self, '_last_curriculum', {})
                    loss_proxy = curriculum.get('loss_proxy', 0.0)
                    first_loss = getattr(self, '_first_epoch_loss', 0)
                    last_loss = getattr(self, '_last_epoch_loss', 0)
                    epochs = self.config.training.epochs_per_experience
                    loss_line = f"  Loss: {first_loss:.4f} -> {last_loss:.4f} ({epochs}ep) | ProxyAnchor={loss_proxy:.4f}"
                    print(loss_line)

                    # (Step 8 출력에서 FNIR@1/5/10%를 CI와 함께 이미 찍음 — 중복 제거)

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

    # ================================================================
    # QAR (Quality-Aware Replay) — tail user 식별 및 rehab 샘플링
    # ================================================================

    def _qar_update_scores(self, class_scores: Dict[int, float]):
        """
        평가 시 계산된 per-class genuine score를 QAR에 저장.
        _evaluate_openset()에서 호출됨.

        Args:
            class_scores: {user_id: mean_genuine_cosine}
        """
        # numpy.int64 등 non-serializable key/value를 Python 기본형으로 변환
        self._qar_class_scores.update({int(k): float(v) for k, v in class_scores.items()})
        for k, v in class_scores.items():  # MRS ⓒ: per-user raw-cosine 궤적 누적
            self._qar_score_history.setdefault(int(k), []).append(float(v))

    def _mrs_eligible_classes(self):
        """MRS ON: 이번 experience의 replay 대상 클래스 = ⓑ cohort 간격 ∪ ⓒ 위험 override.
        ⓑ가 None(recall_interval<=1 또는 warmup 전)이면 None 반환 → 현행 균등(byte-identical)."""
        eligible = self.cohort_scheduler.eligible(self.experience_count)
        if eligible is None:
            return None
        at_risk = set()
        floor = self._qar_raw_floor
        if floor is not None:
            # ⓒ: raw-cosine 궤적이 floor 아래로 떨어질 것으로 예측되는 등록 사용자 강제 포함
            self.decay_predictor.floor = float(floor)
            cands = [u for u in self._qar_score_history
                     if u in self.memory_buffer.buffer_groups]
            at_risk = self.decay_predictor.at_risk(
                _HistTracker(self._qar_score_history), candidate_ids=cands
            )
        return set(eligible) | set(at_risk)

    def _qar_identify_tail_users(self) -> List[int]:
        """
        현재 tau_cos + rehab_margin 이하인 tail user를 식별.

        Returns:
            tail user ID 리스트
        """
        if not self._qar_class_scores:
            return []

        # ⚠️ QAR floor는 raw cosine 단위여야 함: _qar_class_scores는 raw cosine(mean genuine)인데
        # NCM tau_cos는 use_snorm 시 per-class z-score → 단위 불일치(전원 tail 오판정) 버그.
        # _calibrate_threshold가 산출한 raw-cosine impostor floor를 사용.
        floor = self._qar_raw_floor
        if floor is None:
            return []

        threshold = floor + self.rehab_margin
        tail_users = [
            uid for uid, score in self._qar_class_scores.items()
            if score < threshold and uid in self.memory_buffer.buffer_groups
        ]
        return tail_users

    def _qar_sample_rehab_batch(self) -> Tuple[List[str], List[int]]:
        """
        Tail user들의 메모리에서 rehab 샘플을 추출.

        Returns:
            (rehab_paths, rehab_labels) — 빈 리스트면 rehab 없음
        """
        # Warmup 체크
        if self.experience_count < self.qar_warmup_users:
            return [], []

        tail_users = self._qar_identify_tail_users()
        if not tail_users:
            return [], []

        rehab_paths = []
        rehab_labels = []

        for uid in tail_users:
            if uid not in self.memory_buffer.buffer_groups:
                continue
            user_buffer = self.memory_buffer.buffer_groups[uid]
            n_available = len(user_buffer.buffer)
            n_sample = min(self.rehab_samples_per_user, n_available)
            if n_sample == 0:
                continue

            paths, labels, _ = user_buffer.sample(n_sample)
            rehab_paths.extend(paths)
            rehab_labels.extend(labels)

        # 진단 로그 (QAR raw-cosine floor 기준)
        floor = self._qar_raw_floor if self._qar_raw_floor is not None else 0.0
        self._qar_rehab_log.append({
            'exp': self.experience_count,
            'raw_floor': float(floor),
            'threshold': float(floor + self.rehab_margin),
            'n_tail': len(tail_users),
            'tail_ids': [int(uid) for uid in tail_users],
            'n_rehab_samples': len(rehab_paths),
        })

        return rehab_paths, rehab_labels

    @torch.no_grad()
    def _calibrate_threshold(self):
        """임계치 캘리브레이션 (동적 비율)"""

        seed = getattr(self.config.training, 'seed', 42)
        img_size = self.config.dataset.height
        channels = self.config.dataset.channels

        if self.verbose:
            print(f" Using MAX scores for calibration ({self.rejection_gate})")
            print(f"\nExtracting Unknown Dev scores for FPIR calibration...")

        unknown_dev_file = getattr(self.config.dataset, 'unknown_dev_file', None)

        # === top1_margin 이중 게이트 캘리브레이션 ===
        if self.rejection_gate == 'top1_margin':
            s_impostor_cos = np.array([])
            s_impostor_margin = np.array([])
            s_genuine_cos = np.array([])
            s_genuine_margin = np.array([])

            # Unknown dev → dual gate scores
            if unknown_dev_file and str(unknown_dev_file) != 'None':
                from coconut.openset.utils import load_paths_labels_excluding
                unk_paths, _ = load_paths_labels_excluding(str(unknown_dev_file), self.registered_users)
                if len(unk_paths) > MAX_UNK_CALIB_SAMPLES:
                    rng = np.random.RandomState(seed)
                    idx = rng.choice(len(unk_paths), MAX_UNK_CALIB_SAMPLES, replace=False)
                    unk_paths = [unk_paths[i] for i in idx]
                if unk_paths:
                    unk_feats = self._extract_features_projected(
                        unk_paths, channels=channels
                    )
                    if len(unk_feats) > 0:
                        unk_tensor = torch.from_numpy(unk_feats).to(self.device)
                        unk_dual = self.ncm.compute_dual_gate_scores(unk_tensor)
                        s_impostor_cos = unk_dual['cosine_max'].cpu().numpy()
                        s_impostor_margin = unk_dual['margin'].cpu().numpy()
                if self.verbose:
                    print(f"   Unknown Dev: {len(s_impostor_cos)} scores (top1_margin)")
            else:
                print("  [WARN] unknown_dev_file not set. τ calibration skipped.")

            # Genuine → dual gate scores
            all_probe_paths = []
            for uid, (paths, labels) in self.probe_data.items():
                all_probe_paths.extend(paths)
            if all_probe_paths:
                gen_feats = self._extract_features_projected(
                    all_probe_paths, channels=channels
                )
                if len(gen_feats) > 0:
                    gen_tensor = torch.from_numpy(gen_feats).to(self.device)
                    gen_dual = self.ncm.compute_dual_gate_scores(gen_tensor)
                    s_genuine_cos = gen_dual['cosine_max'].cpu().numpy()
                    s_genuine_margin = gen_dual['margin'].cpu().numpy()

            if self.verbose:
                print(f"   Genuine (probe): {len(s_genuine_cos)} scores")
                print(f"   Impostor (unknown_dev): {len(s_impostor_cos)} scores")

            # DET curve용 스코어 캐시 (cosine 기준)
            self._last_genuine_scores = s_genuine_cos.copy() if len(s_genuine_cos) > 0 else np.array([])
            self._last_impostor_scores = s_impostor_cos.copy() if len(s_impostor_cos) > 0 else np.array([])

            # Joint 캘리브레이션: margin τ 먼저, cosine τ는 conditional
            if len(s_impostor_cos) >= 10:
                old_tau_cos = None if not self._first_calibration_done else self.ncm.tau_cos
                old_tau_margin = None if not self._first_calibration_done else self.ncm.tau_margin
                target_far = self.openset_config.target_far

                # Step 1: margin τ — 독립 캘리브레이션
                result_margin = self.threshold_calibrator_margin.calibrate(
                    genuine_scores=s_genuine_margin,
                    impostor_scores=s_impostor_margin,
                    old_tau=old_tau_margin
                )
                tau_margin_new = result_margin['tau_smoothed']

                # Step 2: cosine τ — margin gate 통과한 impostor 기준 conditional
                margin_pass = s_impostor_cos[s_impostor_margin >= tau_margin_new]
                if len(margin_pass) >= 10:
                    p_margin_pass = np.mean(s_impostor_margin >= tau_margin_new)
                    conditional_rate = min(target_far / max(p_margin_pass, 1e-6), 1.0)
                    tau_cos_raw = np.quantile(margin_pass, 1 - conditional_rate)
                else:
                    # margin이 대부분 걸러서 충분한 샘플 없음 → cosine만으로
                    tau_cos_raw = np.quantile(s_impostor_cos, 1 - target_far)

                # cosine τ에 EMA smoothing 적용
                tau_cos_new = self.threshold_calibrator_cos.smooth_tau(old_tau_cos, tau_cos_raw)
                self.threshold_calibrator_cos.tau_s_current = tau_cos_new

                self._first_calibration_done = True

                self.ncm.set_thresholds(
                    tau_cos=tau_cos_new,
                    tau_margin=tau_margin_new
                )
                self.ncm.tau_s = tau_cos_new  # 레거시 호환
                self._qar_raw_floor = float(tau_cos_new)  # top1_margin: cosine τ = raw-cosine floor

                # Joint FPIR 검증
                joint_pass = (s_impostor_cos >= tau_cos_new) & (s_impostor_margin >= tau_margin_new)
                joint_fpir = np.mean(joint_pass)
                joint_frr = 0.0
                if len(s_genuine_cos) > 0:
                    gen_pass = (s_genuine_cos >= tau_cos_new) & (s_genuine_margin >= tau_margin_new)
                    joint_frr = np.mean(~gen_pass)

                if self.verbose:
                    print(f"Dual Gate Calibration (Joint)  "
                          f"[unknown_dev_file 기반, FPIR=비등록자 수락율, FRR=등록자 거부율]")
                    print(f"   Target FPIR: {target_far*100:.1f}%")
                    print(f"   τ_cos: {tau_cos_new:.4f}, τ_margin: {tau_margin_new:.4f}")
                    print(f"   Joint FPIR: {joint_fpir*100:.2f}%, Joint FRR: {joint_frr*100:.2f}%")
            else:
                print("  [WARN] Not enough unknown_dev samples for calibration")

            return  # 이중 게이트 캘리브레이션 완료

        # === 기존 단일 threshold 캘리브레이션 (top1_only) ===
        s_impostor = np.array([])

        if unknown_dev_file and str(unknown_dev_file) != 'None':
            if self.use_snorm:
                # S-norm: feature 직접 추출하여 per-class cohort 통계 계산
                from coconut.openset.utils import load_paths_labels_excluding
                unk_paths, _ = load_paths_labels_excluding(str(unknown_dev_file), self.registered_users)
                if len(unk_paths) > MAX_UNK_CALIB_SAMPLES:
                    rng = np.random.RandomState(seed)
                    idx = rng.choice(len(unk_paths), MAX_UNK_CALIB_SAMPLES, replace=False)
                    unk_paths = [unk_paths[i] for i in idx]
                if unk_paths:
                    unk_feats = self._extract_features_projected(
                        unk_paths, channels=channels
                    )
                    if len(unk_feats) > 0:
                        unk_tensor = torch.from_numpy(unk_feats).to(self.device)
                        # Pass A: raw cosine으로 per-class cohort μ, σ 계산
                        self.ncm.snorm_enabled = False
                        raw_unk = self.ncm.forward(unk_tensor, apply_snorm=False)  # (N, C)
                        cohort_mu_dict = {}
                        cohort_sigma_dict = {}
                        for c in self.ncm.class_means_dict.keys():
                            col = raw_unk[:, c]
                            cohort_mu_dict[c] = float(col.mean().item())
                            cohort_sigma_dict[c] = float(col.std().item())
                        self.ncm.set_cohort_stats(cohort_mu_dict, cohort_sigma_dict)
                        if self.verbose:
                            _mu_arr = np.array(list(cohort_mu_dict.values()))
                            _sg_arr = np.array(list(cohort_sigma_dict.values()))
                            print(f"   S-norm cohort ({len(cohort_mu_dict)} classes): "
                                  f"cohort_mu={_mu_arr.mean():.4f}+/-{_mu_arr.std():.4f}, "
                                  f"cohort_sigma={_sg_arr.mean():.4f}+/-{_sg_arr.std():.4f}")
                        # Pass B: S-norm 적용된 z-score max
                        snorm_scores = self.ncm.forward(unk_tensor)  # snorm_enabled=True
                        registered_ids_local = sorted(self.ncm.class_means_dict.keys())
                        s_impostor = snorm_scores[:, registered_ids_local].max(dim=1).values.cpu().numpy()
                        # QAR용 raw-cosine impostor floor (tau_cos는 z-score라 QAR 단위에 못 씀)
                        _raw_imp_max = raw_unk[:, registered_ids_local].max(dim=1).values.cpu().numpy()
                        self._qar_raw_floor = float(np.quantile(_raw_imp_max, 1 - self.openset_config.target_far))
                    else:
                        s_impostor = np.array([])
                else:
                    s_impostor = np.array([])
            else:
                s_impostor = extract_scores_impostor_unknown(
                    self.model, self.ncm,
                    str(unknown_dev_file),
                    self.registered_users,
                    self.test_transform, self.device,
                    max_eval=MAX_UNK_CALIB_SAMPLES,
                    channels=channels
                )
            if self.verbose:
                snorm_tag = " (S-norm)" if self.use_snorm else ""
                print(f"   Unknown Dev: {len(s_impostor)} scores from {unknown_dev_file}"
                      + snorm_tag)
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
            channels=channels
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
            self.ncm.set_thresholds(tau_s=new_tau, tau_cos=new_tau)
            if not self.use_snorm:
                self._qar_raw_floor = float(new_tau)  # non-snorm: tau_cos가 곧 raw-cosine floor

            if self.verbose:
                print(f"FPIR Target Results:")
                print(f"   Target FPIR: {self.openset_config.target_far*100:.1f}%")
                print(f"   Achieved FPIR: {result.get('current_far', 0)*100:.1f}%")
                print(f"   Threshold τ: {result['tau_smoothed']:.4f}")
        else:
            print("  [WARN] Not enough unknown_dev samples for calibration")

    @staticmethod
    def _bootstrap_fnir_ci(mated_max, mated_rank1, nonmated_max,
                           target_fpir=0.01, n_iter=500, seed=42):
        """
        Bootstrap 95% CI for FNIR@target_fpir (single-threshold mode).

        매 iter마다 mated(쌍)와 nonmated를 복원 추출 →
        τ = quantile(nonmated, 1-target_fpir) →
        FNIR = P(mated_max < τ  OR  rank1 miss)
        """
        m = len(mated_max)
        nm = len(nonmated_max)
        if m == 0 or nm == 0:
            return (float('nan'), float('nan'))

        rng = np.random.RandomState(seed + int(target_fpir * 1e6))
        fnirs = np.empty(n_iter, dtype=np.float64)
        for i in range(n_iter):
            nm_idx = rng.randint(0, nm, size=nm)
            m_idx = rng.randint(0, m, size=m)
            tau = np.quantile(nonmated_max[nm_idx], 1 - target_fpir)
            det_fail = mated_max[m_idx] < tau
            id_fail = ~mated_rank1[m_idx]
            fnirs[i] = np.mean(det_fail | id_fail)
        return (float(np.percentile(fnirs, 2.5)),
                float(np.percentile(fnirs, 97.5)))

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

        self.model.eval()

        if self.verbose:
            print("\n  [Su et al.] Open-set Evaluation (FNIR@FPIR protocol):")

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
        mated_feats = self._extract_features_projected(
            mated_paths, channels=channels, batch_size=64
        )

        nonmated_feats = self._extract_features_projected(
            nonmated_paths, channels=channels, batch_size=64
        )

        # ========================================
        # Step 4: NCM 스코어 계산
        # ========================================
        registered_ids = sorted(self.ncm.class_means_dict.keys())

        # 이중 게이트용 margin 저장소
        nonmated_margins = np.array([])
        mated_margins_arr = np.array([])
        mated_cosine_max = np.array([])

        # 4a. Non-mated probe → max score per probe (FPIR 임계치 결정용)
        nonmated_max_scores = np.array([])
        if len(nonmated_feats) > 0:
            nonmated_tensor = torch.from_numpy(nonmated_feats).to(self.device)
            if self.rejection_gate == 'top1_margin':
                nm_dual = self.ncm.compute_dual_gate_scores(nonmated_tensor)
                nonmated_max_scores = nm_dual['cosine_max'].cpu().numpy()
                nonmated_margins = nm_dual['margin'].cpu().numpy()
            else:
                nonmated_ncm_scores = self.ncm.forward(nonmated_tensor)
                if nonmated_ncm_scores.numel() > 0:
                    registered_scores = nonmated_ncm_scores[:, registered_ids]
                    nonmated_max_scores = registered_scores.max(dim=1).values.cpu().numpy()

        # 4b. Mated probe → genuine score & rank-1 prediction
        mated_genuine_scores = []
        mated_rank1_correct = []
        mated_max_scores = []

        if len(mated_feats) > 0:
            mated_tensor = torch.from_numpy(mated_feats).to(self.device)

            if self.rejection_gate == 'top1_margin':
                # 이중 게이트: dual scores + genuine score 계산
                m_dual = self.ncm.compute_dual_gate_scores(mated_tensor)
                mated_cosine_max = m_dual['cosine_max'].cpu().numpy()
                mated_margins_arr = m_dual['margin'].cpu().numpy()
                mated_pred_ids = m_dual['pred_ids'].cpu().numpy()

                # genuine score 계산 (정답 클래스와의 cosine)
                mated_ncm_scores = self.ncm.forward(mated_tensor)
                mated_registered_scores = mated_ncm_scores[:, registered_ids]
                id_to_reg_idx = {cid: idx for idx, cid in enumerate(registered_ids)}

                for i in range(len(mated_labels)):
                    true_id = mated_labels[i]
                    if true_id in id_to_reg_idx:
                        genuine_score = mated_registered_scores[i][id_to_reg_idx[true_id]].item()
                    else:
                        genuine_score = -1.0

                    pred_id = int(mated_pred_ids[i])
                    max_score = mated_cosine_max[i]

                    mated_genuine_scores.append(genuine_score)
                    mated_max_scores.append(max_score)
                    mated_rank1_correct.append(pred_id == true_id)
            else:
                # 기존 로직 (top1_only)
                mated_ncm_scores = self.ncm.forward(mated_tensor)
                mated_registered_scores = mated_ncm_scores[:, registered_ids]
                id_to_reg_idx = {cid: idx for idx, cid in enumerate(registered_ids)}

                for i in range(len(mated_labels)):
                    true_id = mated_labels[i]
                    scores_reg = mated_registered_scores[i]

                    if true_id in id_to_reg_idx:
                        genuine_score = scores_reg[id_to_reg_idx[true_id]].item()
                    else:
                        genuine_score = -1.0

                    pred_reg_idx = scores_reg.argmax().item()
                    pred_id = registered_ids[pred_reg_idx]

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

        # Bootstrap CI를 위해 원본 스코어 배열 캐시
        _boot_nm = nonmated_max_scores.copy() if len(nonmated_max_scores) > 0 else np.array([])
        _boot_mated_max = mated_max_scores_arr.copy() if len(mated_max_scores_arr) > 0 else np.array([])
        _boot_rank1 = mated_rank1_correct.copy() if len(mated_rank1_correct) > 0 else np.array([])

        for target_fpir in target_fpirs:
            if self.rejection_gate == 'top1_margin' and len(nonmated_margins) > 0:
                # 이중 게이트: joint threshold로 정확한 FPIR 달성
                # margin τ를 고정 후 cosine τ를 sweep하여 joint FPIR = target
                tau_margin_fixed = np.quantile(nonmated_margins, 1 - target_fpir)

                # margin gate를 통과한 impostor만 대상으로 cosine τ 결정
                margin_pass_mask = nonmated_margins >= tau_margin_fixed
                if margin_pass_mask.sum() > 0:
                    cos_after_margin = nonmated_max_scores[margin_pass_mask]
                    # 이 중에서 target_fpir 비율만 최종 통과하도록 cosine τ 설정
                    # joint FPIR = P(margin pass) * P(cosine pass | margin pass) = target_fpir
                    # P(cosine pass | margin pass) = target_fpir / P(margin pass)
                    p_margin_pass = margin_pass_mask.mean()
                    conditional_rate = min(target_fpir / p_margin_pass, 1.0)
                    tau_cos = np.quantile(cos_after_margin, 1 - conditional_rate)
                else:
                    tau_cos = np.quantile(nonmated_max_scores, 1 - target_fpir)

                # 실제 달성 FPIR 검증
                impostor_pass = (nonmated_max_scores >= tau_cos) & (nonmated_margins >= tau_margin_fixed)
                achieved_fpir = np.mean(impostor_pass)

                if len(mated_genuine_scores) > 0:
                    genuine_pass = (mated_cosine_max >= tau_cos) & (mated_margins_arr >= tau_margin_fixed)
                    detection_fail = ~genuine_pass
                    identification_fail = ~mated_rank1_correct
                    is_fn = detection_fail | identification_fail

                    fnir = np.mean(is_fn)
                    tar = 1.0 - fnir
                    det_only = np.mean(detection_fail & ~identification_fail)
                    id_only = np.mean(~detection_fail & identification_fail)
                    both = np.mean(detection_fail & identification_fail)
                else:
                    fnir = 1.0
                    tar = 0.0
                    det_only = id_only = both = 0.0

                tau = tau_cos  # 대표값으로 cosine τ 저장

            elif len(nonmated_max_scores) > 0:
                # 기존 단일 threshold
                tau = np.quantile(nonmated_max_scores, 1 - target_fpir)
                achieved_fpir = np.mean(nonmated_max_scores >= tau)

                if len(mated_genuine_scores) > 0:
                    detection_fail = mated_max_scores_arr < tau
                    identification_fail = ~mated_rank1_correct
                    is_fn = detection_fail | identification_fail

                    fnir = np.mean(is_fn)
                    tar = 1.0 - fnir
                    det_only = np.mean(detection_fail & ~identification_fail)
                    id_only = np.mean(~detection_fail & identification_fail)
                    both = np.mean(detection_fail & identification_fail)
                else:
                    fnir = 1.0
                    tar = 0.0
                    det_only = id_only = both = 0.0
            else:
                tau = self.ncm.tau_s if self.ncm.tau_s is not None else 0.5
                achieved_fpir = 0.0
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

            # Bootstrap 95% CI (단일 threshold 모드에서만)
            if (self.rejection_gate != 'top1_margin'
                    and len(_boot_nm) > 0 and len(_boot_mated_max) > 0):
                ci_lo, ci_hi = self._bootstrap_fnir_ci(
                    _boot_mated_max, _boot_rank1, _boot_nm,
                    target_fpir=target_fpir, n_iter=500, seed=eval_seed,
                )
                results[f'FNIR@{fpir_key}%FPIR_ci_lo'] = ci_lo
                results[f'FNIR@{fpir_key}%FPIR_ci_hi'] = ci_hi

        # ========================================
        # Step 6: Closed-set Rank-1 + xdomain FPIR
        # ========================================
        rank1 = np.mean(mated_rank1_correct) if len(mated_rank1_correct) > 0 else 0.0
        results['Rank1'] = rank1

        # FPIR_xdom (크로스도메인)
        TRR_n = FAR_n = None
        _negref_source = str(self.config.dataset.xdomain_file)
        negref_paths, _ = load_paths_labels_from_txt(_negref_source)

        if len(negref_paths) > MAX_NEGREF_EVAL_SAMPLES:
            rng = np.random.RandomState(eval_seed + 2000)
            negref_paths = rng.choice(negref_paths, MAX_NEGREF_EVAL_SAMPLES, replace=False).tolist()

        if negref_paths:
            preds_neg = predict_batch(
                self.model, self.ncm, negref_paths, self.test_transform, self.device,
                channels=channels
            )
            TRR_n = sum(1 for p in preds_neg if p == -1) / len(preds_neg)
            FAR_n = 1 - TRR_n

        results['TRR_negref'] = TRR_n
        results['FPIR_xdom'] = FAR_n

        # ========================================
        # Step 6.5: 종합 진단 로그
        # ========================================
        if len(mated_max_scores_arr) >= 10:
            # _ms는 항상 cosine max score (진단 기준선)
            if self.rejection_gate == 'top1_margin':
                _ms = mated_cosine_max  # 이미 cosine
                _margins_diag = mated_margins_arr
            elif self.use_snorm:
                # S-norm 모드: 진단은 raw cosine으로 (시계열 비교용)
                _mt_cos = self.ncm.forward(torch.from_numpy(mated_feats).to(self.device), apply_snorm=False)
                _ms = _mt_cos[:, registered_ids].max(dim=1).values.cpu().numpy()
                _margins_diag = None
            else:
                _ms = mated_max_scores_arr  # 이미 cosine
                _margins_diag = None

            _gs = mated_genuine_scores
            _rc = mated_rank1_correct
            _labels = np.array(mated_labels)

            # --- 1. Genuine score 분포 상세 ---
            print(f"  [Genuine] cos μ={_ms.mean():.3f} σ={_ms.std():.3f} "
                  f"p5={np.percentile(_ms,5):.3f} min={_ms.min():.3f}")

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
                _all_scores = self.ncm.forward(_mt, apply_snorm=False)  # 진단: raw cosine
                _reg_scores = _all_scores[:, registered_ids]

                if _reg_scores.shape[1] >= 2:
                    _topk = _reg_scores.topk(2, dim=1)
                    _top1 = _topk.values[:, 0].cpu().numpy()
                    _top2 = _topk.values[:, 1].cpu().numpy()
                    _margin = _top1 - _top2

                    print(f"  [Margin] top1-top2 μ={_margin.mean():.3f} "
                          f"p5={np.percentile(_margin,5):.3f}")

                    # tail vs normal (QAR/진단 유지용 계산, 출력은 verbose만)
                    _tail_mask = _ms < np.percentile(_ms, 20)
                    _normal_mask = _ms >= np.percentile(_ms, 50)

                    if self.verbose and _tail_mask.sum() > 0 and _normal_mask.sum() > 0:
                        print(f"  [Tail20 vs Top50] "
                              f"tail cos={_ms[_tail_mask].mean():.3f} r1={_rc[_tail_mask].mean():.3f} | "
                              f"norm cos={_ms[_normal_mask].mean():.3f} r1={_rc[_normal_mask].mean():.3f}")

            # --- 3. Per-class Worst/Best ---
            _class_scores = {}
            for i, lbl in enumerate(_labels):
                if lbl not in _class_scores:
                    _class_scores[lbl] = []
                _class_scores[lbl].append(_ms[i])

            _class_means = {k: np.mean(v) for k, v in _class_scores.items()}

            # QAR/MRS: per-class genuine score 갱신 (MRS ⓒ도 이 궤적을 씀 → use_mrs도 포함)
            if self.use_qar or self.use_mrs:
                self._qar_update_scores(_class_means)

            _worst5 = sorted(_class_means.items(), key=lambda x: x[1])[:5]
            _best5 = sorted(_class_means.items(), key=lambda x: x[1], reverse=True)[:5]

            if self.verbose:
                for uid, score in _worst5:
                    print(f"  Worst: User {uid} -> mean_score={score:.4f} (n={len(_class_scores[uid])})")
                for uid, score in _best5:
                    print(f"  Best:  User {uid} -> mean_score={score:.4f} (n={len(_class_scores[uid])})")

            # --- 5. Impostor vs Genuine margin 비교 ---
            if len(nonmated_feats) > 0 and _margin is not None:
                _nmt = torch.from_numpy(nonmated_feats).to(self.device)

                if self.rejection_gate == 'top1_margin':
                    # 이중 게이트: compute_dual_gate_scores로 통일
                    _nm_dual = self.ncm.compute_dual_gate_scores(_nmt)
                    _nm_top1 = _nm_dual['cosine_max'].cpu().numpy()
                    _nm_margin = _nm_dual['margin'].cpu().numpy()
                else:
                    _nm_scores = self.ncm.forward(_nmt, apply_snorm=False)  # 진단: raw cosine
                    _nm_reg = _nm_scores[:, registered_ids]
                    if _nm_reg.shape[1] >= 2:
                        _nm_topk = _nm_reg.topk(2, dim=1)
                        _nm_top1 = _nm_topk.values[:, 0].cpu().numpy()
                        _nm_top2 = _nm_topk.values[:, 1].cpu().numpy()
                        _nm_margin = _nm_top1 - _nm_top2
                    else:
                        _nm_top1 = None
                        _nm_margin = None

                if _nm_top1 is not None:
                    _sep_cos = _ms.mean() - _nm_top1.mean()
                    print(f"  [Separation] gen-imp cos={_sep_cos:.3f} "
                          f"(gen μ={_ms.mean():.3f}, imp μ={_nm_top1.mean():.3f})")

                    # nonmated 분포 꼬리 (95/99 percentile) — open-set 핵심 진단
                    _nm_p95 = np.percentile(_nm_top1, 95)
                    _nm_p99 = np.percentile(_nm_top1, 99)
                    _gen_p5 = np.percentile(_ms, 5)
                    print(f"  [Tails] nonmated p95={_nm_p95:.3f} p99={_nm_p99:.3f} | "
                          f"genuine p5={_gen_p5:.3f}")

                    _tau_cos_diag = _nm_p99
                    _tau_margin_diag = np.percentile(_nm_margin, 99)
                    _fn_cos = np.mean(_ms < _tau_cos_diag)
                    _fn_margin = np.mean(_margin < _tau_margin_diag)

            # --- 6. 진단 데이터 저장 ---
            _diag_entry = {
                'exp': len(self.registered_users) - 1,
                'n_classes': self.ncm.get_num_classes(),
                'n_mated': len(_ms),
                'rank1': float(_rc.mean()),
                'gate_mode': self.rejection_gate,
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
                        'sim_tau_cosine': float(_tau_cos_diag),
                        'sim_frr_cosine': float(_fn_cos),
                        'sim_tau_margin': float(_tau_margin_diag),
                        'sim_frr_margin': float(_fn_margin),
                    })

            # 이중 게이트 메트릭 추가
            if self.rejection_gate == 'top1_margin' and _margins_diag is not None:
                _diag_entry.update({
                    'genuine_margin_mean': float(_margins_diag.mean()),
                    'genuine_margin_std': float(_margins_diag.std()),
                    'genuine_margin_min': float(_margins_diag.min()),
                    'genuine_margin_p5': float(np.percentile(_margins_diag, 5)),
                    'tau_cos': float(self.ncm.tau_cos) if self.ncm.tau_cos is not None else None,
                    'tau_margin': float(self.ncm.tau_margin) if self.ncm.tau_margin is not None else None,
                })

            if '_worst5' in dir():
                _diag_entry['worst_users'] = [(int(uid), float(sc)) for uid, sc in _worst5]
                _diag_entry['best_users'] = [(int(uid), float(sc)) for uid, sc in _best5]

            # Bootstrap CI + nonmated tail 추가
            for fpir_pct in [1, 5, 10]:
                ci_lo = results.get(f'FNIR@{fpir_pct}%FPIR_ci_lo')
                ci_hi = results.get(f'FNIR@{fpir_pct}%FPIR_ci_hi')
                if ci_lo is not None:
                    _diag_entry[f'fnir{fpir_pct}_ci'] = [ci_lo, ci_hi]
                _diag_entry[f'fnir{fpir_pct}'] = results.get(f'FNIR@{fpir_pct}%FPIR', 0)
            if _nm_top1 is not None:
                _diag_entry['nonmated_p95'] = float(np.percentile(_nm_top1, 95))
                _diag_entry['nonmated_p99'] = float(np.percentile(_nm_top1, 99))

            # QAR 진단 정보 추가
            if self.use_qar and self._qar_rehab_log:
                _qar_last = self._qar_rehab_log[-1]
                _diag_entry['qar'] = {
                    'n_tail': _qar_last['n_tail'],
                    'n_rehab_samples': _qar_last['n_rehab_samples'],
                    'threshold': _qar_last['threshold'],
                    'tail_ids': _qar_last['tail_ids'],
                }

            self._diag_history.append(_diag_entry)

            # === 10 경험마다 리포트 파일 저장 (paper_minimal_outputs=True 면 skip) ===
            _exp_num = _diag_entry['exp']
            _paper_minimal = bool(getattr(self.config.training, 'paper_minimal_outputs', False))
            if not _paper_minimal and ((_exp_num + 1) % 10 == 0 or _exp_num == 0):
                import json, os
                _save_dir = str(self.config.training.results_path)
                os.makedirs(_save_dir, exist_ok=True)

                _json_path = os.path.join(_save_dir, 'diag_history.json')
                with open(_json_path, 'w') as f:
                    import numpy as _np
                    def _json_default(o):
                        if isinstance(o, _np.integer): return int(o)
                        if isinstance(o, _np.floating): return float(o)
                        if isinstance(o, _np.ndarray): return o.tolist()
                        return str(o)
                    json.dump(self._diag_history, f, indent=2, ensure_ascii=False, default=_json_default)

                _txt_path = os.path.join(_save_dir, f'diag_report_exp{_exp_num:03d}.txt')
                with open(_txt_path, 'w') as f:
                    f.write(f"{'='*80}\n")
                    f.write(f"COCONUT Diagnostic Report - Experience {_exp_num}\n")
                    f.write(f"{'='*80}\n\n")

                    f.write(f"Column Legend:\n")
                    f.write(f"  R1      = Closed-set Rank-1 accuracy\n")
                    f.write(f"  cos_u/s = Genuine max cosine mean/std\n")
                    f.write(f"  cos_mn  = Genuine max cosine min\n")
                    f.write(f"  cos_p5  = Genuine max cosine 5th percentile\n")
                    f.write(f"  mrg_u   = Genuine top1-top2 margin mean\n")
                    f.write(f"  mrg_p5  = Genuine top1-top2 margin 5th percentile\n")
                    f.write(f"  imp_c/m = Impostor(unknown) max cosine/margin mean\n")
                    f.write(f"  sp_c/m  = Separation (genuine - impostor) cosine/margin\n")
                    f.write(f"  frr_c/m = @1%FPIR FRR simulation (detection 실패만, rank-1 miss 미포함)\n\n")

                    f.write(f"{'Exp':>4} {'Cls':>4} {'R1':>5} "
                            f"{'cos_u':>6} {'cos_s':>6} {'cos_mn':>6} {'cos_p5':>6} "
                            f"{'mrg_u':>6} {'mrg_p5':>6} "
                            f"{'imp_c':>6} {'imp_m':>6} "
                            f"{'sp_c':>6} {'sp_m':>6} "
                            f"{'frr_c':>6} {'frr_m':>6}\n")
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
                        if 'sim_frr_cosine' in d:
                            f.write(f"{d['sim_frr_cosine']:6.3f} {d['sim_frr_margin']:6.3f}")
                        else:
                            f.write(f"{'N/A':>6} {'N/A':>6}")
                        f.write("\n")

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

                    # QAR 추이
                    if any('qar' in d for d in self._diag_history):
                        f.write(f"\n\nQAR (Quality-Aware Replay) trend\n{'='*60}\n")
                        f.write(f"  [tail = per-user mean cosine < tau_cos + rehab_margin]\n")
                        for d in self._diag_history:
                            if 'qar' in d:
                                q = d['qar']
                                f.write(f"  Exp {d['exp']:3d}: "
                                        f"tail={q['n_tail']:2d} users, "
                                        f"rehab={q['n_rehab_samples']:3d} samples, "
                                        f"threshold={q['threshold']:.4f}\n")

                    # 핵심 결론
                    f.write(f"\n\nKey Summary\n{'='*60}\n")
                    latest = self._diag_history[-1]
                    if 'sim_frr_cosine' in latest and 'sim_frr_margin' in latest:
                        f.write(f"  @1%FPIR genuine FRR (detection 실패율만, rank-1 miss 미포함):\n")
                        f.write(f"    cosine: {latest['sim_frr_cosine']*100:.1f}%\n")
                        f.write(f"    margin: {latest['sim_frr_margin']*100:.1f}%\n")
                        if latest['sim_frr_margin'] < latest['sim_frr_cosine']:
                            diff = (latest['sim_frr_cosine'] - latest['sim_frr_margin']) * 100
                            f.write(f"    -> margin is {diff:.1f}%p better\n")
                        else:
                            f.write(f"    -> cosine is better\n")

                if self.verbose:
                    print(f"  [DIAG] Report saved: {_txt_path}")
                    print(f"  [DIAG] JSON saved: {_json_path}")

        # ========================================
        # Step 7: DET curve용 스코어 캐시
        # ========================================
        # DET curve용: detection 실패 판단에 max score 사용 (genuine score 아님)
        self._last_genuine_scores = mated_max_scores_arr.copy() if len(mated_max_scores_arr) > 0 else np.array([])
        self._last_impostor_scores = nonmated_max_scores.copy() if len(nonmated_max_scores) > 0 else np.array([])

        # τ 참고값 저장
        results['tau_s_current'] = self.ncm.tau_s
        if self.rejection_gate == 'top1_margin':
            results['tau_cos_current'] = self.ncm.tau_cos
            results['tau_margin_current'] = self.ncm.tau_margin

        # ========================================
        # Step 8: 결과 출력 — 성능 평가에 필요한 핵심 지표만
        # ========================================
        print(f"\n  [Eval] Gallery={self.ncm.get_num_classes()} | "
              f"mated={len(mated_paths)} nonmated={len(nonmated_paths)} | "
              f"Rank-1={rank1:.3f}")
        for fpir_pct in [1, 5, 10]:
            fnir_v = results[f'FNIR@{fpir_pct}%FPIR']
            det_v = results[f'det_fail@{fpir_pct}%']
            idf_v = results[f'id_fail@{fpir_pct}%']
            ci_lo = results.get(f'FNIR@{fpir_pct}%FPIR_ci_lo')
            ci_hi = results.get(f'FNIR@{fpir_pct}%FPIR_ci_hi')
            ci_str = f" [{ci_lo:.3f},{ci_hi:.3f}]" if ci_lo is not None else ""
            print(f"        FNIR@{fpir_pct:>2}%FPIR = {fnir_v:.3f}{ci_str}  "
                  f"(FRR={det_v:.3f}, RankMiss={idf_v:.3f})")
        if TRR_n is not None:
            print(f"        FPIR_xdom = {FAR_n:.3f}")

        # B3: legacy aliases removed; callers read canonical keys directly
        # (FNIR@1%FPIR, achieved_FPIR@1%, det_fail@1%, id_fail@1%).
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

            # getFeatureCode is monkey-patched to include projection when enabled
            features = self.model.getFeatureCode(data)

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

                # getFeatureCode is monkey-patched to include projection when enabled
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

        # When projection wrapping is on, save the underlying ccnet's
        # state_dict (not the wrapper's). The projection is saved
        # separately below via projection_state_dict so it can be
        # reconstructed even if a future load runs without projection.
        if self.projection is not None:
            model_sd = self.model.ccnet.state_dict()
        else:
            model_sd = self.model.state_dict()

        checkpoint_dict = {
            'model_state_dict': model_sd,
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

        # ProjectionHead 저장 (활성화된 경우)
        if self.projection is not None:
            try:
                checkpoint_dict['projection_state_dict'] = self.projection.state_dict()
                checkpoint_dict['projection_meta'] = {
                    'in_dim': self.projection.in_dim,
                    'out_dim': self.projection.out_dim,
                    'pca_initialised': self.projection._pca_initialised,
                }
            except Exception as e:
                print(f"Warning: Could not save projection state: {e}")

        # 오픈셋 관련 추가 저장
        if self.openset_enabled:
            try:
                checkpoint_dict['openset_data'] = {
                    'tau_s': self.ncm.tau_s,
                    'tau_cos': getattr(self.ncm, 'tau_cos', None),
                    'tau_margin': getattr(self.ncm, 'tau_margin', None),
                    'rejection_gate': getattr(self, 'rejection_gate', 'top1_only'),
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
        # When wrapped, the saved state_dict is the underlying ccnet's
        # (we strip the wrapper at save time) so load it into ccnet directly.
        if self.projection is not None:
            self.model.ccnet.load_state_dict(checkpoint['model_state_dict'])
        else:
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

        # ProjectionHead 복원 (저장된 체크포인트가 있고 현재 trainer에 projection이 켜져있는 경우)
        if 'projection_state_dict' in checkpoint and self.projection is not None:
            try:
                meta = checkpoint.get('projection_meta', {})
                ckpt_in = meta.get('in_dim')
                ckpt_out = meta.get('out_dim')
                if ckpt_in is not None and (ckpt_in != self.projection.in_dim or ckpt_out != self.projection.out_dim):
                    print(f"Warning: projection dim mismatch (ckpt {ckpt_in}->{ckpt_out} vs "
                          f"current {self.projection.in_dim}->{self.projection.out_dim}). Skipping projection restore.")
                else:
                    self.projection.load_state_dict(checkpoint['projection_state_dict'])
                    self.projection._pca_initialised = bool(meta.get('pca_initialised', False))
                    if self.verbose:
                        print(f"[OK] ProjectionHead restored ({self.projection.in_dim}->{self.projection.out_dim})")
            except Exception as e:
                print(f"Warning: Could not restore projection state: {e}")
        elif 'projection_state_dict' in checkpoint and self.projection is None:
            print("Warning: checkpoint has projection state but current trainer has projection disabled. Ignored.")

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
                self.ncm.set_thresholds(tau_s=tau_s)
            tau_cos = od.get('tau_cos')
            tau_margin = od.get('tau_margin')
            if tau_cos is not None:
                self.ncm.tau_cos = tau_cos
            if tau_margin is not None:
                self.ncm.tau_margin = tau_margin
            if self.verbose:
                if self.rejection_gate == 'top1_margin':
                    print(f"[OK] Openset data restored ({len(self.registered_users)} registered users, "
                          f"τ_cos={tau_cos}, τ_margin={tau_margin})")
                else:
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
