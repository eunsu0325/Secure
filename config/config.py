# config/config.py
import dataclasses
from pathlib import Path
from typing import Optional

@dataclasses.dataclass
class Dataset:
    # 등록 대상 사용자 파일 — 내부에서 90% train/NCM + 10% probe로 자동 분리
    enroll_file: Path
    # 독립 per-user 평가용 probe — BWT/forgetting 측정 전용 (enroll_file과 겹치지 않아야 함)
    eval_probe_file: Path
    height: int
    width: int
    channels: int
    augmentation: bool
    # 크로스도메인 FPIR_xdom 평가 전용 파일 — 학습에 사용하지 않음 (예: IITD 데이터셋)
    xdomain_file: Path
    num_xdomain_classes: int
    # τ calibration용 비등록 사용자 데이터 (enroll_file 미등록 ID 앞 50% 권장)
    unknown_dev_file: Optional[Path] = None
    # FPIR_in 최종 평가용 비등록 사용자 데이터 (enroll_file 미등록 ID 뒤 50% 권장)
    unknown_test_file: Optional[Path] = None

@dataclasses.dataclass
class Model:
    architecture: str
    competition_weight: float
    use_pretrained: bool = False
    pretrained_path: Optional[Path] = None

@dataclasses.dataclass
class Training:
    # 기본값이 없는 필수 필드들
    experience_batch_size: int
    memory_batch_size: int
    num_experiences: int
    memory_size: int
    min_samples_per_class: int
    epochs_per_experience: int
    iterations_per_epoch: int
    num_workers: int
    learning_rate: float
    scheduler_step_size: int
    scheduler_gamma: float
    test_interval: int
    checkpoint_path: Path
    results_path: Path
    gpu_ids: str
    ncm_momentum: float

    # 기본값이 있는 선택적 필드들
    batch_size: int = 128
    seed: int = 42  # 추가!

    # ProxyAnchorLoss 설정
    use_proxy_anchor: bool = True
    proxy_margin: float = 0.1        # Proxy Anchor margin δ
    proxy_alpha: float = 32          # Proxy Anchor scaling α
    proxy_lr_ratio: float = 10       # 프록시 학습률 배수
    proxy_lambda: float = 1.0        # 단독 손실 가중치 (0.5/0.3은 contrastive 동시학습 잔재 → 1.0 정정)
    # Loss formulation: False=legacy (P+ only neg term), True=paper Eq. (4) canonical.
    # See Stage A ablation; legacy default preserves backward-compat with prior runs.
    use_canonical_proxy_loss: bool = False

    # ProjectionHead 설정 (2048D → projection_dim linear projection w/ PCA init)
    # use_projection_head=False (default): bypass — CCNet 2048D 그대로 다운스트림 전달
    # use_projection_head=True: enable projection at trainer init time
    use_projection_head: bool = False
    projection_dim: int = 512        # 권장: 128, 256, 512. <= 2048 (CCNet output)
    projection_lr_ratio: float = 1.0 # backbone LR 대비 projection LR 배율

    # 로그 출력 설정
    verbose: bool = False  # True: 전체 출력, False: compact 출력 (논문 지표 중심)

    # QAR (Quality-Aware Replay) — tail user 선별 재학습
    use_qar: bool = False
    rehab_margin: float = 0.05        # tau_cos + margin 이하를 tail로 판정
    rehab_samples_per_user: int = 4   # tail user당 추가 replay 샘플 수
    qar_warmup_users: int = 10        # QAR 활성화 최소 등록 사용자 수

    # Loss-head ablation 대조군 (ProxyAnchor vs softmax) — default 'proxy' = 현행
    loss_head: str = 'proxy'          # 'proxy' | 'cosine_softmax' | 'vanilla_softmax'
    softmax_lr_ratio: float = 50.0    # softmax head LR = base_lr × 이 값 (proxy와 동등 대우)
    softmax_scale: float = 32.0       # cosine_softmax logit scale s = proxy_alpha(32) matched treatment
    softmax_lambda: float = 1.0       # softmax 손실 weight = proxy_lambda(1.0) matched (단독손실 자연값)

    # MRS (Memory-risk Replay Scheduling) — 전부 default OFF = 기존 동작 byte-identical
    use_mrs: bool = False             # ⓑ cohort 간격 + ⓒ 위험 override replay 활성화
    mrs_recall_interval: int = 1      # ⓑ: 각 사용자를 R experience마다 recall (1=off=균등)
    mrs_warmup_users: int = 10        # ⓑ: 등록 사용자 < 이 수면 균등(off)
    mrs_override_cap: int = 0         # ⓒ: 한 번에 override할 최대 사용자 수 (0=무제한)
    w_ssl: float = 0.0                # ⓐ: SSL consistency loss 가중 (0=off)

    # A8: diagnostic Phase 2 용 minimal-output 모드 (Drive I/O 비용 절감)
    paper_minimal_outputs: bool = False  # True 면 per-step PNG/CSV/체크포인트 저장 skip

    # AAVB Phase 0b: 공정한 replay 예산 배분 (plan §L B3/D8)
    # False(기본)=legacy(remainder를 早등록 클래스에 front-load, byte-identical).
    # True=remainder를 무작위 클래스로 → M<N 저예산 uniform이 최신 클래스를 굶기는 편향 +
    #   M>=N에서 앞 remainder명이 +1 더 받는 편향 제거. ⚠️ AAVB 실험은 baseline 포함 전 arm에서 True.
    fair_remainder: bool = False

    # AAVB (Activation-Adaptive View-Batch Replay) — plan §L. 전부 OFF=byte-identical.
    # 상호배타: use_qar/use_mrs/use_aavb 중 최대 1개만 True(trainer가 검증, D13).
    use_aavb: bool = False            # 마스터 스위치 (C1 view-batch). False면 현행.
    view_batch_V: int = 3             # 뷰 수 V (use_aavb 시만; 1 weak + V-1 strong). VBM x3~4 검증치.
    aavb_ssl: bool = False            # C2 one-to-many KL (Phase 3). use_aavb일 때만.
    aavb_ssl_weight: float = 1.0      # C2 weight (VBM Eq4: L_sup + L_ssl = 1.0 equal).
    aavb_adaptive: bool = False       # C3 activation-state 스케줄링 (Phase 4). use_aavb일 때만.
    aavb_peak_window: int = 10        # C3 decline window-max W (D15). W >= 복습간격(N/K).
    aavb_samples_per_user_target: int = 5  # C3 K_target = memory_distinct // 이 값 (D5).

@dataclasses.dataclass
class Openset:
    enabled: bool = True
    warmup_users: int = 10
    initial_tau: float = 0.7
    
    # 통일된 임계치 파라미터
    threshold_mode: str = 'far'
    target_far: float = 0.01              # FAR 타겟 (1%)

    dev_ratio: float = 0.2

    # 추가 옵션
    verbose_calibration: bool = True      # 상세 출력

    rejection_gate: str = 'top1_only'  # 'top1_only' | 'top1_margin'
    use_snorm: bool = False              # per-class Z-score normalization (S-norm)

    # GHOST (Gaussian Hypothesis Open-Set Technique, AAAI 2025)
    use_ghost: bool = False             # GHOST z-score rejection 활성화
    ghost_n_augment: int = 10           # per-class σ 추정용 이미지당 augmentation 횟수
    ghost_shrinkage_min_n: int = 10     # shrinkage 기준 (augmented feature 수)


