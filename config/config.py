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
    temperature: float
    test_interval: int
    checkpoint_path: Path
    results_path: Path
    gpu_ids: str
    ncm_momentum: float

    # 기본값이 있는 선택적 필드들
    batch_size: int = 128
    seed: int = 42  # 추가!

    # ProxyAnchorLoss 설정 추가
    use_proxy_anchor: bool = True
    proxy_margin: float = 0.1        # Proxy Anchor margin δ
    proxy_alpha: float = 32          # Proxy Anchor scaling α
    proxy_lr_ratio: float = 10       # 프록시 학습률 배수
    proxy_lambda: float = 0.3        # 고정 가중치 (SupCon: 0.7, ProxyAnchor: 0.3)

    curriculum_ramp_users: int = 12

    # 로그 출력 설정
    verbose: bool = False  # True: 전체 출력, False: compact 출력 (논문 지표 중심)

    # IDL+RTM Loss 설정 (Su et al., "Open-Set Biometrics")
    use_idl_rtm: bool = False
    idl_alpha_det: float = 6.0          # α: Detection sigmoid steepness (Eq. 7)
    idl_beta_id: float = 0.2            # β: Identification sigmoid steepness (Eq. 9)
    idl_gamma_rank: float = 6.0         # γ: Rank sigmoid steepness (Eq. 10)
    idl_rtm_weight: float = 4.0         # λ: L^RTM weight in L = L^IDL + λ·L^RTM (Eq. 13)
    idl_rtm_lambda: float = 0.3         # COCONUT 전체 손실 내 IDL+RTM 외부 가중치
    idl_rtm_warmup_users: int = 5       # warmup 기간 (사용자 수)
    idl_min_gallery_classes: int = 2    # 최소 gallery 클래스 수
    idl_gallery_fraction: float = 0.7   # gallery로 사용할 클래스 비율

    # DER++ (Dark Experience Replay) settings — feature distillation
    der_alpha: float = 0.0       # 0.0 = 비활성화. 권장 범위: 0.1~0.3
    der_batch_size: int = 32     # DER loss 계산용 버퍼 샘플 수
    der_warmup_users: int = 3    # DER 활성화 전 warmup 기간

    # QAR (Quality-Aware Replay) — tail user 선별 재학습
    use_qar: bool = False
    rehab_margin: float = 0.05        # tau_cos + margin 이하를 tail로 판정
    rehab_samples_per_user: int = 4   # tail user당 추가 replay 샘플 수
    qar_warmup_users: int = 10        # QAR 활성화 최소 등록 사용자 수

    # A8: diagnostic Phase 2 용 minimal-output 모드 (Drive I/O 비용 절감)
    paper_minimal_outputs: bool = False  # True 면 per-step PNG/CSV/체크포인트 저장 skip

@dataclasses.dataclass
class Openset:
    enabled: bool = True
    warmup_users: int = 10
    initial_tau: float = 0.7
    
    # 통일된 임계치 파라미터
    threshold_mode: str = 'far'
    target_far: float = 0.01              # FAR 타겟 (1%)
    threshold_alpha: float = 0.2          # EMA 계수 (기존 smoothing_alpha)
    threshold_max_delta: float = 0.03     # 최대 변화폭 (기존 max_delta)
    
    dev_ratio: float = 0.2

    # 추가 옵션
    verbose_calibration: bool = True      # 상세 출력
    
    # Score mode (Method 1: Shared Diagonal Mahalanobis)
    score_mode: str = 'cosine'          # 'cosine' | 'mahalanobis'
    var_reg_alpha: float = 1e-4         # Mahalanobis variance regularization
    mahalanobis_variant: str = 'diagonal'  # 'diagonal' | 'full_whitened' | 'projection_only'
    pca_explained_var: float = 0.99     # adaptive k 결정 기준 (full_whitened)
    pca_max_k: int = 256                # k 상한
    # Shrinkage / k ablation knobs (Phase A)
    pca_shrinkage_mode: str = 'auto'     # 'auto' | 'fixed' | 'none'
    pca_shrinkage_lambda: float = 0.1    # fixed 모드 λ
    pca_k_mode: str = 'adaptive'         # 'adaptive' | 'fixed'
    pca_fixed_k: int = 32                # fixed 모드 k
    rejection_gate: str = 'top1_only'  # 'top1_only' | 'top1_margin' (score_mode-agnostic)
    use_snorm: bool = False              # per-class Z-score normalization (S-norm)

    # GHOST (Gaussian Hypothesis Open-Set Technique, AAAI 2025)
    use_ghost: bool = False             # GHOST z-score rejection 활성화
    ghost_n_augment: int = 10           # per-class σ 추정용 이미지당 augmentation 횟수
    ghost_shrinkage_min_n: int = 10     # shrinkage 기준 (augmented feature 수)


