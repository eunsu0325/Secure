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
    # 프로젝션 헤드 설정 추가
    use_projection: bool = True
    projection_dim: int = 128  # 출력 차원만 조절 가능
    use_projection_for_ncm: bool = False  # 🍑 NCM도 512D projection 사용할지 (false=6144D, true=512D)

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
    projection_learning_rate: float = 0.0005  # 프로젝션 헤드 학습률
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

    # Herding buffer 설정
    use_herding: bool = False  # Herding buffer 사용 여부
    max_samples_per_class: int = 20  # Herding 시 클래스당 최대 샘플 수
    herding_batch_size: int = 32  # Feature extraction 배치 크기
    drift_threshold: float = 0.5  # Feature drift 감지 임계값

    # IDL+RTM Loss 설정 (Open-set representation 최적화)
    use_idl_rtm: bool = False
    idl_alpha_det: float = 1.0          # Detection loss 내부 가중치
    idl_beta_id: float = 1.0            # Identification loss 내부 가중치
    idl_gamma_rtm: float = 0.5          # RTM loss 내부 가중치
    idl_rtm_lambda: float = 0.3         # 전체 손실 내 IDL+RTM 가중치
    idl_rtm_warmup_users: int = 5       # warmup 기간 (사용자 수)
    idl_n_quantiles: int = 5            # Detection threshold quantile 후보 수
    idl_rank_target: float = 1.0        # Identification target rank
    idl_rank_gamma: float = 1.0         # Identification rank loss temperature
    idl_rank_temperature: float = 0.5   # soft rank 스케일링 (수치 안정성)
    idl_rtm_temperature: float = 0.1    # RTM softmax temperature
    idl_det_steepness: float = 10.0     # Detection sigmoid steepness
    idl_min_gallery_classes: int = 2    # 최소 gallery 클래스 수
    idl_gallery_fraction: float = 0.7   # gallery로 사용할 클래스 비율

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
    
    # TTA (Test-Time Augmentation) 설정 추가
    tta_n_views: int = 1  # 1이면 비활성화, 2~3 권장
    tta_include_original: bool = True  # 원본 포함 여부
    tta_agree_k: int = 0  # 0이면 과반 자동 계산
    tta_augmentation_strength: float = 0.5  # 증강 강도 (0.0~1.0)
    tta_aggregation: str = 'median'  # 'median' or 'mean' for scores
    
    # TTA 반복 설정
    tta_n_repeats: int = 1  # 전체 기본값 (타입별 설정이 없을 때 사용)
    tta_repeat_aggregation: str = 'median'  # 반복 간 집계 방법
    tta_verbose: bool = False  # TTA 디버깅 출력
    
    # 타입별 독립적인 TTA 반복 설정 추가
    tta_n_repeats_genuine: Optional[int] = None      # Genuine 점수용
    tta_n_repeats_between: Optional[int] = None      # Between impostor용
    
    def __post_init__(self):
        """타입별 반복 설정이 없으면 기본값 사용"""
        # Genuine: 기본값 사용
        if self.tta_n_repeats_genuine is None:
            self.tta_n_repeats_genuine = self.tta_n_repeats

        # Between: 기본값 사용
        if self.tta_n_repeats_between is None:
            self.tta_n_repeats_between = self.tta_n_repeats

