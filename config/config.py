# config/config.py
import dataclasses
from pathlib import Path
from typing import Optional, Dict

@dataclasses.dataclass
class Dataset:
    train_set_file: Path
    test_set_file: Path
    height: int
    width: int
    channels: int
    augmentation: bool
    negative_samples_file: Path
    num_negative_classes: int

@dataclasses.dataclass
class Model:
    architecture: str
    competition_weight: float
    use_pretrained: bool = False
    pretrained_path: Optional[Path] = None
    # 프로젝션 헤드 설정 추가
    use_projection: bool = True
    projection_dim: int = 128  # 출력 차원만 조절 가능

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

@dataclasses.dataclass
class Openset:
    enabled: bool = True
    warmup_users: int = 10
    initial_tau: float = 0.7
    
    # 통일된 임계치 파라미터
    threshold_mode: str = 'far'           # 'eer' or 'far'
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
    
    # Impostor 점수 비율 설정 추가
    impostor_ratio_between: float = 1.0   # Between impostor 비율 (100%)
    impostor_balance_total: int = 6000    # 균형 맞출 총 샘플 수
    
    def __post_init__(self):
        """타입별 반복 설정이 없으면 기본값 사용"""
        # Genuine: 기본값 사용
        if self.tta_n_repeats_genuine is None:
            self.tta_n_repeats_genuine = self.tta_n_repeats
        
        # Between: 기본값 사용
        if self.tta_n_repeats_between is None:
            self.tta_n_repeats_between = self.tta_n_repeats
        
        # Between ratio validation (should be 1.0)
        if abs(self.impostor_ratio_between - 1.0) > 1e-6:
            self.impostor_ratio_between = 1.0  # Force to 100%

@dataclasses.dataclass
class Negative:
    warmup_experiences: int = 4
    max_per_batch: int = 1
    r0: float = 0.5
    base_id: int = 1