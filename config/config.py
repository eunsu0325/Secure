# config/config.py
import dataclasses
from pathlib import Path
from typing import Optional, Dict  # 💎 Dict 추가

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
    # 🧀 프로젝션 헤드 설정 추가
    use_projection: bool = True
    projection_dim: int = 128  # 출력 차원만 조절 가능

@dataclasses.dataclass
class Training:
    scr_batch_size: int
    memory_batch_size: int
    num_experiences: int
    memory_size: int
    min_samples_per_class: int
    scr_epochs: int
    iterations_per_epoch: int
    num_epochs: int
    num_workers: int
    learning_rate: float
    scheduler_step_size: int
    scheduler_gamma: float
    temperature: float
    save_interval: int
    test_interval: int
    checkpoint_path: Path
    results_path: Path
    gpu_ids: str
    ncm_momentum: float
    batch_size: int = 128
    
    # 💎 ProxyContrastLoss 설정 추가
    use_proxy_loss: bool = True
    proxy_lambda: float = 0.3
    proxy_temperature: float = 0.15  # SupCon보다 높게!
    proxy_topk: int = 30
    proxy_full_until: int = 100
    proxy_warmup_classes: int = 5
    
    # 💎 Lambda 스케줄링 (Optional)
    proxy_lambda_schedule: Optional[Dict[str, float]] = None
    
    # 💎 커버리지 샘플링 설정
    use_coverage_sampling: bool = True
    coverage_k_per_class: int = 2
    
    # 💎 프로토타입 설정
    prototype_beta: float = 0.05  # EMA 계수 (현재는 미사용)
    
    def __post_init__(self):
        """💎 Lambda 스케줄 기본값 설정"""
        if self.proxy_lambda_schedule is None and self.use_proxy_loss:
            self.proxy_lambda_schedule = {
                'warmup': 0.1,
                'warmup_10': 0.2,
                'warmup_20': 0.3
            }

@dataclasses.dataclass
class Openset:
    enabled: bool = True
    warmup_users: int = 10
    initial_tau: float = 0.7
    
    # 🍏 기존 파라미터 (제거/변경)
    # smoothing_alpha: float = 0.2  # threshold_alpha로 변경
    # max_delta: float = 0.03        # threshold_max_delta로 변경
    
    # 🍎 통일된 임계치 파라미터
    threshold_mode: str = 'far'           # 🍎 'eer' or 'far'
    target_far: float = 0.01              # 🍎 FAR 타겟 (1%)
    threshold_alpha: float = 0.2          # 🍎 EMA 계수 (기존 smoothing_alpha)
    threshold_max_delta: float = 0.03     # 🍎 최대 변화폭 (기존 max_delta)
    
    use_margin: bool = True
    margin_tau: float = 0.05
    dev_ratio: float = 0.2
    negref_max_eval: int = 5000
    
    # 🍎 추가 옵션
    verbose_calibration: bool = True      # 🍎 상세 출력

@dataclasses.dataclass
class Negative:
    warmup_experiences: int = 4
    max_per_batch: int = 1
    r0: float = 0.5
    base_id: int = 1