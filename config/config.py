# 🐋 config/config.py 수정
import dataclasses
from pathlib import Path
from typing import Optional


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
    batch_size: int = 128  # 🐋 평가용 배치 사이즈 기본값


@dataclasses.dataclass
class Openset:  # 🐋 신규 추가
    """Open-set configuration"""
    enabled: bool = True
    warmup_users: int = 10
    initial_tau: float = 0.7
    smoothing_alpha: float = 0.2
    max_delta: float = 0.03
    use_margin: bool = True
    margin_tau: float = 0.05
    dev_ratio: float = 0.2
    negref_max_eval: int = 5000


@dataclasses.dataclass
class Negative:  # 🔥 신규 추가
    """Negative sample configuration"""
    warmup_experiences: int = 4     # exp0~3까지 네거티브 사용
    max_per_batch: int = 1          # 배치당 네거티브 클래스 최대 1장
    r0: float = 0.5                 # 초기 메모리:현재유저 비율
    base_id: int = 1                # 런타임에 동적 설정 (max_user_id + 1)