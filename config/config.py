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
    # SCR 추가 🐣
    negative_samples_file: Path  # 🐣
    num_negative_classes: int = 10  # 🐣
    config_file: Optional[Path] = None

@dataclasses.dataclass
class Model:
    architecture: str
    # num_classes: int  # 🪦 제거
    competition_weight: float
    config_file: Optional[Path] = None

@dataclasses.dataclass
class Training:
    # batch_size: int  # 🪦 기존 배치 사이즈 제거
    scr_batch_size: int  # 🐣 새 사용자 샘플 수 (기본 10)
    memory_batch_size: int  # 🐣 메모리에서 가져올 샘플 수
    
    # SCR 특화 설정 🐣
    num_experiences: int  # 🐣
    memory_size: int  # 🐣
    min_samples_per_class: int  # 🐣
    scr_epochs: int  # 🐣 각 experience당 epoch 수
    
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
    inference_checkpoint: Optional[Path] = None
    config_file: Optional[Path] = None