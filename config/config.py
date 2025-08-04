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
    config_file: Optional[Path] = None

@dataclasses.dataclass
class Model:
    architecture: str
    # num_classes: int  # 🔥 제거
    competition_weight: float
    config_file: Optional[Path] = None

@dataclasses.dataclass
class Training:
    batch_size: int
    num_epochs: int
    num_workers: int
    learning_rate: float
    scheduler_step_size: int
    scheduler_gamma: float
    # ce_weight: float  # 🔥 제거
    # contrastive_weight: float  # 🔥 제거
    temperature: float
    save_interval: int
    test_interval: int
    checkpoint_path: Path
    results_path: Path
    gpu_ids: str
    config_file: Optional[Path] = None