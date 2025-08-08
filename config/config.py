# config/config.py
import dataclasses
from pathlib import Path
from typing import Optional, List

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
    # 🔥 사전훈련 관련 추가
    use_pretrained: bool = False
    pretrained_path: Optional[Path] = None
    config_file: Optional[Path] = None

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

@dataclasses.dataclass
class Config:
    dataset: Dataset
    model: Model
    training: Training