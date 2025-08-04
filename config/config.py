import dataclasses
from pathlib import Path

@dataclasses.dataclass
class Dataset:
    config_file: Path
    dataset: str
    train_set_file: Path
    test_set_file: Path
    height: int
    width: int
    channels: int
    augmentation: bool

@dataclasses.dataclass
class Model:
    config_file: Path
    architecture: str
    num_classes: int
    competition_weight: float

@dataclasses.dataclass
class Training:
    config_file: Path
    batch_size: int
    num_epochs: int
    num_workers: int
    learning_rate: float
    scheduler_step_size: int
    scheduler_gamma: float
    ce_weight: float
    contrastive_weight: float
    temperature: float
    save_interval: int
    test_interval: int
    checkpoint_path: Path
    results_path: Path
    gpu_ids: str