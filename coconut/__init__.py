"""
COCONUT: Compact Online Continual Neural User Templates

A continual learning framework combining:
- CCNet backbone
- Supervised Contrastive Replay
- Open-set recognition with threshold calibration
- Class-balanced memory buffer
"""

# Core components
from coconut.memory import ClassBalancedBuffer, ReservoirSamplingBuffer
from coconut.data import ExperienceStream, MemoryDataset
from coconut.classifiers import NCMClassifier, ThresholdCalibrator
from coconut.training import COCONUTTrainer, ContinualLearningEvaluator
from coconut.losses import SupConLoss, ProxyAnchorLoss

__version__ = '1.0.0'

__all__ = [
    # Memory components
    'ClassBalancedBuffer',
    'ReservoirSamplingBuffer',
    'ExperienceStream',

    # Classifiers
    'NCMClassifier',
    'ThresholdCalibrator',

    # Training
    'COCONUTTrainer',
    'ContinualLearningEvaluator',
    'MemoryDataset',

    # Losses
    'SupConLoss',
    'ProxyAnchorLoss',
]
