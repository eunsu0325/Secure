"""
COCONUT: Continual Learning with CCNet + ProxyAnchor + SupCon

Main modules:
- losses: SupConLoss, ProxyAnchorLoss
- memory: ClassBalancedBuffer, ReservoirSamplingBuffer, ExperienceStream
- classifiers: NCMClassifier, ThresholdCalibrator
- training: COCONUTTrainer, ContinualLearningEvaluator
"""

from .losses import SupConLoss, ProxyAnchorLoss
from .memory import ClassBalancedBuffer, ReservoirSamplingBuffer, ExperienceStream
from .classifiers import NCMClassifier, ThresholdCalibrator
from .training import COCONUTTrainer, ContinualLearningEvaluator

__version__ = "1.0.0"

__all__ = [
    # Losses
    'SupConLoss',
    'ProxyAnchorLoss',
    # Memory
    'ClassBalancedBuffer',
    'ReservoirSamplingBuffer',
    'ExperienceStream',
    # Classifiers
    'NCMClassifier',
    'ThresholdCalibrator',
    # Training
    'COCONUTTrainer',
    'ContinualLearningEvaluator',
]