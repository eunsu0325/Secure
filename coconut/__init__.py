"""
COCONUT: Compact Online Continual Neural User Templates
A continual learning framework for palm vein recognition
"""

from .data_stream import ExperienceStream
from .memory_buffer import ClassBalancedBuffer, ReservoirSamplingBuffer
from .ncm_classifier import NCMClassifier
from .trainer import COCONUTTrainer
from .threshold import ThresholdCalibrator

__version__ = "1.0.0"
__all__ = [
    'ExperienceStream',
    'ClassBalancedBuffer',
    'ReservoirSamplingBuffer',
    'NCMClassifier',
    'COCONUTTrainer',
    'ThresholdCalibrator'
]