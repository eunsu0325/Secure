"""Memory components for COCONUT"""

from .buffer import ReservoirSamplingBuffer, ClassBalancedBuffer
from .cohort_scheduler import CohortScheduler
from .decay_predictor import DecayPredictor
# A4: ExperienceStream lives in coconut.data.stream as canonical; re-export here
# for backward compatibility with imports like `from coconut.memory import ExperienceStream`.
from coconut.data.stream import ExperienceStream

__all__ = [
    'ReservoirSamplingBuffer', 'ClassBalancedBuffer', 'ExperienceStream',
    'CohortScheduler', 'DecayPredictor',
]
