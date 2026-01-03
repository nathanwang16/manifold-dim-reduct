"""
Phase 6: Steering & Alignment Techniques

Provides representation engineering and alignment evaluation for ChromatinCNN.
"""

from .utils import (
    reverse_complement_tensor,
    compute_softmax,
    compute_entropy,
    MetricsTracker,
)
from .activation_cache import ActivationCache
from .steering_vectors import SteeringVectorComputer
from .inference_steering import SteeringInferenceEngine
from .contrastive_steering import ContrastiveSteeringEngine
from .temperature_scaling import TemperatureScaler
from .alignment_evaluation import AlignmentEvaluator

__all__ = [
    'reverse_complement_tensor',
    'compute_softmax',
    'compute_entropy',
    'MetricsTracker',
    'ActivationCache',
    'SteeringVectorComputer',
    'SteeringInferenceEngine',
    'ContrastiveSteeringEngine',
    'TemperatureScaler',
    'AlignmentEvaluator',
]
