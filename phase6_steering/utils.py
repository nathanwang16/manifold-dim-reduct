"""
Shared utilities for Phase 6 steering and alignment.

Provides common functions for tensor operations, metrics computation,
and logging helpers.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
from logger import get_logger

logger = get_logger(__name__)


def get_device() -> str:
    """Get the best available device (MPS for M1, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def reverse_complement_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Compute reverse complement of one-hot encoded DNA sequence.

    Args:
        x: One-hot tensor of shape (batch, 200, 4) or (batch, 4, 200)
           Channel order: A=0, C=1, G=2, T=3

    Returns:
        Reverse complement tensor with same shape.
        RC swaps A<->T (0<->3) and C<->G (1<->2) and reverses position.
    """
    # Determine if input is (batch, seq_len, 4) or (batch, 4, seq_len)
    if x.shape[1] == 4:
        # Channel first: (batch, 4, 200)
        # Reverse along position axis and swap channels
        return x.flip(dims=[2])[:, [3, 2, 1, 0], :]
    else:
        # Channel last: (batch, 200, 4)
        # Reverse along position axis and swap channels
        return x.flip(dims=[1])[:, :, [3, 2, 1, 0]]


def compute_softmax(
    logits: np.ndarray,
    temperature: float = 1.0
) -> np.ndarray:
    """
    Compute softmax probabilities with optional temperature scaling.

    Args:
        logits: Raw logits of shape (n_samples, n_classes)
        temperature: Temperature for scaling (higher = softer distribution)

    Returns:
        Probabilities of shape (n_samples, n_classes)
    """
    scaled = logits / temperature
    # Numerical stability: subtract max
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    exp_logits = np.exp(scaled)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def compute_entropy(probs: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Compute entropy of probability distributions.

    Args:
        probs: Probability array of shape (n_samples, n_classes)
        eps: Small constant for numerical stability

    Returns:
        Entropy values of shape (n_samples,)
    """
    # Clip to avoid log(0)
    probs_clipped = np.clip(probs, eps, 1.0)
    return -np.sum(probs_clipped * np.log(probs_clipped), axis=1)


def compute_kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    eps: float = 1e-10
) -> float:
    """
    Compute KL divergence D_KL(P || Q).

    Args:
        p: Reference distribution
        q: Approximating distribution
        eps: Small constant for numerical stability

    Returns:
        KL divergence value
    """
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.sum(p * np.log(p / q))


def interpolate_sequence(
    seq1: torch.Tensor,
    seq2: torch.Tensor,
    alpha: float
) -> torch.Tensor:
    """
    Interpolate between two sequences in one-hot space.

    Projects back to valid one-hot via argmax at each position.

    Args:
        seq1: First sequence (batch, 200, 4) or (200, 4)
        seq2: Second sequence, same shape
        alpha: Interpolation weight (0 = seq1, 1 = seq2)

    Returns:
        Interpolated sequence, same shape
    """
    # Linear interpolation
    interpolated = (1 - alpha) * seq1 + alpha * seq2

    # Project back to valid one-hot
    if interpolated.dim() == 2:
        # Single sequence (200, 4)
        idx = torch.argmax(interpolated, dim=1)
        result = torch.zeros_like(interpolated)
        result.scatter_(1, idx.unsqueeze(1), 1.0)
    else:
        # Batch (batch, 200, 4)
        idx = torch.argmax(interpolated, dim=2)
        result = torch.zeros_like(interpolated)
        result.scatter_(2, idx.unsqueeze(2), 1.0)

    return result


def cosine_similarity(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        v1: First vector
        v2: Second vector
        eps: Small constant for numerical stability

    Returns:
        Cosine similarity in [-1, 1]
    """
    norm1 = np.linalg.norm(v1) + eps
    norm2 = np.linalg.norm(v2) + eps
    return np.dot(v1, v2) / (norm1 * norm2)


def l2_normalize(vectors: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    L2 normalize vectors along specified axis.

    Args:
        vectors: Array of vectors
        axis: Axis along which to normalize

    Returns:
        Normalized vectors
    """
    norms = np.linalg.norm(vectors, axis=axis, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return vectors / norms


class MetricsTracker:
    """
    Tracks and logs metrics during steering experiments.

    Provides methods for logging individual metrics and computing
    summary statistics.
    """

    def __init__(self, logger_instance=None):
        """
        Initialize metrics tracker.

        Args:
            logger_instance: Logger to use, defaults to module logger
        """
        self.metrics: Dict[str, List[float]] = {}
        self.logger = logger_instance or logger

    def log(self, name: str, value: float) -> None:
        """
        Log a metric value.

        Args:
            name: Metric name
            value: Metric value
        """
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
        self.logger.debug(f"Metric {name}: {value:.4f}")

    def log_dict(self, metrics_dict: Dict[str, float]) -> None:
        """
        Log multiple metrics at once.

        Args:
            metrics_dict: Dictionary of metric name -> value
        """
        for name, value in metrics_dict.items():
            self.log(name, value)

    def get_latest(self, name: str) -> Optional[float]:
        """Get the most recent value for a metric."""
        if name in self.metrics and self.metrics[name]:
            return self.metrics[name][-1]
        return None

    def get_all(self, name: str) -> List[float]:
        """Get all values for a metric."""
        return self.metrics.get(name, [])

    def summarize(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics of all tracked metrics.

        Returns:
            Dictionary mapping metric name to stats (mean, std, min, max, last)
        """
        summary = {}
        for name, values in self.metrics.items():
            if values:
                arr = np.array(values)
                summary[name] = {
                    'mean': float(np.mean(arr)),
                    'std': float(np.std(arr)),
                    'min': float(np.min(arr)),
                    'max': float(np.max(arr)),
                    'last': float(arr[-1]),
                    'count': len(values),
                }
        return summary

    def clear(self) -> None:
        """Clear all tracked metrics."""
        self.metrics.clear()

    def to_dict(self) -> Dict[str, List[float]]:
        """Export all metrics as dictionary."""
        return dict(self.metrics)


def pool_bottleneck_activations(
    bottleneck_acts: torch.Tensor
) -> torch.Tensor:
    """
    Apply global max and average pooling to bottleneck activations
    and concatenate, matching the model's pooling behavior.

    Args:
        bottleneck_acts: Tensor of shape (batch, 512, 200)

    Returns:
        Pooled tensor of shape (batch, 1024)
    """
    # Global max pooling
    x_max, _ = torch.max(bottleneck_acts, dim=2)  # (batch, 512)

    # Global average pooling
    x_avg = torch.mean(bottleneck_acts, dim=2)  # (batch, 512)

    # Concatenate
    return torch.cat([x_max, x_avg], dim=1)  # (batch, 1024)


def pool_bottleneck_activations_np(
    bottleneck_acts: np.ndarray
) -> np.ndarray:
    """
    NumPy version of pooling for cached activations.

    Args:
        bottleneck_acts: Array of shape (batch, 512, 200)

    Returns:
        Pooled array of shape (batch, 1024)
    """
    # Global max pooling
    x_max = np.max(bottleneck_acts, axis=2)  # (batch, 512)

    # Global average pooling
    x_avg = np.mean(bottleneck_acts, axis=2)  # (batch, 512)

    # Concatenate
    return np.concatenate([x_max, x_avg], axis=1)  # (batch, 1024)
