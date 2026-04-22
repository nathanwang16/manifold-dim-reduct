"""Steering-vector machinery for `ChromatinCNNAttentionV2`.

Main operations:

* `compute_steering_vectors(activations, labels)` — compute per-class
  steering directions as class centroid minus global centroid on the
  bottleneck (384-d) activations.
* `steered_forward(model, x, feat, alpha, direction)` — run a forward
  pass while *adding* `alpha * direction` to the bottleneck activation
  (via a `forward_hook` on `model.bottleneck`).

Both are wrapped by `run_phase6.py` for full-population sweeps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch


@dataclass
class SteeringVectors:
    centroids: np.ndarray            # (n_classes, D)
    global_centroid: np.ndarray      # (D,)
    directions: np.ndarray           # (n_classes, D)   centroid - global
    unit_directions: np.ndarray      # directions / ||directions||
    counts: np.ndarray               # (n_classes,) samples contributing

    def to_tensor(self, device: str | torch.device = "cpu") -> "TorchSteering":
        device = torch.device(device)
        return TorchSteering(
            directions=torch.from_numpy(self.directions).float().to(device),
            unit_directions=torch.from_numpy(self.unit_directions).float().to(device),
            centroids=torch.from_numpy(self.centroids).float().to(device),
            global_centroid=torch.from_numpy(self.global_centroid).float().to(device),
        )


@dataclass
class TorchSteering:
    directions: torch.Tensor
    unit_directions: torch.Tensor
    centroids: torch.Tensor
    global_centroid: torch.Tensor


def compute_steering_vectors(
    activations: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
) -> SteeringVectors:
    """Per-class centroid minus global centroid on (N, D) activations."""
    d = activations.shape[1]
    centroids = np.zeros((n_classes, d), dtype=np.float32)
    counts = np.zeros(n_classes, dtype=np.int64)
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        centroids[c] = activations[mask].mean(axis=0)
        counts[c] = int(mask.sum())
    global_centroid = activations.mean(axis=0).astype(np.float32)
    directions = centroids - global_centroid[None, :]
    norms = np.linalg.norm(directions, axis=1, keepdims=True).clip(min=1e-8)
    unit_directions = directions / norms
    return SteeringVectors(
        centroids=centroids,
        global_centroid=global_centroid,
        directions=directions,
        unit_directions=unit_directions,
        counts=counts,
    )


def steered_forward(
    model: torch.nn.Module,
    x: torch.Tensor,
    feat: Optional[torch.Tensor],
    direction: torch.Tensor,
    alpha: float,
    *,
    use_unit: bool = False,
) -> Dict[str, torch.Tensor]:
    """Run a forward pass with `bottleneck += alpha * direction` applied.

    Parameters
    ----------
    model         : ChromatinCNNAttentionV2 (or DDP wrapper).
    x, feat       : input batch + engineered features.
    direction     : (D,) or (B, D) tensor to add to the bottleneck.
    alpha         : scalar multiplier.
    use_unit      : if True, `direction` is assumed L2-normalised; alpha is
                    interpreted as a distance along the unit direction. If
                    False, alpha is a plain scalar.
    """
    core = model.module if hasattr(model, "module") else model

    def hook(_module, _inp, output):
        if direction.dim() == 1:
            delta = direction.unsqueeze(0).expand_as(output)
        else:
            delta = direction
        return output + alpha * delta

    h = core.bottleneck.register_forward_hook(hook)
    try:
        return core(x, engineered=feat)
    finally:
        h.remove()
