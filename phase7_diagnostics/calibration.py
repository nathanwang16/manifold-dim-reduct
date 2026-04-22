"""Temperature scaling + expected calibration error."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def expected_calibration_error(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:
    probs = F.softmax(logits, dim=-1)
    confidences, predictions = probs.max(dim=-1)
    accuracies = predictions.eq(labels).float()
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    ece = torch.zeros(1, device=logits.device)
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        prop = mask.float().mean()
        if prop > 0:
            acc = accuracies[mask].mean()
            conf = confidences[mask].mean()
            ece += prop * (acc - conf).abs()
    return float(ece.item())


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    max_iter: int = 50,
    lr: float = 0.01,
    device: str = "cpu",
) -> Tuple[float, float, float]:
    """Fit a scalar temperature on (logits, labels) and report ECE before/after."""
    logits = logits.to(device)
    labels = labels.to(device)
    ece_before = expected_calibration_error(logits, labels)

    T = torch.nn.Parameter(torch.ones(1, device=device))
    optimizer = torch.optim.LBFGS([T], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / T.clamp(min=1e-3), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    T_val = float(T.detach().clamp(min=1e-3).item())
    ece_after = expected_calibration_error(logits / T_val, labels)
    return T_val, ece_before, ece_after
