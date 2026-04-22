"""SmoothGrad saliency on DNA inputs.

`smoothgrad_saliency(model, x, target, n_samples, noise_std)` returns a
`(B, L, 4)` tensor of mean absolute gradients of `logits[:, target]` w.r.t.
noisy one-hot inputs, averaged over `n_samples` Gaussian perturbations.

Using the absolute gradient (rather than signed) avoids direction
ambiguity across replicates; for sequence logos we sum across the 4
channels to get a per-base importance score.
"""

from __future__ import annotations

from typing import Optional

import torch


def smoothgrad_saliency(
    model: torch.nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    *,
    engineered: Optional[torch.Tensor] = None,
    n_samples: int = 20,
    noise_std: float = 0.1,
) -> torch.Tensor:
    """Return (B, L, 4) tensor of mean |∂ logit_target / ∂ x|."""
    model.eval()
    if x.dim() != 3:
        raise ValueError("x must be (B, L, 4) one-hot tensor")
    device = x.device
    accum = torch.zeros_like(x)
    for _ in range(n_samples):
        noise = torch.randn_like(x) * noise_std
        x_noisy = (x + noise).requires_grad_(True)
        out = model(x_noisy, engineered=engineered)
        logits = out["logits"]
        idx = torch.arange(logits.shape[0], device=device)
        selected = logits[idx, target]
        grads = torch.autograd.grad(
            outputs=selected.sum(), inputs=x_noisy,
            retain_graph=False, create_graph=False,
        )[0]
        accum = accum + grads.abs()
    return accum / n_samples
