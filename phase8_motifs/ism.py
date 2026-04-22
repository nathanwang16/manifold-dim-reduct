"""In-silico mutagenesis (ISM).

For each input sequence `x`, we compute the logit delta for the predicted
(or supplied) class when substituting the base at each position with each
of the 4 possible bases:

    delta[b, l, n] = logit_c(x_{l <- n}) - logit_c(x)

For one-hot inputs, the native base has delta=0 by construction. The
resulting `delta` tensor is a richer per-position signal than SmoothGrad:
it decomposes into both position importance and direction (which base
change hurts / helps).

We batch the mutants so the whole 200*4 substitution grid for one sample
is evaluated with a single forward pass over 800 sequences.
"""

from __future__ import annotations

from typing import Optional

import torch


@torch.no_grad()
def in_silico_mutagenesis(
    model: torch.nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    *,
    engineered: Optional[torch.Tensor] = None,
    mutation_batch: int = 800,
) -> torch.Tensor:
    """Return per-sample logit deltas of shape (B, L, 4)."""
    model.eval()
    if x.dim() != 3:
        raise ValueError("x must be (B, L, 4) one-hot tensor")
    B, L, C = x.shape
    device = x.device
    baseline = model(x, engineered=engineered)["logits"]     # (B, n_classes)
    idx_b = torch.arange(B, device=device)
    ref_logit = baseline[idx_b, target]                     # (B,)
    deltas = torch.zeros(B, L, C, device=device)
    for b in range(B):
        x_b = x[b]                                           # (L, 4)
        feat_b = engineered[b] if engineered is not None else None
        muts = x_b.unsqueeze(0).expand(L * C, L, C).clone()
        # For position l & target base n, set the l-th row to one-hot[n].
        l_idx = torch.arange(L, device=device).repeat_interleave(C)
        n_idx = torch.arange(C, device=device).repeat(L)
        m_idx = torch.arange(L * C, device=device)
        muts[m_idx, l_idx, :] = 0.0
        muts[m_idx, l_idx, n_idx] = 1.0
        mut_logits = torch.zeros(L * C, baseline.shape[1], device=device)
        for start in range(0, L * C, mutation_batch):
            end = min(start + mutation_batch, L * C)
            sub = muts[start:end]
            feat_sub = feat_b.unsqueeze(0).expand(sub.shape[0], -1) if feat_b is not None else None
            mut_logits[start:end] = model(sub, engineered=feat_sub)["logits"]
        tgt_logit_mut = mut_logits[:, target[b]]
        deltas[b] = (tgt_logit_mut - ref_logit[b]).view(L, C)
    return deltas
