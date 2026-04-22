"""Sparse Autoencoder definitions.

Two variants are supported:

* ``TopKSAE``  — Anthropic-style: keep only the K largest encoder outputs,
  zero the rest. No explicit sparsity penalty needed; sparsity is built-in.
* ``L1SAE``    — Classical sparse autoencoder: ReLU encoder + L1 penalty
  on the hidden activations.

Both share the same weight parameterisation:
    encode:  h = act(W_enc @ (x - b_dec) + b_enc)
    decode:  x_hat = W_dec @ h + b_dec

This "pre-subtract b_dec" trick (from Anthropic's
`SoLU`/`neuron-in-a-haystack` code) keeps the decoder's bias meaningful
even when the encoder's output is pinned to 0 by top-K.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseSAE(nn.Module):
    def __init__(
        self,
        activation_dim: int,
        feature_dim: int,
        *,
        tied_init: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.activation_dim = activation_dim
        self.feature_dim = feature_dim

        w_enc = torch.empty(feature_dim, activation_dim, dtype=dtype)
        nn.init.kaiming_uniform_(w_enc, a=5 ** 0.5)
        if tied_init:
            w_dec = w_enc.clone().T.contiguous()
        else:
            w_dec = torch.empty(activation_dim, feature_dim, dtype=dtype)
            nn.init.kaiming_uniform_(w_dec, a=5 ** 0.5)

        self.W_enc = nn.Parameter(w_enc)
        self.W_dec = nn.Parameter(w_dec)
        self.b_enc = nn.Parameter(torch.zeros(feature_dim, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(activation_dim, dtype=dtype))

    def normalize_decoder(self) -> None:
        """Re-scale each dictionary column to unit L2 norm (after each step)."""
        with torch.no_grad():
            col_norm = self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-8)
            self.W_dec.div_(col_norm)

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - self.b_dec
        return F.linear(centered, self.W_enc, self.b_enc)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return F.linear(h, self.W_dec, self.b_dec)

    # Subclasses implement this
    def encode(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        pre = self.encode_pre(x)
        h = self.activate(pre)
        x_hat = self.decode(h)
        recon = F.mse_loss(x_hat, x, reduction="mean")
        out: Dict[str, torch.Tensor] = {
            "recon": recon,
            "h": h,
            "x_hat": x_hat,
            "pre": pre,
            "l0": (h != 0).float().sum(dim=-1).mean(),
        }
        return out

    def activate(self, pre: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class TopKSAE(BaseSAE):
    def __init__(self, activation_dim: int, feature_dim: int, k: int, **kw):
        super().__init__(activation_dim, feature_dim, **kw)
        self.k = int(k)

    def activate(self, pre: torch.Tensor) -> torch.Tensor:
        # Top-K over the feature dimension, zero out everything else.
        topk_vals, topk_idx = pre.topk(self.k, dim=-1)
        # Only keep positive activations (acts like ReLU after top-K).
        topk_vals = F.relu(topk_vals)
        out = torch.zeros_like(pre)
        out.scatter_(-1, topk_idx, topk_vals)
        return out

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = super().forward(x)
        out["loss"] = out["recon"]
        return out


class L1SAE(BaseSAE):
    def __init__(self, activation_dim: int, feature_dim: int, l1_coeff: float = 1e-3, **kw):
        super().__init__(activation_dim, feature_dim, **kw)
        self.l1_coeff = float(l1_coeff)

    def activate(self, pre: torch.Tensor) -> torch.Tensor:
        return F.relu(pre)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = super().forward(x)
        l1 = out["h"].abs().sum(dim=-1).mean()
        out["l1"] = l1
        out["loss"] = out["recon"] + self.l1_coeff * l1
        return out


def build_sae(cfg: Dict[str, Any]) -> BaseSAE:
    """Construct an SAE from the `phase4_sae` block of config.json."""
    if "phase4_sae" in cfg:
        c = cfg["phase4_sae"]
    else:
        c = cfg
    activation_dim = int(c.get("activation_dim", 384))
    feature_dim = int(c.get("feature_dim", 4096))
    sae_type = c.get("sae_type", "topk").lower()

    if sae_type == "topk":
        return TopKSAE(activation_dim, feature_dim, k=int(c.get("topk_k", 32)))
    if sae_type in ("l1", "relu_l1"):
        return L1SAE(activation_dim, feature_dim, l1_coeff=float(c.get("l1_weight", 1e-3)))
    raise ValueError(f"Unknown sae_type={sae_type}")
