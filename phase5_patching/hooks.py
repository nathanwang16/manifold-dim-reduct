"""Lightweight hook machinery for `ChromatinCNNAttentionV2`.

Exposes a `ModelWithHooks` wrapper that can either *record* activations at
a named hook point, or *replace* them with a pre-recorded tensor during
forward. The set of supported hook points:

    stem          — output of the stem conv + BN + ReLU       (B, 256, 200)
    res_block1    — after residual block 1 (pre pool)          (B, 256, 200)
    conv_expand1  — after the 256→384 expand + BN + ReLU       (B, 384, 100)
    res_block2    — after residual block 2 (pre pool)          (B, 384, 100)
    conv_expand2  — after the 384→512 expand + BN + ReLU       (B, 512,  50)
    res_block3    — after residual block 3                      (B, 512,  50)
    attn_out      — after all transformer blocks                (B, 512,  50)
    pooled        — global avg-pooled trunk                      (B, 512)
    bottleneck    — the 384-d bottleneck (SAE extraction)        (B, 384)

Internally we use `nn.Module.register_forward_hook` rather than rewriting
the model's forward, so the model itself stays untouched.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn

HOOK_NAMES: List[str] = [
    "stem",
    "res_block1",
    "conv_expand1",
    "res_block2",
    "conv_expand2",
    "res_block3",
    "attn_out",
    "pooled",
    "bottleneck",
]


class ModelWithHooks:
    """Context-manager-style wrapper around `ChromatinCNNAttentionV2`.

    Usage:
        wrapped = ModelWithHooks(model)

        # 1) record activations at a donor example's forward
        with wrapped.record(["bottleneck", "res_block2"]) as cache:
            wrapped.model(x_donor, engineered=feat_donor)
        donor_acts = cache  # dict name -> Tensor

        # 2) replay them during a clean run
        with wrapped.patch({"bottleneck": donor_acts["bottleneck"]}):
            out = wrapped.model(x_clean, engineered=feat_clean)
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._recorders: Dict[str, Optional[torch.Tensor]] = {}
        self._patches: Dict[str, torch.Tensor] = {}
        self._handles: List = []

    # ------------------------------------------------------------------
    # Target module resolution
    # ------------------------------------------------------------------
    def _module_for(self, name: str) -> nn.Module:
        core = self.model.module if hasattr(self.model, "module") else self.model
        mapping = {
            "stem": core.stem_bn,
            "res_block1": core.res_block1,
            "conv_expand1": core.bn_expand1,
            "res_block2": core.res_block2,
            "conv_expand2": core.bn_expand2,
            "res_block3": core.res_block3,
            "attn_out": core.attn_blocks[-1],
            "pooled": core.global_pool,
            "bottleneck": core.bottleneck,
        }
        if name not in mapping:
            raise KeyError(f"Unknown hook name: {name}")
        return mapping[name]

    def _post_hook(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        # Some modules return channel-first (B, C, L); others channel-last.
        # We normalize "attn_out" to channel-first for consistency with the
        # trunk. attn_blocks return (B, L, C).
        if name == "attn_out":
            tensor = tensor.transpose(1, 2).contiguous()
        if name == "pooled":
            tensor = tensor.squeeze(-1)
        return tensor

    def _pre_hook_output(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        # Inverse of _post_hook when patching outputs back into the graph.
        if name == "attn_out":
            return tensor.transpose(1, 2).contiguous()
        if name == "pooled":
            return tensor.unsqueeze(-1)
        return tensor

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    @contextmanager
    def record(self, names: Iterable[str]):
        cache: Dict[str, torch.Tensor] = {}

        def make_hook(name: str):
            def hook(_module, _inp, output):
                cache[name] = self._post_hook(name, output.detach().clone())
            return hook

        handles = []
        for n in names:
            mod = self._module_for(n)
            handles.append(mod.register_forward_hook(make_hook(n)))
        try:
            yield cache
        finally:
            for h in handles:
                h.remove()

    # ------------------------------------------------------------------
    # Patching
    # ------------------------------------------------------------------
    @contextmanager
    def patch(self, patches: Dict[str, torch.Tensor]):
        """Replace the output of each named module with the supplied tensor."""
        handles = []

        def make_hook(name: str, replacement: torch.Tensor):
            def hook(_module, _inp, _output):
                return self._pre_hook_output(name, replacement)
            return hook

        for n, t in patches.items():
            mod = self._module_for(n)
            handles.append(mod.register_forward_hook(make_hook(n, t)))
        try:
            yield
        finally:
            for h in handles:
                h.remove()
