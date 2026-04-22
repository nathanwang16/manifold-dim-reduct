"""Phase 5: activation patching + circuit localization.

The goal is to find *where* in the network class-discriminative information
lives for pairs of confusable chromatin states. For each candidate hook
(stem, res_block*, attn_out, pooled, bottleneck) we replace the activation
on a "clean" forward pass with the one recorded from a "donor" example of
the opposite class, and measure how much the logits shift.

Outputs:
    patching_results.csv   long-form table: (hook, class_a, class_b, metric)
    circuit_summary.json   aggregated per-hook recovery scores
    layer_dla.json         layer-wise direct logit attribution per class
"""

from .hooks import ModelWithHooks, HOOK_NAMES

__all__ = ["ModelWithHooks", "HOOK_NAMES"]
