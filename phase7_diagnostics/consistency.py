"""Reverse-complement prediction consistency."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def reverse_complement_consistency(
    model: torch.nn.Module,
    loader,
    *,
    device: str = "cuda:0",
    use_feat: bool = True,
) -> Dict:
    """Fraction of val samples where argmax(model(x)) == argmax(model(RC(x))).

    Also reports per-class consistency and KL divergence between the two
    softmax distributions.
    """
    from chromatin_lib import reverse_complement_onehot

    model.eval()
    total = 0
    match = 0
    kl_sum = 0.0
    per_class_total = np.zeros(18, dtype=np.int64)
    per_class_match = np.zeros(18, dtype=np.int64)
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].numpy()
            feat = batch.get("feat")
            if use_feat and feat is not None:
                feat = feat.to(device, non_blocking=True)
            else:
                feat = None
            out = model(x, engineered=feat)
            x_rc = reverse_complement_onehot(x)
            out_rc = model(x_rc, engineered=feat)
            p = torch.softmax(out["logits"], dim=-1)
            p_rc = torch.softmax(out_rc["logits"], dim=-1)
            preds = out["logits"].argmax(dim=-1).cpu().numpy()
            preds_rc = out_rc["logits"].argmax(dim=-1).cpu().numpy()
            eq = preds == preds_rc
            match += int(eq.sum())
            total += eq.shape[0]
            kl = (p * (p.clamp(min=1e-12).log() - p_rc.clamp(min=1e-12).log())).sum(dim=-1)
            kl_sum += float(kl.sum().item())
            for c in range(18):
                mask = y == c
                per_class_total[c] += int(mask.sum())
                per_class_match[c] += int(eq[mask].sum())
    return {
        "overall_consistency": float(match / max(1, total)),
        "mean_kl": float(kl_sum / max(1, total)),
        "per_class_consistency": (
            per_class_match / np.maximum(per_class_total, 1)
        ).tolist(),
        "per_class_counts": per_class_total.tolist(),
    }
