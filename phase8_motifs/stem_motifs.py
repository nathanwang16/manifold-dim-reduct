"""Extract PWMs from stem Conv1d filters.

Strategy (TF-MoDISco-lite style):
    1. Pass a balanced val subset through `model.stem_conv + stem_bn` only.
    2. For each filter f, find the top-K windows (across batch × position)
       where the activation is highest.
    3. For each such window, extract the corresponding 19-bp slice of the
       original one-hot input and accumulate into a position-weight
       matrix (PWM).
    4. Normalise and report per-filter information content (bits).
    5. Associate each filter with the classes whose samples most often
       produced its top-K windows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F


def _pwm_info_content(pwm: np.ndarray, eps: float = 1e-9) -> float:
    """Sum(2 + Σ p log2 p) across positions. PWM shape (L, 4)."""
    H = -np.sum(pwm * np.log2(pwm + eps), axis=-1)
    return float(np.sum(2.0 - H))


def extract_stem_pwms(
    model: torch.nn.Module,
    loader,
    *,
    device: str = "cuda:0",
    top_k_per_filter: int = 500,
    batch_limit: int | None = None,
) -> Dict[str, np.ndarray]:
    """Return dict with keys `pwms` (F, L, 4), `info_content` (F,), and
    `class_counts` (F, n_classes)."""
    model.eval()
    stem = model.stem_conv
    bn = model.stem_bn
    F_filters, _, L = stem.weight.shape
    n_classes = int(getattr(model, "class_head").out_features)

    top_vals: List[torch.Tensor] = [torch.full((top_k_per_filter,), -float("inf"))
                                    for _ in range(F_filters)]
    # tuples: (sample_idx_within_stream, position, class_label)
    top_meta: List[List] = [[] for _ in range(F_filters)]
    # Store reference to one-hot windows on CPU for PWM accumulation
    top_windows: List[List[np.ndarray]] = [[] for _ in range(F_filters)]

    seen = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if batch_limit is not None and bi >= batch_limit:
                break
            x = batch["x"].to(device, non_blocking=True)  # (B, 200, 4)
            y = batch["y"].numpy()
            B, Lseq, _ = x.shape
            x_t = x.transpose(1, 2).contiguous()          # (B, 4, 200)
            act = F.relu(bn(stem(x_t)))                   # (B, F, 200)
            # We need activation values & positions per filter.
            flat = act.view(B, F_filters, -1)             # same
            for f in range(F_filters):
                vals = flat[:, f, :]                      # (B, 200)
                # Best activation per position then top-k overall
                topv, topi = vals.reshape(-1).topk(
                    min(top_k_per_filter, vals.numel())
                )
                topv_cpu = topv.cpu()
                topi_cpu = topi.cpu().numpy()
                # Merge with running top
                merged_vals = torch.cat([top_vals[f], topv_cpu])
                merged_meta = top_meta[f] + [
                    (seen + int(i // Lseq), int(i % Lseq), int(y[i // Lseq]),
                     topv_cpu[j].item())
                    for j, i in enumerate(topi_cpu)
                ]
                # Also keep windows for new entries; for old entries we
                # track stored windows.
                new_windows = []
                for i in topi_cpu:
                    sample_idx = int(i // Lseq)
                    pos = int(i % Lseq)
                    start = max(0, pos - L // 2)
                    end = min(Lseq, start + L)
                    start = max(0, end - L)
                    win = x[sample_idx, start:end, :].cpu().numpy()
                    if win.shape[0] < L:
                        pad = np.zeros((L - win.shape[0], 4), dtype=win.dtype)
                        win = np.concatenate([win, pad], axis=0)
                    new_windows.append(win)
                merged_windows = top_windows[f] + new_windows
                # Rank and truncate
                order = torch.argsort(merged_vals, descending=True)[:top_k_per_filter]
                top_vals[f] = merged_vals[order]
                top_meta[f] = [merged_meta[i] for i in order.tolist()]
                top_windows[f] = [merged_windows[i] for i in order.tolist()]
            seen += B

    # Build PWMs & class counts
    pwms = np.zeros((F_filters, L, 4), dtype=np.float32)
    class_counts = np.zeros((F_filters, n_classes), dtype=np.int64)
    info = np.zeros(F_filters, dtype=np.float32)
    for f in range(F_filters):
        if not top_windows[f]:
            continue
        stacked = np.stack(top_windows[f], axis=0)                # (K, L, 4)
        counts = stacked.sum(axis=0)                              # (L, 4)
        pwm = counts / np.maximum(counts.sum(axis=-1, keepdims=True), 1.0)
        pwms[f] = pwm.astype(np.float32)
        info[f] = _pwm_info_content(pwm)
        for (_, _, cls, _) in top_meta[f]:
            class_counts[f, cls] += 1
    return {
        "pwms": pwms,
        "info_content": info,
        "class_counts": class_counts,
    }


def pwm_to_consensus(pwm: np.ndarray, threshold: float = 0.6) -> str:
    """Convert a PWM to a consensus string. Lowercase if no base passes threshold."""
    bases = "ACGT"
    out = []
    for row in pwm:
        j = int(np.argmax(row))
        if row[j] >= threshold:
            out.append(bases[j])
        else:
            out.append(bases[j].lower())
    return "".join(out)
