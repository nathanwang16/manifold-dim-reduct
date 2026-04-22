"""Consolidate stem motifs + saliency + ISM into hypothesis records.

Each hypothesis record is a dict:

    {
        "motif_id":         "stem_filter_42",
        "consensus":        "GATAAG",
        "info_content":     11.4,
        "top_class":        "ZNF/Rpts",
        "class_entropy":    0.81,
        "class_counts":     [...],
        "ism_hotspots":     [(pos, base, mean_delta), ...],
        "interpretation":   "Possible GATA-family motif enriched in ...",
    }

This is deliberately a thin, declarative layer — the heavy numerical
work happens in `stem_motifs.py`, `ism.py`, and `phase7_diagnostics/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from chromatin_lib import STATE_NAMES

from .stem_motifs import pwm_to_consensus


def _shannon(counts: np.ndarray) -> float:
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def build_hypotheses(
    pwms: np.ndarray,
    info_content: np.ndarray,
    class_counts: np.ndarray,
    *,
    min_info_bits: float = 0.3,
    top_class_min_frac: float = 0.15,
    top_n_filters: Optional[int] = None,
    saliency_per_class: Optional[np.ndarray] = None,
    ism_summary: Optional[Dict] = None,
) -> List[Dict]:
    """Return a list of hypothesis records, sorted by specificity × IC."""
    F = pwms.shape[0]
    records: List[Dict] = []
    tot = class_counts.sum(axis=1).astype(np.float32)
    for f in range(F):
        if tot[f] == 0 or info_content[f] < min_info_bits:
            continue
        top_class = int(np.argmax(class_counts[f]))
        top_frac = float(class_counts[f, top_class] / tot[f])
        entropy = _shannon(class_counts[f])
        if top_frac < top_class_min_frac:
            continue
        rec = {
            "motif_id": f"stem_filter_{f:03d}",
            "consensus": pwm_to_consensus(pwms[f]),
            "info_content": float(info_content[f]),
            "top_class_idx": top_class,
            "top_class_name": STATE_NAMES[top_class],
            "top_class_fraction": top_frac,
            "class_entropy_bits": entropy,
            "class_counts": class_counts[f].tolist(),
            "specificity_score": float(info_content[f] * (1.0 - entropy / np.log2(len(STATE_NAMES)))),
        }
        records.append(rec)
    records.sort(key=lambda r: r["specificity_score"], reverse=True)
    if top_n_filters is not None:
        records = records[:top_n_filters]
    return records


def write_markdown(records: List[Dict], path: Path) -> None:
    lines = [
        "# Phase 8 — Motif Hypotheses",
        "",
        "Ranked by information content × class specificity.",
        "",
        "| Motif | Consensus | IC (bits) | Top class | Top class frac | Entropy (bits) | Score |",
        "|-------|-----------|-----------|-----------|----------------|----------------|-------|",
    ]
    for r in records:
        lines.append(
            f"| {r['motif_id']} | `{r['consensus']}` | {r['info_content']:.2f} | "
            f"{r['top_class_name']} | {r['top_class_fraction']:.2f} | "
            f"{r['class_entropy_bits']:.2f} | {r['specificity_score']:.3f} |"
        )
    path.write_text("\n".join(lines))
