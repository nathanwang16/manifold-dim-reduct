"""Engineered sequence-level features (GC, CpG o/e, entropy, repeats).

These match the features the phase 3 CNN takes as auxiliary input when
`use_engineered_features=True`, and the features phase 8 uses for
per-state sequence diagnostics.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def compute_engineered_features(sequence: str) -> np.ndarray:
    """Return a 5-dim float32 vector: (gc, cpg_ratio, entropy, max_run, repeat_density)."""
    s = sequence.upper()
    n = len(s)
    if n == 0:
        return np.array([0.5, 1.0, 2.0, 1.0, 0.0], dtype=np.float32)

    a = s.count("A"); c = s.count("C"); g = s.count("G"); t = s.count("T")
    total = a + c + g + t
    if total == 0:
        return np.array([0.5, 1.0, 2.0, 1.0, 0.0], dtype=np.float32)

    gc = (g + c) / total

    cpg_obs = 0
    for i in range(n - 1):
        if s[i] == "C" and s[i + 1] == "G":
            cpg_obs += 1
    c_freq = c / total
    g_freq = g / total
    cpg_exp = max(1e-8, (n - 1) * c_freq * g_freq)
    cpg_ratio = float(cpg_obs) / float(cpg_exp)

    p = np.array([a, c, g, t], dtype=np.float64) / float(total)
    ent = float(-(p[p > 0] * np.log2(p[p > 0])).sum())

    max_run = 1
    rep_positions = 0
    run_len = 1
    for i in range(1, n):
        if s[i] == s[i - 1] and s[i] in "ACGT":
            run_len += 1
        else:
            if run_len > max_run:
                max_run = run_len
            if run_len >= 3:
                rep_positions += run_len
            run_len = 1
    if run_len > max_run:
        max_run = run_len
    if run_len >= 3:
        rep_positions += run_len
    repeat_density = rep_positions / max(1, n)

    return np.array([gc, cpg_ratio, ent, float(max_run), repeat_density], dtype=np.float32)


def compute_engineered_features_batch(sequences: Sequence[str]) -> np.ndarray:
    out = np.zeros((len(sequences), 5), dtype=np.float32)
    for i, s in enumerate(sequences):
        out[i] = compute_engineered_features(s)
    return out
