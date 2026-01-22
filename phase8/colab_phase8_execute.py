"""
Phase 8 (Colab): Chromatin label → biological state identification

Usage (Colab):
1) Put your CSVs in Google Drive:
   /content/drive/MyDrive/chromatin_data/train_sequences.csv
   /content/drive/MyDrive/chromatin_data/train_labels.csv
2) Run this script in a Colab cell (copy/paste), or as a file:
   python colab_phase8_execute.py

Outputs (default):
  /content/drive/MyDrive/chromatin_phase8/label_identification_output/
    - label_state_mapping.json
    - identification_report.txt
    - label_feature_profiles.csv
    - similarity_matrix.npy
    - feature_distributions.png
    - label_scatter.png
    - label_centroids.png
"""

from __future__ import annotations

import os
import re
import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 0) Drive mounting & paths (Colab-safe, but works locally too)
# -----------------------------------------------------------------------------

try:
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")
    BASE_DIR = "/content/drive/MyDrive"
except Exception:
    BASE_DIR = "."
    print("Drive not mounted; using local directory.")

DATA_DIR_DEFAULT = f"{BASE_DIR}/chromatin_data"
OUT_DIR_DEFAULT = f"{BASE_DIR}/chromatin_phase8/label_identification_output"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


# -----------------------------------------------------------------------------
# 1) Robust CSV loading
# -----------------------------------------------------------------------------

def _pick_sequence_column(df: pd.DataFrame) -> int:
    """
    Heuristic:
    - Prefer an explicit column name containing 'seq'
    - Else: choose the column with the highest mean string length
    """
    lowered = {str(c).lower(): i for i, c in enumerate(df.columns)}
    for key in lowered:
        if "seq" in key:
            return lowered[key]

    # Fallback: pick the column that looks most like sequences
    best_i, best_len = 0, -1.0
    for i in range(df.shape[1]):
        col = df.iloc[:, i].astype(str)
        mean_len = col.str.len().mean()
        if mean_len > best_len:
            best_len = mean_len
            best_i = i
    return best_i


def _pick_label_column(df: pd.DataFrame) -> int:
    """
    Heuristic:
    - Prefer an explicit column name containing 'label'
    - Else: if 2+ columns, choose the most numeric-like
    - Else: use column 0
    """
    lowered = {str(c).lower(): i for i, c in enumerate(df.columns)}
    for key in lowered:
        if "label" in key:
            return lowered[key]

    if df.shape[1] == 1:
        return 0

    best_i, best_score = 0, -1.0
    for i in range(df.shape[1]):
        col = pd.to_numeric(df.iloc[:, i], errors="coerce")
        score = col.notna().mean()
        if score > best_score:
            best_score = score
            best_i = i
    return best_i


def load_sequences_and_labels(
    sequences_path: str,
    labels_path: str,
) -> Tuple[List[str], List[int]]:
    seq_df = pd.read_csv(sequences_path)
    lab_df = pd.read_csv(labels_path)

    seq_col = _pick_sequence_column(seq_df)
    lab_col = _pick_label_column(lab_df)

    sequences = seq_df.iloc[:, seq_col].astype(str).str.strip().tolist()
    labels_raw = lab_df.iloc[:, lab_col].tolist()

    labels = []
    for x in labels_raw:
        try:
            labels.append(int(x))
        except Exception:
            # try stripping strings like "13.0"
            labels.append(int(float(str(x).strip())))

    if len(sequences) != len(labels):
        raise ValueError(
            f"Length mismatch: {len(sequences)} sequences vs {len(labels)} labels"
        )

    # Normalize to 1..18 if needed (many pipelines store 0..17)
    min_lab, max_lab = min(labels), max(labels)
    if min_lab == 0 and max_lab == 17:
        labels = [x + 1 for x in labels]

    return sequences, labels


# -----------------------------------------------------------------------------
# 2) Feature extraction (aligned with STRATEGY.md)
# -----------------------------------------------------------------------------

DNA_ALPHABET = set("ACGT")


def compute_gc_content(sequence: str) -> float:
    seq = sequence.upper()
    if not seq:
        return 0.0
    gc = sum(1 for b in seq if b in ("G", "C"))
    return gc / len(seq)


def compute_cpg_ratio(sequence: str) -> float:
    """
    CpG observed/expected:
      O/E = (count("CG") * L) / (count("C") * count("G"))
    """
    seq = sequence.upper()
    L = len(seq)
    if L == 0:
        return 0.0
    cpg = seq.count("CG")
    c = seq.count("C")
    g = seq.count("G")
    if c == 0 or g == 0:
        return 0.0
    expected = (c * g) / L
    return float(cpg / expected) if expected > 0 else 0.0


def compute_cpg_frequency(sequence: str) -> float:
    seq = sequence.upper()
    L = len(seq)
    if L < 2:
        return 0.0
    return seq.count("CG") / (L - 1)


def _regex_count_overlapping(pattern: str, seq: str) -> int:
    # Use lookahead to count overlapping matches
    return len(re.findall(rf"(?=({pattern}))", seq))


def homopolymer_run_bases(sequence: str, min_len: int = 4) -> int:
    seq = sequence.upper()
    total = 0
    for base in "ACGT":
        for m in re.finditer(rf"{base}{{{min_len},}}", seq):
            total += (m.end() - m.start())
    return total


def dinucleotide_repeat_bases(sequence: str, min_repeats: int = 4) -> int:
    """
    Count bases in simple dinucleotide tandem repeats like (AT){4,}.
    """
    seq = sequence.upper()
    total = 0
    # All 16 dinucs, but restrict to those commonly repetitive to avoid noise
    for dinuc in ("AT", "TA", "CA", "AC", "TG", "GT", "CG", "GC"):
        for m in re.finditer(rf"(?:{dinuc}){{{min_repeats},}}", seq):
            total += (m.end() - m.start())
    return total


def compute_repeat_density(sequence: str) -> float:
    """
    Heuristic repeat density (no RepeatMasker):
    fraction of bases participating in homopolymer runs (>=4) or dinuc repeats (>=4).
    """
    seq = sequence.upper()
    L = len(seq)
    if L == 0:
        return 0.0
    rep = homopolymer_run_bases(seq, min_len=4) + dinucleotide_repeat_bases(seq, min_repeats=4)
    return float(min(rep / L, 1.0))


def shannon_entropy(sequence: str) -> float:
    seq = [b for b in sequence.upper() if b in DNA_ALPHABET]
    if not seq:
        return 0.0
    counts = np.array([seq.count(b) for b in "ACGT"], dtype=np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def scan_motifs(sequence: str) -> Dict[str, int]:
    seq = sequence.upper()
    # Core motifs from STRATEGY.md (+ a couple already used in repo)
    patterns = {
        "TATA_box": r"TATA[AT]A[AT]",
        "GC_box": r"GGGCGG",
        "AP1": r"TGA[CG]TCA",
        "ETS": r"[AC]GGA[AT]G",
        "CTCF_core": r"CC[AG]C[CG]AGGGGGC",
        # "Inr" / promoter-ish; not in STRATEGY list but useful
        "Inr": r"[CT][CT]A[ACGT][AT][CT][CT]",
        # Simple CpG-dense proxy (polycomb-ish / CpG islands)
        "CpG_dense": r"CGCG",
        # KRAB-ZNF-ish proxy (very weak, but keep for continuity)
        "KRAB_ZNF": r"TGCAG",
    }
    out: Dict[str, int] = {}
    for name, pat in patterns.items():
        out[name] = _regex_count_overlapping(pat, seq)
    return out


KEY_FEATURES = [
    "gc_content",
    "at_content",
    "cpg_ratio",
    "cpg_freq",
    "repeat_density",
    "entropy",
    "motif_TATA_box",
    "motif_GC_box",
    "motif_AP1",
    "motif_ETS",
    "motif_CTCF_core",
    "motif_Inr",
    "motif_CpG_dense",
    "motif_KRAB_ZNF",
]

SAFE_FEATURES = [
    # Stable, interpretable, and present in expected prototypes
    "gc_content",
    "at_content",
    "cpg_ratio",
    "cpg_freq",
    "repeat_density",
    "entropy",
]


def extract_features(sequence: str) -> Dict[str, float]:
    seq = sequence.upper().strip()
    L = len(seq)
    feats: Dict[str, float] = {}
    feats["gc_content"] = compute_gc_content(seq)
    feats["at_content"] = 1.0 - feats["gc_content"]
    feats["cpg_ratio"] = compute_cpg_ratio(seq)
    feats["cpg_freq"] = compute_cpg_frequency(seq)
    feats["repeat_density"] = compute_repeat_density(seq)
    feats["entropy"] = shannon_entropy(seq)

    motifs = scan_motifs(seq)
    # Normalize motif counts per 200bp-ish length (still robust if sequences differ a bit)
    denom = max(L / 200.0, 1e-6)
    for k, v in motifs.items():
        feats[f"motif_{k}"] = float(v) / denom
    return feats


# -----------------------------------------------------------------------------
# 3) Expected state prototypes (from phase8/label_identification_plan.py)
# -----------------------------------------------------------------------------

def expected_state_profiles() -> Dict[str, Dict[str, float]]:
    """
    Prototype profiles in 0..1-ish units (heuristic, from literature).
    These are not probabilities; they are relative expected levels.
    """
    # Keep the same state list/order as the existing repo module for consistency.
    return {
        "TssA": {"gc_content": 0.65, "cpg_ratio": 0.85, "cpg_freq": 0.08, "repeat_density": 0.05, "motif_TATA_box": 0.3, "motif_GC_box": 0.6, "motif_Inr": 0.4, "at_content": 0.35, "entropy": 1.8},
        "TssFlnk": {"gc_content": 0.55, "cpg_ratio": 0.60, "cpg_freq": 0.05, "repeat_density": 0.10, "motif_TATA_box": 0.2, "motif_GC_box": 0.4, "at_content": 0.45, "entropy": 1.9},
        "TssFlnkU": {"gc_content": 0.52, "cpg_ratio": 0.55, "cpg_freq": 0.04, "repeat_density": 0.10, "at_content": 0.48, "entropy": 1.9},
        "TssFlnkD": {"gc_content": 0.50, "cpg_ratio": 0.45, "cpg_freq": 0.035, "repeat_density": 0.12, "at_content": 0.50, "entropy": 1.95},
        "Tx": {"gc_content": 0.45, "cpg_ratio": 0.35, "cpg_freq": 0.025, "repeat_density": 0.15, "at_content": 0.55, "entropy": 1.95},
        "TxWk": {"gc_content": 0.42, "cpg_ratio": 0.30, "cpg_freq": 0.020, "repeat_density": 0.18, "at_content": 0.58, "entropy": 1.95},
        "EnhG1": {"gc_content": 0.48, "cpg_ratio": 0.40, "cpg_freq": 0.030, "repeat_density": 0.12, "motif_AP1": 0.3, "motif_ETS": 0.3, "at_content": 0.52, "entropy": 1.95},
        "EnhG2": {"gc_content": 0.46, "cpg_ratio": 0.38, "cpg_freq": 0.028, "repeat_density": 0.14, "motif_AP1": 0.25, "motif_ETS": 0.28, "at_content": 0.54, "entropy": 1.95},
        "EnhA1": {"gc_content": 0.47, "cpg_ratio": 0.42, "cpg_freq": 0.032, "repeat_density": 0.10, "motif_AP1": 0.4, "motif_ETS": 0.35, "at_content": 0.53, "entropy": 1.95},
        "EnhA2": {"gc_content": 0.45, "cpg_ratio": 0.38, "cpg_freq": 0.028, "repeat_density": 0.12, "motif_AP1": 0.35, "motif_ETS": 0.30, "at_content": 0.55, "entropy": 1.95},
        "EnhWk": {"gc_content": 0.42, "cpg_ratio": 0.32, "cpg_freq": 0.022, "repeat_density": 0.18, "motif_AP1": 0.15, "motif_ETS": 0.15, "at_content": 0.58, "entropy": 1.95},
        "ZnfRpts": {"gc_content": 0.52, "cpg_ratio": 0.45, "cpg_freq": 0.035, "repeat_density": 0.35, "motif_KRAB_ZNF": 0.5, "at_content": 0.48, "entropy": 1.7},
        "Het": {"gc_content": 0.35, "cpg_ratio": 0.20, "cpg_freq": 0.010, "repeat_density": 0.45, "at_content": 0.65, "entropy": 1.3},
        "TssBiv": {"gc_content": 0.58, "cpg_ratio": 0.70, "cpg_freq": 0.060, "repeat_density": 0.08, "motif_CpG_dense": 0.5, "at_content": 0.42, "entropy": 1.85},
        "EnhBiv": {"gc_content": 0.48, "cpg_ratio": 0.45, "cpg_freq": 0.035, "repeat_density": 0.15, "motif_AP1": 0.2, "motif_ETS": 0.2, "at_content": 0.52, "entropy": 1.95},
        "ReprPC": {"gc_content": 0.50, "cpg_ratio": 0.55, "cpg_freq": 0.040, "repeat_density": 0.12, "motif_CpG_dense": 0.35, "at_content": 0.50, "entropy": 1.8},
        "ReprPCWk": {"gc_content": 0.45, "cpg_ratio": 0.40, "cpg_freq": 0.028, "repeat_density": 0.18, "at_content": 0.55, "entropy": 1.9},
        "Quies": {"gc_content": 0.40, "cpg_ratio": 0.25, "cpg_freq": 0.015, "repeat_density": 0.25, "at_content": 0.60, "entropy": 1.8},
    }


# -----------------------------------------------------------------------------
# 4) Profiling & Hungarian assignment
# -----------------------------------------------------------------------------

@dataclass
class IdentificationConfig:
    sequences_path: str
    labels_path: str
    output_dir: str = OUT_DIR_DEFAULT
    seed: int = 42
    max_per_label: int = 6000  # keep Colab runtime reasonable; increase for final
    key_features: Tuple[str, ...] = tuple(KEY_FEATURES)


def stratified_subsample_indices(labels: List[int], max_per_label: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels_arr = np.asarray(labels, dtype=np.int32)
    keep: List[int] = []
    for lab in range(1, 19):
        idx = np.where(labels_arr == lab)[0]
        if idx.size == 0:
            continue
        take = min(max_per_label, idx.size)
        keep.append(rng.choice(idx, size=take, replace=False))
    if not keep:
        return np.array([], dtype=np.int64)
    return np.concatenate(keep).astype(np.int64)


def compute_label_profiles(
    sequences: List[str],
    labels: List[int],
    max_per_label: int,
    seed: int,
    feature_names: List[str],
) -> Dict[int, Dict[str, float]]:
    idx = stratified_subsample_indices(labels, max_per_label=max_per_label, seed=seed)
    if idx.size == 0:
        raise ValueError("No samples found after subsampling (check labels).")

    # Accumulate sums for means
    sums: Dict[int, Dict[str, float]] = {lab: {k: 0.0 for k in feature_names} for lab in range(1, 19)}
    counts: Dict[int, int] = {lab: 0 for lab in range(1, 19)}

    for n, i in enumerate(idx.tolist()):
        if n % 5000 == 0:
            print(f"  feature extraction: {n}/{idx.size}")
        lab = int(labels[i])
        feats = extract_features(sequences[i])
        for k in feature_names:
            sums[lab][k] += float(feats.get(k, 0.0))
        counts[lab] += 1

    profiles: Dict[int, Dict[str, float]] = {}
    for lab in range(1, 19):
        if counts[lab] == 0:
            continue
        profiles[lab] = {k: sums[lab][k] / counts[lab] for k in feature_names}
    return profiles


def _build_matrices(
    label_profiles: Dict[int, Dict[str, float]],
    state_profiles: Dict[str, Dict[str, float]],
    labels_sorted: List[int],
    states_sorted: List[str],
    key_features: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    L = np.zeros((len(labels_sorted), len(key_features)), dtype=np.float64)
    S = np.zeros((len(states_sorted), len(key_features)), dtype=np.float64)
    for i, lab in enumerate(labels_sorted):
        prof = label_profiles.get(lab, {})
        for j, f in enumerate(key_features):
            L[i, j] = float(prof.get(f, 0.0))
    for i, st in enumerate(states_sorted):
        prof = state_profiles.get(st, {})
        for j, f in enumerate(key_features):
            S[i, j] = float(prof.get(f, 0.0))
    return L, S


def _similarity_matrix(
    label_profiles: Dict[int, Dict[str, float]],
    state_profiles: Dict[str, Dict[str, float]],
    labels_sorted: List[int],
    states_sorted: List[str],
    key_features: List[str],
) -> np.ndarray:
    from scipy.spatial.distance import cdist

    L, S = _build_matrices(label_profiles, state_profiles, labels_sorted, states_sorted, key_features)
    mu = L.mean(axis=0)
    sigma = L.std(axis=0) + 1e-8
    Lz = (L - mu) / sigma
    Sz = (S - mu) / sigma
    dist = cdist(Lz, Sz, metric="euclidean")
    return -dist


def hungarian_match(
    label_profiles: Dict[int, Dict[str, float]],
    state_profiles: Dict[str, Dict[str, float]],
    key_features: List[str],
) -> Tuple[Dict[int, str], np.ndarray, List[str]]:
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment

    labels_sorted = list(range(1, 19))
    states_sorted = list(state_profiles.keys())

    L, S = _build_matrices(label_profiles, state_profiles, labels_sorted, states_sorted, key_features)

    # Standardize using label distribution (keeps state prototypes as "targets" in same z-space)
    mu = L.mean(axis=0)
    sigma = L.std(axis=0) + 1e-8
    Lz = (L - mu) / sigma
    Sz = (S - mu) / sigma

    dist = cdist(Lz, Sz, metric="euclidean")
    sim = -dist

    row_ind, col_ind = linear_sum_assignment(dist)
    mapping: Dict[int, str] = {}
    for r, c in zip(row_ind.tolist(), col_ind.tolist()):
        mapping[labels_sorted[r]] = states_sorted[c]

    return mapping, sim, states_sorted


def infer_anchor_assignments(
    label_profiles: Dict[int, Dict[str, float]],
    state_profiles: Dict[str, Dict[str, float]],
) -> Dict[str, int]:
    """
    Heuristic anchors to stabilize assignment:
      - TssA: highest CpG ratio (and/or CpG freq)
      - Het: very low GC + low entropy + high repeats
      - Quies: closest to Quies prototype among remaining
      - ZnfRpts: closest to ZnfRpts prototype among remaining
    Returns mapping: state_name -> label_id
    """
    labels = list(range(1, 19))

    def v(lab: int, key: str, default: float = 0.0) -> float:
        return float(label_profiles.get(lab, {}).get(key, default))

    # Anchor 1: TssA
    tssA = max(labels, key=lambda lab: (v(lab, "cpg_ratio"), v(lab, "cpg_freq"), v(lab, "gc_content")))

    remaining = [lab for lab in labels if lab != tssA]

    # Anchor 2: Het (low GC, low entropy, high repeats)
    # Take bottom-4 GC as candidates, then pick the one with max (repeat_density - entropy*0.1)
    gc_sorted = sorted(remaining, key=lambda lab: v(lab, "gc_content"))
    het_pool = gc_sorted[:4] if len(gc_sorted) >= 4 else gc_sorted
    het = max(het_pool, key=lambda lab: (v(lab, "repeat_density") - 0.10 * v(lab, "entropy"), -v(lab, "gc_content")))

    remaining = [lab for lab in remaining if lab != het]

    # Helper: closest-by-safe-features to a prototype
    safe_feats = SAFE_FEATURES
    state_names = list(state_profiles.keys())

    def dist_to_state(lab: int, state: str) -> float:
        # Euclidean in raw safe-feature space (ok for ranking; main similarity still z-scored)
        x = np.array([v(lab, f) for f in safe_feats], dtype=np.float64)
        y = np.array([float(state_profiles[state].get(f, 0.0)) for f in safe_feats], dtype=np.float64)
        return float(np.linalg.norm(x - y))

    # Anchor 3: Quies
    quies = min(remaining, key=lambda lab: dist_to_state(lab, "Quies"))
    remaining = [lab for lab in remaining if lab != quies]

    # Anchor 4: ZnfRpts
    znf = min(remaining, key=lambda lab: dist_to_state(lab, "ZnfRpts"))

    return {"TssA": tssA, "Het": het, "Quies": quies, "ZnfRpts": znf}


def constrained_assignment(
    label_profiles: Dict[int, Dict[str, float]],
    state_profiles: Dict[str, Dict[str, float]],
    key_features: List[str],
    anchors: Dict[str, int],
) -> Tuple[Dict[int, str], np.ndarray, List[str], Dict[str, int]]:
    """
    Fix anchor states to chosen labels, then run Hungarian assignment on remaining labels/states.
    Returns full mapping over labels 1..18.
    """
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment

    all_labels = list(range(1, 19))
    all_states = list(state_profiles.keys())

    anchored_states = [s for s in anchors.keys() if s in all_states]
    anchored_labels = [int(anchors[s]) for s in anchored_states]

    rem_labels = [lab for lab in all_labels if lab not in anchored_labels]
    rem_states = [st for st in all_states if st not in anchored_states]

    # Build matrices for remaining
    L, S = _build_matrices(label_profiles, state_profiles, rem_labels, rem_states, key_features)
    mu = L.mean(axis=0)
    sigma = L.std(axis=0) + 1e-8
    Lz = (L - mu) / sigma
    Sz = (S - mu) / sigma
    dist = cdist(Lz, Sz, metric="euclidean")
    sim_rem = -dist

    row_ind, col_ind = linear_sum_assignment(dist)
    mapping: Dict[int, str] = {}

    # Set anchors
    for st in anchored_states:
        mapping[int(anchors[st])] = st

    # Fill remaining by assignment
    for r, c in zip(row_ind.tolist(), col_ind.tolist()):
        mapping[int(rem_labels[r])] = rem_states[int(c)]

    # Full similarity matrix for reporting (18x18) in the original state order
    full_sim = _similarity_matrix(
        label_profiles=label_profiles,
        state_profiles=state_profiles,
        labels_sorted=list(range(1, 19)),
        states_sorted=all_states,
        key_features=key_features,
    )

    return mapping, full_sim, all_states, anchors


def confusion_consistency_score(
    phase6_confusion_json_path: str,
    label_to_state: Dict[int, str],
) -> Dict[str, object]:
    """
    Uses phase6 confusion pairs as a weak validation signal:
    "Frequently confused labels should map to states in the same coarse family."
    """
    with open(phase6_confusion_json_path, "r") as f:
        payload = json.load(f)

    pairs = payload.get("confused_pairs", [])

    def family(state: str) -> str:
        if state.startswith("Tss") or state.startswith("ReprPC"):
            return "promoter_polycomb"
        if state.startswith("Enh"):
            return "enhancer"
        if state.startswith("Tx"):
            return "transcribed"
        if state in ("Het", "Quies", "ZnfRpts"):
            return "other"
        return "other"

    total_w = 0.0
    ok_w = 0.0
    annotated = []
    for p in pairs:
        # phase6 file uses 0-based labels
        li = int(p["label_i"]) + 1
        lj = int(p["label_j"]) + 1
        w = float(p.get("score", 1.0))
        si = label_to_state.get(li, "Unknown")
        sj = label_to_state.get(lj, "Unknown")
        fi = family(si)
        fj = family(sj)
        ok = fi == fj and si != "Unknown" and sj != "Unknown"
        total_w += w
        ok_w += (w if ok else 0.0)
        annotated.append(
            {
                "label_i": li,
                "label_j": lj,
                "score": w,
                "state_i": si,
                "state_j": sj,
                "family_i": fi,
                "family_j": fj,
                "same_family": ok,
            }
        )

    return {
        "weighted_same_family_fraction": (ok_w / total_w) if total_w > 0 else None,
        "total_weight": total_w,
        "pairs": annotated,
    }


def generate_report(
    label_profiles: Dict[int, Dict[str, float]],
    mapping: Dict[int, str],
    sim: np.ndarray,
    state_names: List[str],
    key_feats_for_table: Optional[List[str]] = None,
    anchors: Optional[Dict[str, int]] = None,
) -> str:
    if key_feats_for_table is None:
        key_feats_for_table = ["gc_content", "cpg_ratio", "repeat_density", "cpg_freq", "entropy"]

    lines: List[str] = []
    lines.append("=" * 88)
    lines.append("PHASE 8: CHROMATIN LABEL → BIOLOGICAL STATE IDENTIFICATION REPORT")
    lines.append("=" * 88)
    lines.append("")
    title = "LABEL → STATE (Hungarian assignment on normalized feature profiles)"
    if anchors:
        title = "LABEL → STATE (Anchors + Hungarian assignment on normalized feature profiles)"
    lines.append(title)
    lines.append("-" * 88)
    if anchors:
        lines.append("Anchors used (state -> label): " + ", ".join([f"{k}->{v}" for k, v in anchors.items()]))
        lines.append("")

    for lab in range(1, 19):
        st = mapping.get(lab, "Unknown")
        # confidence proxy: margin between best and second best similarity
        sims = sim[lab - 1]
        order = np.argsort(-sims)
        best, second = float(sims[order[0]]), float(sims[order[1]])
        margin = best - second
        top1_state = state_names[int(order[0])] if state_names else "?"
        forced = "" if st == top1_state else f"  [not top1={top1_state}]"
        lines.append(f"  Label {lab:2d} → {st:10s}   (margin={margin: .3f}, best_sim={best: .3f}){forced}")

    lines.append("")
    lines.append("Per-label feature centroids (subset)")
    lines.append("-" * 88)
    header = f"{'Label':>6} {'State':>10} | " + " | ".join(f"{f:>14}" for f in key_feats_for_table)
    lines.append(header)
    lines.append("-" * len(header))
    for lab in range(1, 19):
        st = mapping.get(lab, "?")
        prof = label_profiles.get(lab, {})
        vals = [f"{prof.get(f, 0.0):14.4f}" for f in key_feats_for_table]
        lines.append(f"{lab:6d} {st:>10} | " + " | ".join(vals))

    lines.append("")
    lines.append("Top-3 candidate states per label (by similarity)")
    lines.append("-" * 88)
    for lab in range(1, 19):
        sims = sim[lab - 1]
        order = np.argsort(-sims)[:3]
        trip = ", ".join([f"{state_names[i]}({sims[i]:.3f})" for i in order.tolist()])
        lines.append(f"  Label {lab:2d}: {trip}")

    # Sanity checks
    lines.append("")
    lines.append("Sanity checks (expected extremes)")
    lines.append("-" * 88)
    gc = {lab: label_profiles.get(lab, {}).get("gc_content", 0.0) for lab in range(1, 19)}
    cpg = {lab: label_profiles.get(lab, {}).get("cpg_ratio", 0.0) for lab in range(1, 19)}
    rep = {lab: label_profiles.get(lab, {}).get("repeat_density", 0.0) for lab in range(1, 19)}
    lines.append(f"  Highest CpG ratio label: {max(cpg, key=cpg.get)}  (mapped to {mapping.get(max(cpg, key=cpg.get), '?')})")
    lines.append(f"  Lowest  GC content label: {min(gc, key=gc.get)}  (mapped to {mapping.get(min(gc, key=gc.get), '?')})")
    lines.append(f"  Highest repeat density label: {max(rep, key=rep.get)} (mapped to {mapping.get(max(rep, key=rep.get), '?')})")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# 5) Quick visualizations (same spirit as phase8/visualization.py)
# -----------------------------------------------------------------------------

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def quick_feature_df(
    sequences: List[str],
    labels: List[int],
    sample_total: int = 30000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(sequences)
    take = min(sample_total, n)
    idx = rng.choice(np.arange(n), size=take, replace=False)
    rows = []
    for t, i in enumerate(idx.tolist()):
        if t % 5000 == 0:
            print(f"  quick-viz features: {t}/{take}")
        seq = sequences[i]
        lab = int(labels[i])
        rows.append(
            {
                "label": lab,
                "gc_content": compute_gc_content(seq),
                "cpg_ratio": compute_cpg_ratio(seq),
                "repeat_density": compute_repeat_density(seq),
                "at_content": 1.0 - compute_gc_content(seq),
            }
        )
    return pd.DataFrame(rows)


def save_quick_plots(df: pd.DataFrame, output_dir: str) -> None:
    import matplotlib.pyplot as plt

    _ensure_dir(output_dir)

    # Violin plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    features = ["gc_content", "cpg_ratio", "repeat_density", "at_content"]
    titles = ["GC Content", "CpG O/E Ratio", "Repeat Density", "AT Content"]
    for ax, feat, title in zip(axes.flatten(), features, titles):
        positions = list(range(1, 19))
        data = [df[df["label"] == l][feat].values for l in positions]
        ax.violinplot(data, positions=positions, showmeans=True)
        ax.set_xlabel("Label")
        ax.set_ylabel(title)
        ax.set_title(f"{title} distribution by label")
        ax.set_xticks(positions)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_distributions.png", dpi=150)
    plt.close()

    # Scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    for label in range(1, 19):
        subset = df[df["label"] == label]
        ax.scatter(subset["gc_content"], subset["cpg_ratio"], alpha=0.25, s=10, label=str(label))
    ax.set_xlabel("GC Content")
    ax.set_ylabel("CpG O/E Ratio")
    ax.set_title("GC vs CpG ratio")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    ax = axes[1]
    for label in range(1, 19):
        subset = df[df["label"] == label]
        ax.scatter(subset["gc_content"], subset["repeat_density"], alpha=0.25, s=10, label=str(label))
    ax.set_xlabel("GC Content")
    ax.set_ylabel("Repeat Density")
    ax.set_title("GC vs repeat density")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/label_scatter.png", dpi=150)
    plt.close()

    # Centroids plot
    centroids = df.groupby("label")[["gc_content", "cpg_ratio", "repeat_density"]].mean()
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.scatter(centroids["gc_content"], centroids["cpg_ratio"], s=200, edgecolors="black")
    for label in centroids.index:
        ax.annotate(
            str(int(label)),
            (centroids.loc[label, "gc_content"], centroids.loc[label, "cpg_ratio"]),
            fontsize=10,
            ha="center",
            va="center",
        )
    ax.axvline(x=0.60, color="red", linestyle="--", alpha=0.5, label="Promoter-ish GC")
    ax.axhline(y=0.60, color="blue", linestyle="--", alpha=0.5, label="CpG island-ish")
    ax.axvline(x=0.38, color="gray", linestyle="--", alpha=0.5, label="Het-ish GC")
    ax.set_xlabel("Mean GC")
    ax.set_ylabel("Mean CpG O/E")
    ax.set_title("Label centroids (GC vs CpG)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/label_centroids.png", dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# 6) Main
# -----------------------------------------------------------------------------

def main(
    sequences_path: str = f"{DATA_DIR_DEFAULT}/trainsequences.csv",
    labels_path: str = f"{DATA_DIR_DEFAULT}/trainlabels.csv",
    output_dir: str = OUT_DIR_DEFAULT,
    max_per_label: int = 6000,
    seed: int = 42,
    quick_viz_sample: int = 30000,
    feature_set: str = "safe",  # safe|full
    use_anchors: bool = True,
    phase6_confusion_json: Optional[str] = None,
) -> Dict[int, str]:
    set_seed(seed)
    _ensure_dir(output_dir)

    print("Loading data...")
    sequences, labels = load_sequences_and_labels(sequences_path, labels_path)
    print(f"Loaded {len(sequences):,} sequences.")
    print(f"Labels range: [{min(labels)}, {max(labels)}], unique={len(set(labels))}")

    feature_set = feature_set.lower().strip()
    if feature_set not in ("safe", "full"):
        raise ValueError("feature_set must be 'safe' or 'full'")
    feats = SAFE_FEATURES if feature_set == "safe" else KEY_FEATURES

    print("\nComputing per-label feature profiles (stratified subsample)...")
    label_profiles = compute_label_profiles(
        sequences=sequences,
        labels=labels,
        max_per_label=max_per_label,
        seed=seed,
        feature_names=feats,
    )

    print(f"\nMatching labels to known states ({'anchors + hungarian' if use_anchors else 'hungarian'})...")
    states = expected_state_profiles()
    anchors: Optional[Dict[str, int]] = None
    if use_anchors:
        anchors = infer_anchor_assignments(label_profiles, states)
        mapping, sim, state_names, anchors = constrained_assignment(
            label_profiles=label_profiles,
            state_profiles=states,
            key_features=list(feats),
            anchors=anchors,
        )
    else:
        mapping, sim, state_names = hungarian_match(
            label_profiles=label_profiles,
            state_profiles=states,
            key_features=list(feats),
        )

    report = generate_report(
        label_profiles=label_profiles,
        mapping=mapping,
        sim=sim,
        state_names=state_names,
        anchors=anchors,
    )
    print("\n" + report)

    # Save outputs
    with open(f"{output_dir}/identification_report.txt", "w") as f:
        f.write(report)

    mapping_json = {str(k): v for k, v in mapping.items()}
    with open(f"{output_dir}/label_state_mapping.json", "w") as f:
        json.dump(mapping_json, f, indent=2)

    pd.DataFrame(label_profiles).T.to_csv(f"{output_dir}/label_feature_profiles.csv", index=True)
    np.save(f"{output_dir}/similarity_matrix.npy", sim)

    # Optional: validate against Phase 6 confusion structure
    if phase6_confusion_json:
        try:
            cc = confusion_consistency_score(phase6_confusion_json, mapping)
            with open(f"{output_dir}/confusion_consistency.json", "w") as f:
                json.dump(cc, f, indent=2)
            print(
                "\nConfusion-consistency (weighted same-family fraction): "
                f"{cc.get('weighted_same_family_fraction')}"
            )
        except Exception as e:
            print(f"\nWarning: failed to compute confusion consistency: {e}")

    # Quick sanity visualizations
    print("\nGenerating quick visualizations (sampled)...")
    df = quick_feature_df(sequences, labels, sample_total=quick_viz_sample, seed=seed)
    df.to_csv(f"{output_dir}/feature_data.csv", index=False)
    save_quick_plots(df, output_dir)

    print(f"\nDone. Outputs written to: {output_dir}")
    return mapping


if __name__ == "__main__":
    # Keep CLI light; Colab users can just run without args.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", default=f"{DATA_DIR_DEFAULT}/trainsequences.csv")
    parser.add_argument("--labels", default=f"{DATA_DIR_DEFAULT}/trainlabels.csv")
    parser.add_argument("--output", default=OUT_DIR_DEFAULT)
    parser.add_argument("--max_per_label", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick_viz_sample", type=int, default=30000)
    parser.add_argument("--feature_set", default="safe", choices=["safe", "full"])
    parser.add_argument("--use_anchors", action="store_true")
    parser.add_argument("--no_anchors", action="store_true")
    parser.add_argument("--phase6_confusion_json", default=None)
    # In notebooks/Colab, IPython injects extra args like:
    #   -f /path/to/kernel.json
    # Use parse_known_args() so the script can be pasted/run in a cell safely.
    args, _unknown = parser.parse_known_args()

    use_anchors = True
    if args.use_anchors:
        use_anchors = True
    if args.no_anchors:
        use_anchors = False

    main(
        sequences_path=args.sequences,
        labels_path=args.labels,
        output_dir=args.output,
        max_per_label=args.max_per_label,
        seed=args.seed,
        quick_viz_sample=args.quick_viz_sample,
        feature_set=args.feature_set,
        use_anchors=use_anchors,
        phase6_confusion_json=args.phase6_confusion_json,
    )


