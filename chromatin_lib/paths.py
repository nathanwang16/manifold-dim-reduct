"""Canonical filesystem locations for the roadmap_18state_full dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "phase0_aggregate" / "data" / "roadmap_18state_full"
RESULTS_ROOT = REPO_ROOT / "results"

MERGED_DIR = DATA_ROOT / "processed" / "merged"
PER_EID_DIR = DATA_ROOT / "processed" / "per_epigenome"
QC_DIR = DATA_ROOT / "logs" / "qc"


def merged_split_paths(split: str) -> Dict[str, Path]:
    """Return paths for a merged split (train / val / test)."""
    split = split.lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split {split!r}; expected train|val|test.")
    base = MERGED_DIR
    return {
        "sequences": base / f"{split}_sequences.csv",
        "labels": base / f"{split}_labels.csv",
        "meta": base / f"{split}_meta.csv",
    }


def per_epigenome_paths(eid: str) -> Dict[str, Path]:
    return {
        "sequences": PER_EID_DIR / f"{eid}_sequences.csv",
        "labels": PER_EID_DIR / f"{eid}_labels.csv",
        "meta": PER_EID_DIR / f"{eid}_meta.csv.gz",
    }
