"""Dataset classes for the roadmap_18state_full corpus.

Two strategies, selected per split:

1. `InMemoryChromatinDataset` — loads every sequence/label into RAM. The
   merged `train_*.csv` is ~210 MB and decodes to ~800 MB one-hot (1.05M *
   200 * 4 float32). Perfect for the balanced train split.

2. `StreamingChromatinDataset` — memory-maps sequence / label files and
   reads lines on demand. Works for the 63 M-row val / test splits without
   exploding RAM. Supports two access patterns:
       - random-access `__getitem__` (via a line-offset index built once)
       - stratified subsampling for "balanced val" eval scenarios.

Both support optional meta alignment (family/subcluster) without loading
the full metadata CSV when only labels are needed.
"""

from __future__ import annotations

import mmap
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .encoding import (
    BASE_TO_IDX,
    one_hot_encode_sequence,
    reverse_complement,
    reverse_complement_onehot,
)
from .features import compute_engineered_features
from .hierarchy import LABEL_TO_FAMILY, LABEL_TO_SUBCLUSTER


# ---------------------------------------------------------------------------
# Line-offset indexing for streaming
# ---------------------------------------------------------------------------

def build_line_offset_index(path: Path, cache_path: Optional[Path] = None) -> np.ndarray:
    """Return an int64 array of byte offsets for every newline-terminated line.

    `offsets[i]` is the byte offset of the start of line i. Cached alongside
    the file as `.offsets.npy` for reuse.

    Uses buffered streaming + vectorized numpy scanning so very large files
    (val / test ~ 12 GB) are indexed without loading them fully into RAM.
    """
    path = Path(path)
    if cache_path is None:
        cache_path = path.with_suffix(path.suffix + ".offsets.npy")
    if cache_path.exists() and cache_path.stat().st_mtime >= path.stat().st_mtime:
        return np.load(cache_path, mmap_mode="r")

    offsets: List[np.ndarray] = [np.array([0], dtype=np.int64)]
    base = 0
    chunk_size = 64 * 1024 * 1024  # 64 MB
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            arr = np.frombuffer(chunk, dtype=np.uint8)
            hits = np.flatnonzero(arr == 0x0A)
            if hits.size:
                offsets.append(hits.astype(np.int64) + 1 + base)
            base += len(chunk)
    idx = np.concatenate(offsets)
    # The file ends with a newline; drop the sentinel that would point past EOF.
    if idx.size and idx[-1] >= base:
        idx = idx[:-1]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, idx)
    return np.load(cache_path, mmap_mode="r")


# ---------------------------------------------------------------------------
# In-memory dataset
# ---------------------------------------------------------------------------

class InMemoryChromatinDataset(Dataset):
    """Loads sequences (+ optional labels/family/subcluster) into RAM.

    If `cache_onehot=True` the full one-hot tensor is allocated up-front
    (useful for small splits; 1.05M train uses ~0.8 GB). Otherwise the
    encoding is done lazily per sample.
    """

    def __init__(
        self,
        sequences_path: str | Path,
        labels_path: Optional[str | Path] = None,
        family_ids: Optional[np.ndarray] = None,
        subcluster_ids: Optional[np.ndarray] = None,
        jitter_prob: float = 0.0,
        jitter_min_len: int = 170,
        noise_prob: float = 0.0,
        rc_prob: float = 0.0,
        sequence_length: int = 200,
        cache_onehot: bool = True,
        compute_features: bool = False,
        max_samples: Optional[int] = None,
        sample_indices: Optional[np.ndarray] = None,
    ):
        self.sequence_length = sequence_length
        self.jitter_prob = jitter_prob
        self.jitter_min_len = jitter_min_len
        self.noise_prob = noise_prob
        self.rc_prob = rc_prob
        self.compute_features = compute_features

        sequences_path = Path(sequences_path)
        with open(sequences_path, "r") as f:
            lines = f.read().splitlines()
        if sample_indices is not None:
            lines = [lines[i] for i in sample_indices]
        elif max_samples is not None:
            lines = lines[:max_samples]
        self.sequences: np.ndarray = np.array(lines, dtype=object)

        self.labels: Optional[np.ndarray] = None
        if labels_path is not None:
            with open(labels_path, "r") as f:
                raw = np.fromiter((int(x) for x in f.read().splitlines()), dtype=np.int64)
            if sample_indices is not None:
                raw = raw[sample_indices]
            elif max_samples is not None:
                raw = raw[:max_samples]
            self.labels = raw - 1  # 1..18 -> 0..17

        self.family_ids = family_ids
        self.subcluster_ids = subcluster_ids
        if self.family_ids is None and self.labels is not None:
            self.family_ids = LABEL_TO_FAMILY[self.labels]
        if self.subcluster_ids is None and self.labels is not None:
            self.subcluster_ids = LABEL_TO_SUBCLUSTER[self.labels]

        self._onehot: Optional[np.ndarray] = None
        if cache_onehot:
            self._onehot = np.zeros((len(self.sequences), sequence_length, 4), dtype=np.float32)
            for i, seq in enumerate(self.sequences):
                self._onehot[i] = one_hot_encode_sequence(seq, sequence_length)

        self._features: Optional[np.ndarray] = None
        if compute_features:
            self._features = np.zeros((len(self.sequences), 5), dtype=np.float32)
            for i, seq in enumerate(self.sequences):
                self._features[i] = compute_engineered_features(seq)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        if self._onehot is not None:
            x = self._onehot[idx].copy()
        else:
            x = one_hot_encode_sequence(self.sequences[idx], self.sequence_length)

        if self.rc_prob > 0 and random.random() < self.rc_prob:
            x = reverse_complement_onehot(x)

        if self.jitter_prob > 0 and random.random() < self.jitter_prob:
            crop = random.randint(self.jitter_min_len, self.sequence_length)
            start = random.randint(0, self.sequence_length - crop)
            cropped = x[start : start + crop].copy()
            x = np.zeros_like(x)
            x[: cropped.shape[0]] = cropped

        if self.noise_prob > 0:
            mask = np.random.random((self.sequence_length, 4)) < self.noise_prob
            if mask.any():
                x[mask.any(axis=1)] = 0.25

        x_t = torch.from_numpy(x).float()
        out = {"x": x_t}
        if self.labels is not None:
            out["y"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
            out["family"] = torch.tensor(int(self.family_ids[idx]), dtype=torch.long)
            out["subcluster"] = torch.tensor(int(self.subcluster_ids[idx]), dtype=torch.long)
        if self._features is not None:
            out["feat"] = torch.from_numpy(self._features[idx]).float()
        return out


# ---------------------------------------------------------------------------
# Streaming dataset (mmap-backed random access)
# ---------------------------------------------------------------------------

class StreamingChromatinDataset(Dataset):
    """Random-access dataset backed by mmap + line-offset index.

    Useful for val/test (~12 GB each) where we only ever read a small subset
    per evaluation.
    """

    def __init__(
        self,
        sequences_path: str | Path,
        labels_path: Optional[str | Path] = None,
        sample_indices: Optional[np.ndarray] = None,
        sequence_length: int = 200,
        compute_features: bool = False,
    ):
        self.sequences_path = Path(sequences_path)
        self.labels_path = Path(labels_path) if labels_path else None
        self.sequence_length = sequence_length
        self.compute_features = compute_features

        self._seq_offsets = build_line_offset_index(self.sequences_path)
        self._lab_offsets = (
            build_line_offset_index(self.labels_path) if self.labels_path else None
        )
        if sample_indices is None:
            sample_indices = np.arange(len(self._seq_offsets), dtype=np.int64)
        self._indices = np.asarray(sample_indices, dtype=np.int64)

        # Open files lazily (per-worker friendly).
        self._seq_file = None
        self._seq_mmap = None
        self._lab_file = None
        self._lab_mmap = None

    def _ensure_mmap(self) -> None:
        if self._seq_mmap is None:
            self._seq_file = open(self.sequences_path, "rb")
            self._seq_mmap = mmap.mmap(
                self._seq_file.fileno(), 0, access=mmap.ACCESS_READ
            )
        if self._lab_mmap is None and self.labels_path is not None:
            self._lab_file = open(self.labels_path, "rb")
            self._lab_mmap = mmap.mmap(
                self._lab_file.fileno(), 0, access=mmap.ACCESS_READ
            )

    def __len__(self) -> int:
        return int(self._indices.shape[0])

    def _read_line(self, mm: mmap.mmap, offset: int) -> bytes:
        end = mm.find(b"\n", offset)
        if end == -1:
            end = len(mm)
        return bytes(mm[offset:end])

    def __getitem__(self, idx: int):
        self._ensure_mmap()
        row = int(self._indices[idx])
        seq_bytes = self._read_line(self._seq_mmap, int(self._seq_offsets[row]))
        seq = seq_bytes.decode("ascii", errors="replace").strip()
        x = one_hot_encode_sequence(seq, self.sequence_length)
        out = {"x": torch.from_numpy(x).float()}
        if self._lab_mmap is not None:
            lab_bytes = self._read_line(self._lab_mmap, int(self._lab_offsets[row]))
            y = int(lab_bytes.decode("ascii", errors="replace").strip()) - 1
            out["y"] = torch.tensor(y, dtype=torch.long)
            out["family"] = torch.tensor(int(LABEL_TO_FAMILY[y]), dtype=torch.long)
            out["subcluster"] = torch.tensor(int(LABEL_TO_SUBCLUSTER[y]), dtype=torch.long)
        if self.compute_features:
            out["feat"] = torch.from_numpy(compute_engineered_features(seq)).float()
        return out


# ---------------------------------------------------------------------------
# Stratified subsampling
# ---------------------------------------------------------------------------

class CachedMmapDataset(Dataset):
    """Loads precomputed one-hot/features/labels `.npy` files via mmap.

    Produced by `phase3_model/precompute_cache.py`. Physical pages are
    shared across DDP ranks via the Linux page cache, so RAM usage stays
    low even with many workers.

    Supports the same light augmentation as `InMemoryChromatinDataset`
    (jitter, per-base noise, RC flip). Each augmentation makes a local
    copy before mutating, so mmap-ed pages remain read-only.
    """

    def __init__(
        self,
        onehot_cache_path: str | Path,
        features_cache_path: Optional[str | Path] = None,
        labels_cache_path: Optional[str | Path] = None,
        sample_indices: Optional[np.ndarray] = None,
        jitter_prob: float = 0.0,
        jitter_min_len: int = 170,
        noise_prob: float = 0.0,
        rc_prob: float = 0.0,
        sequence_length: int = 200,
    ):
        self.sequence_length = sequence_length
        self.jitter_prob = jitter_prob
        self.jitter_min_len = jitter_min_len
        self.noise_prob = noise_prob
        self.rc_prob = rc_prob

        self._onehot = np.load(onehot_cache_path, mmap_mode="r")
        self._features = (
            np.load(features_cache_path, mmap_mode="r") if features_cache_path else None
        )
        self._labels = (
            np.load(labels_cache_path, mmap_mode="r") if labels_cache_path else None
        )

        if sample_indices is None:
            n = self._onehot.shape[0]
            self._indices = np.arange(n, dtype=np.int64)
        else:
            self._indices = np.asarray(sample_indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self._indices.shape[0])

    def __getitem__(self, idx: int):
        row = int(self._indices[idx])
        x = np.array(self._onehot[row], dtype=np.float32, copy=True)

        if self.rc_prob > 0 and random.random() < self.rc_prob:
            x = reverse_complement_onehot(x)

        if self.jitter_prob > 0 and random.random() < self.jitter_prob:
            crop = random.randint(self.jitter_min_len, self.sequence_length)
            start = random.randint(0, self.sequence_length - crop)
            cropped = x[start : start + crop].copy()
            x = np.zeros_like(x)
            x[: cropped.shape[0]] = cropped

        if self.noise_prob > 0:
            mask = np.random.random((self.sequence_length, 4)) < self.noise_prob
            if mask.any():
                x[mask.any(axis=1)] = 0.25

        out = {"x": torch.from_numpy(x).float()}
        if self._labels is not None:
            y = int(self._labels[row])
            out["y"] = torch.tensor(y, dtype=torch.long)
            out["family"] = torch.tensor(int(LABEL_TO_FAMILY[y]), dtype=torch.long)
            out["subcluster"] = torch.tensor(int(LABEL_TO_SUBCLUSTER[y]), dtype=torch.long)
        if self._features is not None:
            out["feat"] = torch.from_numpy(
                np.array(self._features[row], dtype=np.float32, copy=True)
            ).float()
        return out


def stratified_label_indices(
    labels_path: str | Path,
    samples_per_class: int,
    n_classes: int = 18,
    seed: int = 0,
    max_rows: Optional[int] = None,
) -> np.ndarray:
    """Reservoir-sample `samples_per_class` row indices per 1-indexed label.

    Returns a flat int64 array of indices (unsorted).
    """
    rng = np.random.default_rng(seed)
    pools: List[List[int]] = [[] for _ in range(n_classes)]
    counts = np.zeros(n_classes, dtype=np.int64)

    with open(labels_path, "r") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            try:
                y = int(line.strip()) - 1
            except ValueError:
                continue
            if not (0 <= y < n_classes):
                continue
            counts[y] += 1
            pool = pools[y]
            if len(pool) < samples_per_class:
                pool.append(i)
            else:
                j = rng.integers(0, counts[y])
                if j < samples_per_class:
                    pool[int(j)] = i

    return np.concatenate([np.asarray(p, dtype=np.int64) for p in pools])


def collate_chromatin(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Default collate: stack every key present in all samples."""
    keys = batch[0].keys()
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        out[k] = torch.stack([item[k] for item in batch], dim=0)
    return out
