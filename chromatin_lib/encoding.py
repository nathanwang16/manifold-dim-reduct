"""One-hot and reverse-complement utilities for 200 bp ACGT sequences."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import torch

BASES: tuple = ("A", "C", "G", "T")
BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
# Reverse complement channel permutation: A<->T (0<->3), C<->G (1<->2)
RC_PERM = np.array([3, 2, 1, 0], dtype=np.int64)


_TABLE = np.full(256, -1, dtype=np.int8)
for base, idx in BASE_TO_IDX.items():
    _TABLE[ord(base)] = idx
    _TABLE[ord(base.lower())] = idx


def one_hot_encode_sequence(sequence: str, length: int = 200) -> np.ndarray:
    """One-hot encode a DNA sequence as a (length, 4) float32 array.

    Unknown characters are encoded as uniform 0.25 across channels.
    """
    arr = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    if arr.shape[0] < length:
        padded = np.full(length, ord("N"), dtype=np.uint8)
        padded[: arr.shape[0]] = arr
        arr = padded
    else:
        arr = arr[:length]
    idx = _TABLE[arr]
    out = np.zeros((length, 4), dtype=np.float32)
    valid = idx >= 0
    out[np.arange(length)[valid], idx[valid]] = 1.0
    out[~valid, :] = 0.25
    return out


def one_hot_encode_batch(sequences: Sequence[str], length: int = 200) -> np.ndarray:
    """Vectorized batch one-hot: returns (N, length, 4) float32."""
    n = len(sequences)
    out = np.zeros((n, length, 4), dtype=np.float32)
    for i, seq in enumerate(sequences):
        out[i] = one_hot_encode_sequence(seq, length)
    return out


_COMPLEMENT = str.maketrans({"A": "T", "C": "G", "G": "C", "T": "A", "N": "N",
                             "a": "t", "c": "g", "g": "c", "t": "a", "n": "n"})


def reverse_complement(sequence: str) -> str:
    """Reverse complement of a DNA string."""
    return sequence.translate(_COMPLEMENT)[::-1]


def reverse_complement_onehot(onehot: np.ndarray | torch.Tensor):
    """Reverse-complement a one-hot array of shape (L, 4) or (B, L, 4).

    Also accepts (4, L) or (B, 4, L) channel-first tensors.
    """
    if isinstance(onehot, np.ndarray):
        if onehot.ndim == 2 and onehot.shape[-1] == 4:
            return onehot[::-1, RC_PERM].copy()
        if onehot.ndim == 3 and onehot.shape[-1] == 4:
            return onehot[:, ::-1, :][:, :, RC_PERM].copy()
        if onehot.ndim == 2 and onehot.shape[0] == 4:
            return onehot[RC_PERM, ::-1].copy()
        if onehot.ndim == 3 and onehot.shape[1] == 4:
            return onehot[:, RC_PERM, :][:, :, ::-1].copy()
        raise ValueError(f"Unexpected shape for one-hot RC: {onehot.shape}")

    if torch.is_tensor(onehot):
        if onehot.dim() == 2 and onehot.shape[-1] == 4:
            perm = torch.as_tensor(RC_PERM, device=onehot.device)
            return onehot.flip(0)[:, perm]
        if onehot.dim() == 3 and onehot.shape[-1] == 4:
            perm = torch.as_tensor(RC_PERM, device=onehot.device)
            return onehot.flip(1)[:, :, perm]
        if onehot.dim() == 2 and onehot.shape[0] == 4:
            perm = torch.as_tensor(RC_PERM, device=onehot.device)
            return onehot.flip(1)[perm, :]
        if onehot.dim() == 3 and onehot.shape[1] == 4:
            perm = torch.as_tensor(RC_PERM, device=onehot.device)
            return onehot.flip(2)[:, perm, :]
        raise ValueError(f"Unexpected shape for one-hot RC: {tuple(onehot.shape)}")

    raise TypeError(f"Unsupported type: {type(onehot)}")
