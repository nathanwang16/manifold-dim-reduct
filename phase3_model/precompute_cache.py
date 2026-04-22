"""Pre-compute one-hot + engineered-feature caches for fast DDP startup.

Produces `.npy` files next to the dataset CSVs:
    {merged_dir}/train_onehot_cache.npy       (N, 200, 4) float32
    {merged_dir}/train_features_cache.npy     (N, 5)      float32
    {merged_dir}/train_labels_cache.npy       (N,)        int64 (0-indexed)

During training each DDP rank mmap-loads these files, so physical pages are
shared across ranks via the page cache.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chromatin_lib import (  # noqa: E402
    compute_engineered_features,
    merged_split_paths,
    one_hot_encode_sequence,
)


def build_cache(split: str, force: bool = False) -> None:
    paths = merged_split_paths(split)
    seq_path = paths["sequences"]
    lab_path = paths["labels"]
    out_dir = seq_path.parent
    onehot_path = out_dir / f"{split}_onehot_cache.npy"
    feat_path = out_dir / f"{split}_features_cache.npy"
    labels_path = out_dir / f"{split}_labels_cache.npy"

    if not force and onehot_path.exists() and feat_path.exists() and labels_path.exists():
        print(f"[{split}] cache exists -> skipping (use --force to rebuild)")
        return

    t0 = time.time()
    print(f"[{split}] Reading labels from {lab_path} ...")
    with open(lab_path) as f:
        labels_raw = np.fromiter((int(x) for x in f.read().splitlines()), dtype=np.int64)
    labels = (labels_raw - 1).astype(np.int64)
    n = labels.shape[0]
    print(f"[{split}]   {n:,} rows")

    print(f"[{split}] Allocating output arrays ({n*200*4*4/1e9:.2f} GB one-hot, {n*5*4/1e6:.1f} MB feat)")
    onehot = np.lib.format.open_memmap(onehot_path, mode="w+", dtype=np.float32, shape=(n, 200, 4))
    feats = np.lib.format.open_memmap(feat_path, mode="w+", dtype=np.float32, shape=(n, 5))

    print(f"[{split}] Encoding sequences from {seq_path} ...")
    with open(seq_path) as f:
        for i, line in enumerate(tqdm(f, total=n, desc=f"{split} encode")):
            seq = line.strip()
            onehot[i] = one_hot_encode_sequence(seq, 200)
            feats[i] = compute_engineered_features(seq)
            if i + 1 >= n:
                break
    onehot.flush()
    feats.flush()
    del onehot, feats

    np.save(labels_path, labels)
    print(f"[{split}] Done in {time.time()-t0:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["train"],
                        choices=["train", "val", "test"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for split in args.splits:
        build_cache(split, force=args.force)


if __name__ == "__main__":
    main()
