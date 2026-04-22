"""Build reusable balanced index arrays for downstream phases.

val/test are massively unbalanced (quiescent dominates). For every MI /
diagnostics / steering experiment we want a reproducible balanced subset.
This script emits:

    {output_dir}/{split}_balanced_{per_class}perclass_seed{seed}.npy

Each file is an int64 numpy array of row indices into the corresponding
sequence/label CSV (0-indexed, unsorted).

Also emits a small stratified "viz" subsample from `train` for phase 2
manifold learning:

    {output_dir}/train_viz_{per_class}perclass_seed{seed}.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chromatin_lib import merged_split_paths, stratified_label_indices  # noqa: E402


def build(
    split: str,
    per_class: int,
    seed: int,
    output_dir: Path,
    max_rows: int | None = None,
    tag: str = "balanced",
) -> Path:
    paths = merged_split_paths(split)
    idx = stratified_label_indices(
        paths["labels"],
        samples_per_class=per_class,
        n_classes=18,
        seed=seed,
        max_rows=max_rows,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{split}_{tag}_{per_class}perclass_seed{seed}.npy"
    np.save(out, idx)
    print(f"[{split}] {tag}: wrote {idx.size:,} indices -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="phase1_filter/outputs")
    parser.add_argument("--seed", type=int, default=20260408)
    parser.add_argument("--train_viz_per_class", type=int, default=5000)
    parser.add_argument("--val_balanced_per_class", type=int, default=20000)
    parser.add_argument("--test_balanced_per_class", type=int, default=20000)
    parser.add_argument(
        "--val_scan_rows",
        type=int,
        default=None,
        help="Optionally cap how many val rows to scan during reservoir sampling "
             "(default: scan full 64M rows).",
    )
    parser.add_argument(
        "--test_scan_rows",
        type=int,
        default=None,
        help="Optionally cap how many test rows to scan.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    build("train", args.train_viz_per_class, args.seed, output_dir, tag="viz")
    build("val", args.val_balanced_per_class, args.seed, output_dir,
          max_rows=args.val_scan_rows, tag="balanced")
    build("test", args.test_balanced_per_class, args.seed, output_dir,
          max_rows=args.test_scan_rows, tag="balanced")


if __name__ == "__main__":
    main()
