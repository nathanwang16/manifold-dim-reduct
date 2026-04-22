"""Build index arrays restricted to a given set of Roadmap epigenome IDs.

Streams the meta CSV to collect row indices whose `eid` column matches any
of the supplied IDs. Writes an int64 `.npy` array (0-indexed into the
corresponding sequence/label CSV).

Useful for cell-type-specific analyses in phases 2/6/8.

Examples:
    python phase1_filter/filter_by_eid.py --split train --eids E003 E034 E123 \
        --output phase1_filter/outputs/train_stem_cells.npy
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, Set

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chromatin_lib import merged_split_paths  # noqa: E402


def iter_eid_mask(meta_path: Path, eids: Set[str]) -> Iterable[bool]:
    with open(meta_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield row["eid"] in eids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--eids", nargs="+", required=True,
                        help="Roadmap epigenome IDs (e.g. E003 E034).")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = merged_split_paths(args.split)
    if not paths["meta"].exists():
        raise FileNotFoundError(f"Missing meta file: {paths['meta']}")
    keep: list[int] = []
    for i, match in enumerate(iter_eid_mask(paths["meta"], set(args.eids))):
        if match:
            keep.append(i)
    arr = np.asarray(keep, dtype=np.int64)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, arr)
    print(f"[{args.split}] {len(keep):,} rows match eids={sorted(args.eids)} -> {args.output}")


if __name__ == "__main__":
    main()
