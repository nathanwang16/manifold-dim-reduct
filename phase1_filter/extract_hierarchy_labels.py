"""Emit one-column family / subcluster label CSVs for each split.

The new merged meta CSVs already have `family` and `subcluster` columns, but
lots of downstream code still expects a plain one-column file alongside
`{split}_labels.csv`. This script produces:

    {output_dir}/{split}_family_labels.csv       # 0-indexed family ids (0..5)
    {output_dir}/{split}_subcluster_labels.csv   # 0-indexed subcluster ids (0..6)

It streams the meta file (val/test are ~5 GB) so memory stays bounded.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chromatin_lib import (  # noqa: E402
    FAMILY_TO_ID,
    LABEL_TO_FAMILY,
    LABEL_TO_SUBCLUSTER,
    merged_split_paths,
)


def _iter_meta_rows(path: Path) -> Iterable[dict]:
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield row


def emit_split(split: str, output_dir: Path, use_meta: bool = False) -> None:
    """Emit family/subcluster label files for `split`.

    By default we derive from the single-column labels file (O(N) fast read),
    which produces identical output to the meta-derived version because the
    DATASET.md mapping is one-to-one. Pass `use_meta=True` to source from meta
    as a correctness audit (~100x slower on val/test).
    """
    paths = merged_split_paths(split)
    output_dir.mkdir(parents=True, exist_ok=True)
    fam_out = output_dir / f"{split}_family_labels.csv"
    sub_out = output_dir / f"{split}_subcluster_labels.csv"

    if use_meta and paths["meta"].exists():
        with open(fam_out, "w") as ff, open(sub_out, "w") as fs:
            for row in _iter_meta_rows(paths["meta"]):
                fam = FAMILY_TO_ID[row["family"]]
                sub = int(row["subcluster"]) - 1
                ff.write(f"{fam}\n")
                fs.write(f"{sub}\n")
        print(f"[{split}] wrote {fam_out} and {sub_out} from meta")
        return

    with open(paths["labels"], "r") as fl, open(fam_out, "w") as ff, open(sub_out, "w") as fs:
        for line in fl:
            y = int(line.strip()) - 1
            ff.write(f"{int(LABEL_TO_FAMILY[y])}\n")
            fs.write(f"{int(LABEL_TO_SUBCLUSTER[y])}\n")
    print(f"[{split}] wrote {fam_out} and {sub_out} from labels")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="phase1_filter/outputs")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument(
        "--from_meta",
        action="store_true",
        default=True,
        help="Source family/subcluster from the meta CSVs (default). "
             "Disable with --labels_only to derive purely from label values.",
    )
    parser.add_argument(
        "--labels_only",
        action="store_false",
        dest="from_meta",
        help="Derive hierarchy from the integer label files rather than meta.",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    for split in args.splits:
        emit_split(split, out, use_meta=args.from_meta)


if __name__ == "__main__":
    main()
