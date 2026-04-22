"""Orchestrate phase 1: emit hierarchy label files + balanced subsample indices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from phase1_filter.extract_hierarchy_labels import emit_split  # noqa: E402
from phase1_filter.build_subsamples import build  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Subset of splits to process.",
    )
    parser.add_argument("--skip_hierarchy", action="store_true")
    parser.add_argument("--skip_subsamples", action="store_true")
    parser.add_argument("--from_meta", action="store_true",
                        help="Source hierarchy from meta (slow audit mode).")
    parser.add_argument("--val_scan_rows", type=int, default=None,
                        help="Limit val rows scanned for reservoir sampling (debug).")
    parser.add_argument("--test_scan_rows", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    output_dir = Path(cfg["phase1"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_hierarchy:
        print("== Extracting hierarchy labels ==")
        for split in args.splits:
            emit_split(split, output_dir, use_meta=args.from_meta)

    if not args.skip_subsamples:
        print("== Building balanced subsample indices ==")
        seed = int(cfg["phase1"]["seed"])
        if "train" in args.splits:
            build("train", int(cfg["phase1"]["viz_subsample_per_class"]), seed,
                  output_dir, tag="viz")
        if "val" in args.splits:
            build("val", int(cfg["phase1"]["eval_subsample_per_class"]), seed,
                  output_dir, max_rows=args.val_scan_rows, tag="balanced")
        if "test" in args.splits:
            build("test", int(cfg["phase1"]["eval_subsample_per_class"]), seed,
                  output_dir, max_rows=args.test_scan_rows, tag="balanced")

    print("Phase 1 complete.")


if __name__ == "__main__":
    main()
