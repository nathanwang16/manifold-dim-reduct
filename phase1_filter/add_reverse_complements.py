"""
Duplicate train/val datasets with reverse-complement sequences.

Usage:
    python phase1_filter/add_reverse_complements.py --data_dir data --in_place

By default the script writes new files with a `_with_rc` suffix so you can
inspect them before overwriting the originals. Pass `--in_place` to replace the
existing CSVs (sequences, labels, and optional hierarchy labels) directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BASE_COMPLEMENT = str.maketrans(
    {
        "A": "T",
        "C": "G",
        "G": "C",
        "T": "A",
        "a": "t",
        "c": "g",
        "g": "c",
        "t": "a",
        "N": "N",
        "n": "n",
    }
)


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return seq.translate(BASE_COMPLEMENT)[::-1]


def load_sequences(path: Path) -> np.ndarray:
    df = pd.read_csv(path, header=None)
    return df.iloc[:, 0].astype(str).str.strip().values


def load_numeric_labels(path: Path) -> np.ndarray:
    df = pd.read_csv(path, header=None)
    out = []
    for item in df.iloc[:, 0].tolist():
        s = str(item).strip()
        try:
            out.append(int(s))
        except ValueError:
            out.append(int(float(s)))
    return np.asarray(out, dtype=np.int32)


def write_one_col(path: Path, values: Iterable) -> None:
    pd.DataFrame(values).to_csv(path, index=False, header=False)


def build_output_path(path: Path, suffix: str, in_place: bool) -> Path:
    if in_place:
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def augment_split(
    data_dir: Path,
    split: str,
    suffix: str,
    in_place: bool,
) -> None:
    seq_path = data_dir / f"{split}_sequences.csv"
    label_path = data_dir / f"{split}_labels.csv"
    if not seq_path.exists() or not label_path.exists():
        raise FileNotFoundError(f"Missing {split} CSVs in {data_dir}")

    sequences = load_sequences(seq_path)
    labels = load_numeric_labels(label_path)
    if len(sequences) != len(labels):
        raise ValueError(f"{split}: sequence/label length mismatch")

    rc_sequences = np.array([reverse_complement(seq) for seq in sequences])
    augmented_sequences = np.concatenate([sequences, rc_sequences])
    augmented_labels = np.concatenate([labels, labels])

    seq_out = build_output_path(seq_path, suffix, in_place)
    lab_out = build_output_path(label_path, suffix, in_place)
    write_one_col(seq_out, augmented_sequences)
    write_one_col(lab_out, augmented_labels)

    # Handle optional hierarchy labels if present.
    for extra in ("family", "subcluster"):
        extra_path = data_dir / f"{split}_{extra}_labels.csv"
        if not extra_path.exists():
            continue
        extra_vals = load_numeric_labels(extra_path)
        if len(extra_vals) != len(labels):
            raise ValueError(f"{split}_{extra}: length mismatch vs labels")
        extra_out = build_output_path(extra_path, suffix, in_place)
        write_one_col(extra_out, np.concatenate([extra_vals, extra_vals]))

    print(
        f"{split}: wrote {len(augmented_sequences)} sequences "
        f"({len(sequences)} originals + RC). "
        f"Output -> {seq_out.name}, {lab_out.name}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment splits with reverse complements.")
    parser.add_argument("--data_dir", default="data", help="Directory containing split CSVs")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Split prefixes to process (e.g., train val mi)",
    )
    parser.add_argument(
        "--suffix",
        default="_with_rc",
        help="Suffix for new files when not using --in_place",
    )
    parser.add_argument(
        "--in_place",
        action="store_true",
        help="Overwrite the existing CSVs instead of writing suffixed copies",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    for split in args.splits:
        augment_split(data_dir, split, args.suffix, args.in_place)

    print("Reverse-complement augmentation complete.")


if __name__ == "__main__":
    main()
