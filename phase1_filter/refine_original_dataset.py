"""
Phase 1 helper: refine the "no underscore" dataset in data/ and generate hierarchy labels.

What it does:
1) Reads:
   - data/trainsequences.csv
   - data/trainlabels.csv
   (both expected to be single-column, but 2-column CSVs are handled)
2) Optionally writes/refreshes the underscore split files:
   - data/train_sequences.csv, data/train_labels.csv
   - data/val_sequences.csv, data/val_labels.csv
   - data/split_indices.npz
3) Writes hierarchical labels (family/subcluster) for train+val:
   - data/train_family_labels.csv, data/train_subcluster_labels.csv
   - data/val_family_labels.csv,   data/val_subcluster_labels.csv
   - data/hierarchy_meta.json

Notes:
- Does NOT overwrite the original no-underscore files.
- Family labels are 0..(n_families-1), subcluster labels are 0..(max_subcluster-1).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _pick_best_sequence_column(df: pd.DataFrame) -> int:
    # Prefer column name containing 'seq', else choose column with longest mean length
    lowered = {str(c).lower(): i for i, c in enumerate(df.columns)}
    for k, i in lowered.items():
        if "seq" in k:
            return i
    best_i, best_len = 0, -1.0
    for i in range(df.shape[1]):
        mean_len = df.iloc[:, i].astype(str).str.len().mean()
        if mean_len > best_len:
            best_len = mean_len
            best_i = i
    return best_i


def _pick_best_label_column(df: pd.DataFrame) -> int:
    lowered = {str(c).lower(): i for i, c in enumerate(df.columns)}
    for k, i in lowered.items():
        if "label" in k:
            return i
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


def load_no_underscore(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    seq_path = data_dir / "trainsequences.csv"
    lab_path = data_dir / "trainlabels.csv"
    seq_df = pd.read_csv(seq_path)
    lab_df = pd.read_csv(lab_path)
    seq_col = _pick_best_sequence_column(seq_df)
    lab_col = _pick_best_label_column(lab_df)
    sequences = seq_df.iloc[:, seq_col].astype(str).str.strip().values

    labels_raw = lab_df.iloc[:, lab_col].values
    labels: List[int] = []
    for x in labels_raw:
        s = str(x).strip()
        try:
            labels.append(int(s))
        except Exception:
            labels.append(int(float(s)))
    labels_arr = np.asarray(labels, dtype=np.int32)

    if len(sequences) != len(labels_arr):
        raise ValueError(f"Length mismatch: {len(sequences)} sequences vs {len(labels_arr)} labels")

    # Normalize 0..17 to 1..18 if needed
    if labels_arr.min() == 0 and labels_arr.max() == 17:
        labels_arr = labels_arr + 1

    return sequences, labels_arr


def load_hierarchy(hierarchy_json: Path) -> Dict[int, Dict[str, object]]:
    with open(hierarchy_json, "r") as f:
        raw = json.load(f)
    out: Dict[int, Dict[str, object]] = {}
    for k, v in raw.items():
        out[int(k)] = v
    return out


def build_family_vocab(hierarchy: Dict[int, Dict[str, object]]) -> Tuple[Dict[str, int], List[str]]:
    families = sorted({str(v["family"]) for v in hierarchy.values()})
    fam_to_id = {name: i for i, name in enumerate(families)}
    return fam_to_id, families


def build_subcluster_vocab(hierarchy: Dict[int, Dict[str, object]]) -> Tuple[int, Dict[int, int]]:
    # map label -> subcluster_id (0-based)
    # if a label has subcluster 0, keep 0
    max_sc = 0
    label_to_sc: Dict[int, int] = {}
    for label, meta in hierarchy.items():
        sc = int(meta.get("subcluster", 0))
        max_sc = max(max_sc, sc)
        label_to_sc[label] = sc
    # if user uses 1-based subcluster, convert to 0-based
    # heuristic: if any subcluster == 0, assume already 0-based
    if 0 not in label_to_sc.values() and max_sc > 0:
        label_to_sc = {k: v - 1 for k, v in label_to_sc.items()}
        max_sc = max_sc - 1
    return max_sc + 1, label_to_sc


def write_one_col_csv(path: Path, values: np.ndarray) -> None:
    pd.DataFrame(values).to_csv(path, index=False, header=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data", help="Path to data/ folder")
    ap.add_argument(
        "--hierarchy_json",
        default="phase8/label_hierarchy_v2.json",
        help="Path to label hierarchy JSON (label -> family/subcluster)",
    )
    ap.add_argument("--make_split_files", action="store_true", help="Create/refresh train_/val_ underscore files")
    ap.add_argument("--val_frac", type=float, default=0.2, help="Validation fraction if make_split_files enabled")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    hierarchy_path = Path(args.hierarchy_json)
    sequences, labels = load_no_underscore(data_dir)
    hierarchy = load_hierarchy(hierarchy_path)

    fam_to_id, fam_names = build_family_vocab(hierarchy)
    _, label_to_sc = build_subcluster_vocab(hierarchy)

    # Build family/subcluster labels aligned to trainsequences/trainlabels
    family_labels = np.array([fam_to_id[str(hierarchy[int(y)]["family"])] for y in labels], dtype=np.int32)
    subcluster_labels = np.array([int(label_to_sc[int(y)]) for y in labels], dtype=np.int32)

    # Optionally generate underscore train/val split files from the originals
    if args.make_split_files:
        train_seq, val_seq, train_lab, val_lab, idx_train, idx_val = train_test_split(
            sequences,
            labels,
            np.arange(len(labels)),
            test_size=args.val_frac,
            stratify=labels,
            random_state=args.seed,
        )

        write_one_col_csv(data_dir / "train_sequences.csv", train_seq)
        write_one_col_csv(data_dir / "train_labels.csv", train_lab)
        write_one_col_csv(data_dir / "val_sequences.csv", val_seq)
        write_one_col_csv(data_dir / "val_labels.csv", val_lab)
        np.savez(data_dir / "split_indices.npz", train=idx_train, val=idx_val)

        # Also create aligned hierarchical labels for train/val
        write_one_col_csv(data_dir / "train_family_labels.csv", family_labels[idx_train])
        write_one_col_csv(data_dir / "train_subcluster_labels.csv", subcluster_labels[idx_train])
        write_one_col_csv(data_dir / "val_family_labels.csv", family_labels[idx_val])
        write_one_col_csv(data_dir / "val_subcluster_labels.csv", subcluster_labels[idx_val])
    else:
        # If user already has train_/val_ files, write hierarchy labels matching those.
        # We assume underscore train/val were derived from original indices stored in split_indices.npz.
        split_path = data_dir / "split_indices.npz"
        if not split_path.exists():
            raise FileNotFoundError(
                "split_indices.npz not found. Either pass --make_split_files, "
                "or create split_indices.npz first."
            )
        splits = np.load(split_path)
        idx_train = splits["train"]
        idx_val = splits["val"]
        write_one_col_csv(data_dir / "train_family_labels.csv", family_labels[idx_train])
        write_one_col_csv(data_dir / "train_subcluster_labels.csv", subcluster_labels[idx_train])
        write_one_col_csv(data_dir / "val_family_labels.csv", family_labels[idx_val])
        write_one_col_csv(data_dir / "val_subcluster_labels.csv", subcluster_labels[idx_val])

    meta = {
        "hierarchy_json": str(hierarchy_path),
        "family_names": fam_names,
        "family_to_id": fam_to_id,
        "subcluster_note": "subcluster_labels are 0-based within the hierarchy JSON; if JSON is 1-based it is auto-shifted",
        "outputs": {
            "train_family_labels": str(data_dir / "train_family_labels.csv"),
            "train_subcluster_labels": str(data_dir / "train_subcluster_labels.csv"),
            "val_family_labels": str(data_dir / "val_family_labels.csv"),
            "val_subcluster_labels": str(data_dir / "val_subcluster_labels.csv"),
        },
    }
    with open(data_dir / "hierarchy_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Done.")
    print("Wrote hierarchy labels to data/:")
    for k, v in meta["outputs"].items():
        print(f"  - {k}: {v}")
    print("Family names:", fam_names)


if __name__ == "__main__":
    main()






