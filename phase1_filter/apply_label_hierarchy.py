"""
Apply an existing label hierarchy to existing split label files.

This is useful when you already have:
  data/train_labels.csv, data/val_labels.csv
and you want:
  data/train_family_labels.csv, data/train_subcluster_labels.csv
  data/val_family_labels.csv,   data/val_subcluster_labels.csv
without regenerating splits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd


def load_one_col_int_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path, header=None)
    col = df.iloc[:, 0].tolist()
    out: List[int] = []
    for x in col:
        s = str(x).strip()
        try:
            out.append(int(s))
        except Exception:
            out.append(int(float(s)))
    arr = np.asarray(out, dtype=np.int32)
    if arr.min() == 0 and arr.max() == 17:
        arr = arr + 1
    return arr


def load_hierarchy(hierarchy_json: Path) -> Dict[int, Dict[str, object]]:
    with open(hierarchy_json, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def build_family_vocab(hierarchy: Dict[int, Dict[str, object]]) -> Tuple[Dict[str, int], List[str]]:
    families = sorted({str(v["family"]) for v in hierarchy.values()})
    fam_to_id = {name: i for i, name in enumerate(families)}
    return fam_to_id, families


def build_subcluster_vocab(hierarchy: Dict[int, Dict[str, object]]) -> Tuple[int, Dict[int, int]]:
    max_sc = 0
    label_to_sc: Dict[int, int] = {}
    for label, meta in hierarchy.items():
        sc = int(meta.get("subcluster", 0))
        max_sc = max(max_sc, sc)
        label_to_sc[label] = sc
    if 0 not in label_to_sc.values() and max_sc > 0:
        label_to_sc = {k: v - 1 for k, v in label_to_sc.items()}
        max_sc = max_sc - 1
    return max_sc + 1, label_to_sc


def write_one_col_csv(path: Path, values: np.ndarray) -> None:
    pd.DataFrame(values).to_csv(path, index=False, header=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--hierarchy_json", default="phase8/label_hierarchy_v2.json")
    ap.add_argument("--train_labels", default="train_labels.csv")
    ap.add_argument("--val_labels", default="val_labels.csv")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    hierarchy = load_hierarchy(Path(args.hierarchy_json))
    fam_to_id, fam_names = build_family_vocab(hierarchy)
    _, label_to_sc = build_subcluster_vocab(hierarchy)

    train = load_one_col_int_csv(data_dir / args.train_labels)
    val = load_one_col_int_csv(data_dir / args.val_labels)

    train_family = np.array([fam_to_id[str(hierarchy[int(y)]["family"])] for y in train], dtype=np.int32)
    train_sc = np.array([int(label_to_sc[int(y)]) for y in train], dtype=np.int32)
    val_family = np.array([fam_to_id[str(hierarchy[int(y)]["family"])] for y in val], dtype=np.int32)
    val_sc = np.array([int(label_to_sc[int(y)]) for y in val], dtype=np.int32)

    write_one_col_csv(data_dir / "train_family_labels.csv", train_family)
    write_one_col_csv(data_dir / "train_subcluster_labels.csv", train_sc)
    write_one_col_csv(data_dir / "val_family_labels.csv", val_family)
    write_one_col_csv(data_dir / "val_subcluster_labels.csv", val_sc)

    meta = {
        "hierarchy_json": args.hierarchy_json,
        "family_names": fam_names,
        "family_to_id": fam_to_id,
    }
    with open(data_dir / "hierarchy_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Done. Wrote:")
    print("  - train_family_labels.csv")
    print("  - train_subcluster_labels.csv")
    print("  - val_family_labels.csv")
    print("  - val_subcluster_labels.csv")
    print("Family names:", fam_names)


if __name__ == "__main__":
    main()






