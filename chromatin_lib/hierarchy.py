"""Ground-truth 18-state -> family / subcluster mapping from the Roadmap DATASET.md.

The new dataset already contains authoritative `family` and `subcluster` columns
in every meta CSV. We also bake the same mapping here as constants so that code
that only has label ints (and no meta row) can still map to family / subcluster.

State -> family (6 classes): promoter, transcribed, enhancer, polycomb,
                             heterochromatin, quiescent.
State -> subcluster (7 classes, 1-indexed).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# 1-indexed state ordering matching the competition/label files.
STATE_ORDER: List[int] = list(range(1, 19))

_STATE_NAME_LIST: List[str] = [
    "1_TssA",
    "2_TssFlnk",
    "3_TssFlnkU",
    "4_TssFlnkD",
    "5_Tx",
    "6_TxWk",
    "7_EnhG1",
    "8_EnhG2",
    "9_EnhA1",
    "10_EnhA2",
    "11_EnhWk",
    "12_ZNF/Rpts",
    "13_Het",
    "14_TssBiv",
    "15_EnhBiv",
    "16_ReprPC",
    "17_ReprPCWk",
    "18_Quies",
]


# 0-indexed: matches how labels flow through the model (y in {0..17}).
STATE_NAMES: List[str] = list(_STATE_NAME_LIST)

FAMILY_NAMES: List[str] = [
    "promoter",
    "transcribed",
    "enhancer",
    "polycomb",
    "heterochromatin",
    "quiescent",
]

FAMILY_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(FAMILY_NAMES)}

# 1-indexed state -> family string (matches DATASET.md §6).
_STATE_TO_FAMILY: Dict[int, str] = {
    1: "promoter",
    2: "promoter",
    3: "promoter",
    4: "promoter",
    5: "transcribed",
    6: "transcribed",
    7: "enhancer",
    8: "enhancer",
    9: "enhancer",
    10: "enhancer",
    11: "enhancer",
    12: "heterochromatin",
    13: "heterochromatin",
    14: "promoter",
    15: "enhancer",
    16: "polycomb",
    17: "polycomb",
    18: "quiescent",
}

_STATE_TO_SUBCLUSTER: Dict[int, int] = {
    1: 1, 2: 1, 3: 1, 4: 1,
    5: 3, 6: 3,
    7: 4, 8: 4,
    9: 5, 10: 5, 11: 5,
    12: 7, 13: 7,
    14: 2,
    15: 5,
    16: 6, 17: 6,
    18: 7,
}


# Numpy lookup tables indexed by 0-indexed label.
LABEL_TO_FAMILY: np.ndarray = np.array(
    [FAMILY_TO_ID[_STATE_TO_FAMILY[s]] for s in STATE_ORDER],
    dtype=np.int64,
)
LABEL_TO_SUBCLUSTER: np.ndarray = np.array(
    [_STATE_TO_SUBCLUSTER[s] - 1 for s in STATE_ORDER],  # 0-indexed
    dtype=np.int64,
)

N_FAMILIES: int = len(FAMILY_NAMES)
N_SUBCLUSTERS: int = int(LABEL_TO_SUBCLUSTER.max()) + 1
N_CLASSES: int = 18


def family_class_indices() -> List[List[int]]:
    """Return list of length N_FAMILIES, each a sorted list of 0-indexed class ids."""
    out: List[List[int]] = [[] for _ in range(N_FAMILIES)]
    for cls_idx in range(N_CLASSES):
        out[int(LABEL_TO_FAMILY[cls_idx])].append(cls_idx)
    for lst in out:
        lst.sort()
    return out


def family_subcluster_indices() -> List[List[int]]:
    """For each family, the sorted list of 0-indexed subclusters present in it."""
    fam_to_subs: List[List[int]] = [[] for _ in range(N_FAMILIES)]
    for cls_idx in range(N_CLASSES):
        fam = int(LABEL_TO_FAMILY[cls_idx])
        sub = int(LABEL_TO_SUBCLUSTER[cls_idx])
        if sub not in fam_to_subs[fam]:
            fam_to_subs[fam].append(sub)
    for lst in fam_to_subs:
        lst.sort()
    return fam_to_subs


def load_hierarchy_from_meta(meta_path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    """Read meta CSV (small slice) to return a canonical label/family/subcluster frame.

    Used for sanity checks against the hard-coded constants.
    """
    df = pd.read_csv(meta_path, usecols=["state_int", "state_name", "family", "subcluster"],
                     nrows=max_rows)
    df["family_id"] = df["family"].map(FAMILY_TO_ID).astype("int64")
    df["subcluster_id"] = df["subcluster"].astype("int64") - 1
    return df
