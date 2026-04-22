"""Shared utilities for the Chromatin Discovery Engine.

All phases (1..8) import data/encoding/hierarchy helpers from here so we have
a single source of truth for the 18-state Roadmap dataset produced by
`phase0_aggregate/roadmap_pipeline.py`.
"""

from .encoding import (
    one_hot_encode_sequence,
    one_hot_encode_batch,
    reverse_complement,
    reverse_complement_onehot,
    BASES,
    BASE_TO_IDX,
)
from .hierarchy import (
    STATE_ORDER,
    STATE_NAMES,
    FAMILY_NAMES,
    FAMILY_TO_ID,
    LABEL_TO_FAMILY,
    LABEL_TO_SUBCLUSTER,
    family_class_indices,
    family_subcluster_indices,
    load_hierarchy_from_meta,
)
from .paths import (
    DATA_ROOT,
    RESULTS_ROOT,
    merged_split_paths,
    per_epigenome_paths,
)
from .features import (
    compute_engineered_features,
    compute_engineered_features_batch,
)
from .data import (
    InMemoryChromatinDataset,
    StreamingChromatinDataset,
    CachedMmapDataset,
    build_line_offset_index,
    stratified_label_indices,
    collate_chromatin,
)

__all__ = [
    "one_hot_encode_sequence",
    "one_hot_encode_batch",
    "reverse_complement",
    "reverse_complement_onehot",
    "BASES",
    "BASE_TO_IDX",
    "STATE_ORDER",
    "STATE_NAMES",
    "FAMILY_NAMES",
    "FAMILY_TO_ID",
    "LABEL_TO_FAMILY",
    "LABEL_TO_SUBCLUSTER",
    "family_class_indices",
    "family_subcluster_indices",
    "load_hierarchy_from_meta",
    "DATA_ROOT",
    "RESULTS_ROOT",
    "merged_split_paths",
    "per_epigenome_paths",
    "compute_engineered_features",
    "compute_engineered_features_batch",
    "InMemoryChromatinDataset",
    "StreamingChromatinDataset",
    "CachedMmapDataset",
    "build_line_offset_index",
    "stratified_label_indices",
    "collate_chromatin",
]
