"""Extract ChromatinCNNAttentionV2 bottleneck activations for phase 4 / 5 / 6.

Loads a trained checkpoint and runs forward-only over a balanced subset
of the val split (single-GPU, batched). Writes:

    {output_dir}/bottleneck.npy      float32  (N, D)   bottleneck features
    {output_dir}/labels.npy          int64    (N,)
    {output_dir}/family.npy          int64    (N,)
    {output_dir}/subcluster.npy      int64    (N,)
    {output_dir}/indices.npy         int64    (N,)     row index into val CSV
    {output_dir}/logits.npy          float16  (N, 18)  (optional)

By default we use the `val_balanced_20000perclass` index from phase 1 and
subsample to `balanced_per_class` per class for the SAE training set so the
total tensor stays in RAM/GPU (default 12k/class × 18 = 216k).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chromatin_lib import (  # noqa: E402
    LABEL_TO_FAMILY,
    LABEL_TO_SUBCLUSTER,
    StreamingChromatinDataset,
    build_line_offset_index,
    collate_chromatin,
    merged_split_paths,
)
from phase3_model.model import build_model  # noqa: E402


def pick_balanced_indices(
    phase1_dir: Path, seed: int, per_class: int
) -> tuple[np.ndarray, np.ndarray]:
    """Load the phase-1 balanced index and sub-sample to `per_class` per class."""
    candidates = sorted(phase1_dir.glob(f"val_balanced_*perclass_seed{seed}.npy"))
    if not candidates:
        raise FileNotFoundError(
            f"No balanced val index in {phase1_dir}; run phase1 `build_subsamples.py` first."
        )
    # Pick the largest (so we can subsample down further if needed).
    candidate = max(candidates, key=lambda p: int(p.stem.split("_")[2].replace("perclass", "")))
    idx = np.load(candidate)
    # Need labels to stratify sub-sample.
    val_paths = merged_split_paths("val")
    # Read labels at these indices via streaming.
    labels_path = val_paths["labels"]
    offsets = np.asarray(build_line_offset_index(labels_path))
    labels = np.empty(idx.shape[0], dtype=np.int64)
    with open(labels_path, "rb") as f:
        content = f.read()  # ~1.2 GB (not 12 GB); label CSV is small ints
    for i, row in enumerate(idx):
        start = int(offsets[row])
        end = content.find(b"\n", start)
        labels[i] = int(content[start:end]) - 1

    rng = np.random.default_rng(seed)
    picked: list[np.ndarray] = []
    picked_labels: list[np.ndarray] = []
    for c in range(18):
        mask = np.flatnonzero(labels == c)
        if mask.size == 0:
            continue
        n_keep = min(per_class, mask.size)
        chosen = rng.choice(mask, size=n_keep, replace=False)
        picked.append(idx[chosen])
        picked_labels.append(np.full(n_keep, c, dtype=np.int64))
    idx_final = np.concatenate(picked)
    lab_final = np.concatenate(picked_labels)
    order = np.argsort(idx_final)
    return idx_final[order], lab_final[order]


def extract(
    checkpoint: Path,
    config_path: Path,
    output_dir: Path,
    per_class: int,
    batch_size: int,
    device: str,
    save_logits: bool,
) -> None:
    with open(config_path) as f:
        cfg = json.load(f)
    p4 = cfg.get("phase4_sae", {})
    phase1_dir = Path(cfg["phase1"]["output_dir"])
    seed = int(cfg["phase1"].get("seed", 20260408))

    indices, labels = pick_balanced_indices(phase1_dir, seed, per_class)
    print(f"Selected {indices.shape[0]:,} samples ({per_class}/class).")

    val_paths = merged_split_paths("val")
    dataset = StreamingChromatinDataset(
        sequences_path=val_paths["sequences"],
        labels_path=val_paths["labels"],
        sample_indices=indices,
        sequence_length=int(cfg["dataset"].get("sequence_length", 200)),
        compute_features=bool(cfg["phase3"].get("use_engineered_features", True)),
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=6, pin_memory=True, collate_fn=collate_chromatin,
    )

    model = build_model(cfg).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"Loaded checkpoint from {checkpoint}")

    activation_dim = int(p4.get("activation_dim", model.cfg.bottleneck_dim))
    n = indices.shape[0]
    bottleneck = np.zeros((n, activation_dim), dtype=np.float32)
    logits_cache: Optional[np.ndarray] = (
        np.zeros((n, cfg["dataset"].get("n_classes", 18)), dtype=np.float16) if save_logits else None
    )

    cursor = 0
    use_feat = bool(cfg["phase3"].get("use_engineered_features", True))
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            feat = batch.get("feat")
            if use_feat and feat is not None:
                feat = feat.to(device, non_blocking=True)
            else:
                feat = None
            out = model(x, engineered=feat)
            bn = out["bottleneck"].float().cpu().numpy()
            bottleneck[cursor : cursor + bn.shape[0]] = bn
            if logits_cache is not None:
                logits_cache[cursor : cursor + bn.shape[0]] = out["logits"].half().cpu().numpy()
            cursor += bn.shape[0]
            if (cursor // batch_size) % 20 == 0:
                print(f"  {cursor:,}/{n:,} extracted")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "bottleneck.npy", bottleneck)
    np.save(output_dir / "labels.npy", labels.astype(np.int64))
    np.save(output_dir / "family.npy", LABEL_TO_FAMILY[labels])
    np.save(output_dir / "subcluster.npy", LABEL_TO_SUBCLUSTER[labels])
    np.save(output_dir / "indices.npy", indices.astype(np.int64))
    if logits_cache is not None:
        np.save(output_dir / "logits.npy", logits_cache)

    stats = {
        "n_samples": int(n),
        "per_class": int(per_class),
        "activation_dim": int(activation_dim),
        "checkpoint": str(checkpoint),
        "feature_mean": float(bottleneck.mean()),
        "feature_std": float(bottleneck.std()),
        "fraction_zero": float((bottleneck == 0).mean()),
    }
    (output_dir / "extract_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"Wrote {output_dir}/bottleneck.npy  stats={stats}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", default="config.json", type=Path)
    parser.add_argument("--output_dir", default=None, type=Path)
    parser.add_argument("--per_class", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no_logits", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    p4 = cfg.get("phase4_sae", {})
    output_dir = args.output_dir or (Path(p4.get("output_dir", "results/phase4_sae")) / "activations")
    per_class = args.per_class if args.per_class is not None else int(p4.get("balanced_per_class", 12000))

    extract(args.checkpoint, args.config, output_dir, per_class, args.batch_size,
            args.device, save_logits=not args.no_logits)


if __name__ == "__main__":
    main()
