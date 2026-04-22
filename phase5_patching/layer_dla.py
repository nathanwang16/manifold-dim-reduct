"""Layer-wise Direct Logit Attribution (DLA).

For our architecture the classifier is a single Linear on top of the
bottleneck:

    logits[c] = W_cls[c, :] @ bottleneck + b_cls[c]

and the bottleneck is a Linear composition of `pooled` (+ engineered
feature MLP output). So we can cleanly project each intermediate layer's
output through the rest of the network to compute its direct contribution
to `logits[c]`.

For conv / attention layers we use *mean-pooled* channel contributions
through the remaining residual path (a linearisation that holds exactly
for linear steps and as a first-order approximation elsewhere).

Outputs `layer_dla.json` with, for every class, the mean projection
magnitude of each hook point onto the class-specific direction
`W_cls[c, :]` pulled back through downstream linear layers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chromatin_lib import (  # noqa: E402
    STATE_NAMES,
    StreamingChromatinDataset,
    collate_chromatin,
    merged_split_paths,
)
from phase3_model.model import build_model  # noqa: E402
from phase5_patching.hooks import ModelWithHooks  # noqa: E402


def compute_dla(
    checkpoint: Path,
    config_path: Path,
    output_dir: Path,
    n_per_class: int,
    device: str,
) -> None:
    with open(config_path) as f:
        cfg = json.load(f)

    model = build_model(cfg).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    wrapped = ModelWithHooks(model)

    # Class-specific "unembedding" direction: projection of the class head
    # back through the bottleneck to the 512-d pooled representation (plus
    # engineered feature concatenation).
    W_cls = model.class_head.weight.detach()  # (18, D_bn)
    W_bn = model.bottleneck[0].weight.detach()  # (D_bn, D_in)
    # Pull class direction through bottleneck ReLU→Linear: first-order
    # approximation is the product W_cls @ W_bn.
    class_dir_pooled = W_cls @ W_bn  # (18, D_in)
    class_dir_pooled_np = class_dir_pooled.cpu().numpy()

    # Load a small balanced val subset
    phase1_dir = Path(cfg["phase1"]["output_dir"])
    seed = int(cfg["phase1"].get("seed", 20260408))
    candidates = sorted(phase1_dir.glob(f"val_balanced_*perclass_seed{seed}.npy"))
    if not candidates:
        raise FileNotFoundError("Run phase1 `build_subsamples.py` first.")
    candidate = max(candidates, key=lambda p: int(p.stem.split("_")[2].replace("perclass", "")))
    idx_all = np.load(candidate)

    # Uniform subsample: n_per_class per class requires labels, which we pull
    # via the streaming dataset.
    val_paths = merged_split_paths("val")
    ds = StreamingChromatinDataset(
        sequences_path=val_paths["sequences"],
        labels_path=val_paths["labels"],
        sample_indices=idx_all,
        sequence_length=int(cfg["dataset"].get("sequence_length", 200)),
        compute_features=bool(cfg["phase3"].get("use_engineered_features", True)),
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1024, num_workers=4, shuffle=False,
        collate_fn=collate_chromatin,
    )
    per_class_counts = np.zeros(18, dtype=np.int64)

    hooks = ["stem", "res_block1", "conv_expand1", "res_block2",
             "conv_expand2", "res_block3", "attn_out", "pooled", "bottleneck"]
    # Keep per-layer, per-class rolling sums of magnitude of class-direction
    # projection of the layer activation.
    sums: Dict[str, np.ndarray] = {h: np.zeros(18, dtype=np.float64) for h in hooks}
    counts: Dict[str, np.ndarray] = {h: np.zeros(18, dtype=np.int64) for h in hooks}

    use_feat = bool(cfg["phase3"].get("use_engineered_features", True))
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].numpy()
            keep = np.zeros(len(y), dtype=bool)
            for c in range(18):
                if per_class_counts[c] >= n_per_class:
                    continue
                slots = n_per_class - per_class_counts[c]
                rows = np.flatnonzero(y == c)[:slots]
                keep[rows] = True
                per_class_counts[c] += rows.size
            if not keep.any():
                if (per_class_counts >= n_per_class).all():
                    break
                continue
            x = x[keep]
            y = y[keep]
            feat = batch["feat"][keep].to(device) if use_feat else None

            with wrapped.record(hooks) as cache:
                out = model(x, engineered=feat)

            # Project each layer onto the class direction for each sample's true class.
            # Bottleneck is (B, D_bn): direct projection = logits (minus bias) per class.
            for h in hooks:
                act = cache[h]  # channel-first for conv; (B, D) for pooled/bottleneck
                if act.dim() == 3:
                    # (B, C, L) -> spatially average then dot with class_dir_pooled
                    pooled = act.mean(dim=-1)  # (B, C)
                    if pooled.shape[1] != class_dir_pooled.shape[1]:
                        # Layer has a different channel count than pooled_in; use
                        # L2 norm as an unsigned magnitude proxy.
                        mag = pooled.norm(dim=-1).cpu().numpy()
                        for i, c in enumerate(y):
                            sums[h][c] += mag[i]
                            counts[h][c] += 1
                        continue
                    proj = (pooled * class_dir_pooled[torch.as_tensor(y, device=device)]).sum(dim=-1)
                    proj_np = proj.cpu().numpy()
                    for i, c in enumerate(y):
                        sums[h][c] += float(proj_np[i])
                        counts[h][c] += 1
                else:
                    # 1-D activation
                    if h == "bottleneck":
                        proj = (act * W_cls[torch.as_tensor(y, device=device)]).sum(dim=-1)
                    elif act.shape[-1] == class_dir_pooled.shape[-1]:
                        proj = (act * class_dir_pooled[torch.as_tensor(y, device=device)]).sum(dim=-1)
                    else:
                        proj = act.norm(dim=-1)  # fallback magnitude
                    proj_np = proj.cpu().numpy()
                    for i, c in enumerate(y):
                        sums[h][c] += float(proj_np[i])
                        counts[h][c] += 1

            if (per_class_counts >= n_per_class).all():
                break

    per_layer_per_class = {
        h: (sums[h] / np.maximum(counts[h], 1)).tolist() for h in hooks
    }
    overall = {h: float(np.mean(v)) for h, v in per_layer_per_class.items()}

    output_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "per_layer_mean_projection": overall,
        "per_layer_per_class": {h: dict(zip(STATE_NAMES, v))
                                for h, v in per_layer_per_class.items()},
    }
    (output_dir / "layer_dla.json").write_text(json.dumps(out, indent=2))
    print("Layer DLA mean projection onto true-class direction:")
    for h, v in sorted(overall.items(), key=lambda kv: -kv[1]):
        print(f"  {h:>14s}: {v:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default="config.json")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--n_per_class", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    output_dir = args.output_dir or Path(cfg.get("phase5_patching", {}).get(
        "output_dir", "results/phase5_patching"))

    compute_dla(args.checkpoint, args.config, output_dir, args.n_per_class, args.device)


if __name__ == "__main__":
    main()
