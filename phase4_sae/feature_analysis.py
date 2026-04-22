"""Compute per-feature diagnostics for a trained SAE.

Produces `features.json` — one record per SAE feature with:

    - feature_idx
    - firing_rate             fraction of samples where the feature fires
    - mean_activation         mean of activation over activated samples
    - top_classes             list of (class_int, pct) for top classes in top-k
                              activating samples
    - top_families            same for family ids
    - entropy_classes         Shannon entropy (bits) over 18-class distribution
                              restricted to top-k activations (lower = more
                              class-specific).
    - specificity_score       1 - entropy/log2(18). Higher = more specific.
    - example_indices         indices (within the extract set) of the top-k
                              activating samples — phase 5/8 can go back to
                              the val CSV via `indices.npy`.

Also writes a tabular CSV for quick visual inspection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chromatin_lib import FAMILY_NAMES, STATE_NAMES  # noqa: E402
from phase4_sae.sae import build_sae  # noqa: E402


def analyze(
    sae_path: Path,
    activations_dir: Path,
    output_dir: Path,
    top_k: int,
    device: str,
) -> None:
    ckpt = torch.load(sae_path, map_location="cpu", weights_only=False)
    sae_cfg = ckpt["config"]
    d = int(ckpt["activation_dim"])
    sae = build_sae({"phase4_sae": {**sae_cfg, "activation_dim": d}})
    sae.load_state_dict(ckpt["state_dict"])
    sae.to(device).eval()

    bottleneck = np.load(activations_dir / "bottleneck.npy")
    labels = np.load(activations_dir / "labels.npy")
    family = np.load(activations_dir / "family.npy")
    mean = ckpt["activation_mean"]
    bottleneck_centered = bottleneck - mean
    n, _ = bottleneck.shape

    feature_dim = sae.feature_dim
    # Encode in chunks and keep activations on disk-friendly float16.
    h_all = np.zeros((n, feature_dim), dtype=np.float16)
    chunk = 4096
    with torch.no_grad():
        for i in range(0, n, chunk):
            xb = torch.from_numpy(bottleneck_centered[i : i + chunk]).float().to(device)
            out = sae(xb)
            h_all[i : i + chunk] = out["h"].cpu().numpy().astype(np.float16)
    firing = (h_all > 0).mean(axis=0)
    mean_act = h_all.astype(np.float32).sum(axis=0) / np.maximum(
        (h_all > 0).sum(axis=0), 1
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "h_all.npy", h_all)
    np.save(output_dir / "firing_rate.npy", firing)
    np.save(output_dir / "mean_activation_when_on.npy", mean_act)

    records: List[Dict] = []
    n_classes = int(labels.max()) + 1 if labels.size else 18
    log_n_classes = float(np.log2(n_classes)) if n_classes > 1 else 1.0
    for f in range(feature_dim):
        col = h_all[:, f].astype(np.float32)
        active_idx = np.flatnonzero(col > 0)
        if active_idx.size == 0:
            records.append({
                "feature_idx": int(f),
                "firing_rate": 0.0,
                "mean_activation": 0.0,
                "top_classes": [],
                "top_families": [],
                "entropy_classes": 0.0,
                "specificity_score": 0.0,
                "example_indices": [],
            })
            continue
        top = active_idx[np.argsort(col[active_idx])[-top_k:][::-1]]
        cls_counts = np.bincount(labels[top], minlength=n_classes)
        fam_counts = np.bincount(family[top], minlength=int(family.max()) + 1)
        p = cls_counts / cls_counts.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
        specificity = 1.0 - (entropy / log_n_classes)
        top_classes = sorted(
            [(int(c), float(cls_counts[c] / cls_counts.sum()))
             for c in np.flatnonzero(cls_counts)],
            key=lambda kv: -kv[1],
        )[:4]
        top_families = sorted(
            [(int(c), float(fam_counts[c] / fam_counts.sum()))
             for c in np.flatnonzero(fam_counts)],
            key=lambda kv: -kv[1],
        )[:4]
        records.append({
            "feature_idx": int(f),
            "firing_rate": float(firing[f]),
            "mean_activation": float(mean_act[f]),
            "top_classes": [{"idx": c, "name": STATE_NAMES[c], "pct": p}
                            for c, p in top_classes],
            "top_families": [{"idx": c, "name": FAMILY_NAMES[c], "pct": p}
                             for c, p in top_families],
            "entropy_classes": entropy,
            "specificity_score": specificity,
            "example_indices": [int(i) for i in top],
        })

    (output_dir / "features.json").write_text(json.dumps(records, indent=2))

    # Simple rollup CSV
    with open(output_dir / "features_summary.csv", "w") as f:
        f.write("feature_idx,firing_rate,mean_activation,specificity,top_class,top_class_pct,top_family,top_family_pct\n")
        for r in records:
            tc = r["top_classes"][0] if r["top_classes"] else {"idx": -1, "pct": 0, "name": "n/a"}
            tf = r["top_families"][0] if r["top_families"] else {"idx": -1, "pct": 0, "name": "n/a"}
            f.write(
                f"{r['feature_idx']},{r['firing_rate']:.5f},{r['mean_activation']:.5f},"
                f"{r['specificity_score']:.4f},{tc['name']},{tc['pct']:.3f},"
                f"{tf['name']},{tf['pct']:.3f}\n"
            )

    dead = sum(1 for r in records if r["firing_rate"] == 0)
    print(
        f"Wrote feature dictionary ({len(records)} features). "
        f"Dead: {dead} ({dead/len(records):.1%}). "
        f"Median specificity: {np.median([r['specificity_score'] for r in records]):.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae_path", type=Path, default=None)
    parser.add_argument("--activations_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default="config.json")
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    base_out = Path(cfg.get("phase4_sae", {}).get("output_dir", "results/phase4_sae"))
    sae_path = args.sae_path or (base_out / "sae.pt")
    output_dir = args.output_dir or (base_out / "feature_analysis")

    analyze(sae_path, args.activations_dir, output_dir, args.top_k, args.device)


if __name__ == "__main__":
    main()
