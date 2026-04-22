"""Phase 6 orchestrator: steering-vector sweeps on the balanced val set.

Steps (all single-GPU):

1. Extract bottleneck activations for the balanced val subset (`phase4_sae`
   already does this — we reuse its output if present; otherwise we call
   `extract_activations.py`).
2. Compute per-class steering vectors (`compute_steering_vectors`).
3. For each class c and a grid of alphas, apply steering and measure:
   - overall val accuracy
   - per-class accuracy change
   - selective boost for class c
4. For the top-K confusable class pairs (from the confusion matrix), try
   adding `alpha * (dir_B - dir_A)` to push predictions A -> B; report
   improvement on actually-confused examples.

Outputs (`results/phase6_steering/`):
    steering_vectors.npz                 directions + unit directions
    steering_sweep.csv                   long-form per-(class, alpha, metric)
    contrastive_pairs.csv                per-pair improvement at best alpha
    summary.json                         headline metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chromatin_lib import (  # noqa: E402
    STATE_NAMES,
    StreamingChromatinDataset,
    collate_chromatin,
    merged_split_paths,
)
from phase3_model.model import build_model  # noqa: E402
from phase6_steering.steering import (  # noqa: E402
    SteeringVectors,
    compute_steering_vectors,
    steered_forward,
)


def _ensure_activations(
    checkpoint: Path, config_path: Path, act_dir: Path, device: str
) -> None:
    bottleneck_path = act_dir / "bottleneck.npy"
    if bottleneck_path.exists():
        return
    print("Extracting activations (reusing phase4 pipeline)...")
    subprocess.run([
        sys.executable, "phase4_sae/extract_activations.py",
        "--checkpoint", str(checkpoint),
        "--config", str(config_path),
        "--output_dir", str(act_dir),
        "--device", device,
    ], check=True, cwd=REPO_ROOT)


def load_val_batch_in_ram(cfg: Dict, per_class: int, device: str):
    phase1_dir = Path(cfg["phase1"]["output_dir"])
    seed = int(cfg["phase1"].get("seed", 20260408))
    candidates = sorted(phase1_dir.glob(f"val_balanced_*perclass_seed{seed}.npy"))
    if not candidates:
        raise FileNotFoundError("Run phase1 `build_subsamples.py` first.")
    candidate = max(candidates, key=lambda p: int(p.stem.split("_")[2].replace("perclass", "")))
    idx = np.load(candidate)
    val_paths = merged_split_paths("val")
    ds = StreamingChromatinDataset(
        sequences_path=val_paths["sequences"],
        labels_path=val_paths["labels"],
        sample_indices=idx,
        sequence_length=int(cfg["dataset"].get("sequence_length", 200)),
        compute_features=bool(cfg["phase3"].get("use_engineered_features", True)),
    )
    loader = DataLoader(ds, batch_size=1024, num_workers=4, shuffle=False,
                        collate_fn=collate_chromatin)

    per_class_counts = np.zeros(18, dtype=np.int64)
    xs, feats, ys = [], [], []
    for batch in loader:
        y = batch["y"].numpy()
        keep = np.zeros(len(y), dtype=bool)
        for c in range(18):
            if per_class_counts[c] >= per_class:
                continue
            slots = per_class - per_class_counts[c]
            rows = np.flatnonzero(y == c)[:slots]
            keep[rows] = True
            per_class_counts[c] += rows.size
        if keep.any():
            xs.append(batch["x"][keep])
            if "feat" in batch:
                feats.append(batch["feat"][keep])
            ys.append(batch["y"][keep])
        if (per_class_counts >= per_class).all():
            break

    x = torch.cat(xs, dim=0).to(device)
    feat = torch.cat(feats, dim=0).to(device) if feats else None
    y = torch.cat(ys, dim=0).to(device)
    return x, feat, y


def sweep_alpha(
    model: torch.nn.Module,
    x: torch.Tensor,
    feat: torch.Tensor,
    y: torch.Tensor,
    directions: torch.Tensor,
    alphas: List[float],
    batch_size: int = 1024,
) -> List[Dict]:
    results: List[Dict] = []
    n_classes = directions.shape[0]
    device = x.device
    with torch.no_grad():
        base_preds = torch.zeros_like(y)
        for i in range(0, x.shape[0], batch_size):
            xb = x[i : i + batch_size]
            fb = feat[i : i + batch_size] if feat is not None else None
            base_preds[i : i + batch_size] = model(xb, engineered=fb)["logits"].argmax(dim=-1)
        base_acc = (base_preds == y).float().mean().item()

        for c in range(n_classes):
            dir_c = directions[c]
            for alpha in alphas:
                preds = torch.zeros_like(y)
                for i in range(0, x.shape[0], batch_size):
                    xb = x[i : i + batch_size]
                    fb = feat[i : i + batch_size] if feat is not None else None
                    out = steered_forward(model, xb, fb, dir_c, alpha)
                    preds[i : i + batch_size] = out["logits"].argmax(dim=-1)
                acc = (preds == y).float().mean().item()
                mask_c = y == c
                acc_c = (preds[mask_c] == c).float().mean().item() if mask_c.any() else 0.0
                fraction_pred_c = (preds == c).float().mean().item()
                results.append({
                    "target_class": c,
                    "target_name": STATE_NAMES[c],
                    "alpha": alpha,
                    "overall_acc": acc,
                    "target_class_acc": acc_c,
                    "fraction_pred_target": fraction_pred_c,
                    "baseline_overall_acc": base_acc,
                })
    return results


def confused_pair_improvement(
    model: torch.nn.Module,
    x: torch.Tensor,
    feat: torch.Tensor,
    y: torch.Tensor,
    directions: torch.Tensor,
    alphas: List[float],
    top_k_pairs: int,
    batch_size: int = 1024,
) -> List[Dict]:
    """For top-K confused (A→B) pairs, try adding alpha*(dir_A - dir_B) to
    push mispredictions back to A.
    """
    device = x.device
    with torch.no_grad():
        base_preds = torch.zeros_like(y)
        for i in range(0, x.shape[0], batch_size):
            xb = x[i : i + batch_size]
            fb = feat[i : i + batch_size] if feat is not None else None
            base_preds[i : i + batch_size] = model(xb, engineered=fb)["logits"].argmax(dim=-1)
    base_preds_np = base_preds.cpu().numpy()
    y_np = y.cpu().numpy()

    # Relative confusion: P(pred=B | true=A)
    conf = np.zeros((18, 18), dtype=np.int64)
    for a, b in zip(y_np, base_preds_np):
        conf[a, b] += 1
    row = conf.sum(axis=1, keepdims=True).clip(min=1)
    rel = conf / row

    pairs = []
    for a in range(18):
        for b in range(18):
            if a != b:
                pairs.append((a, b, float(rel[a, b])))
    pairs.sort(key=lambda t: -t[2])
    pairs = pairs[:top_k_pairs]

    results: List[Dict] = []
    for a, b, conf_p in pairs:
        # Target: misclassified-as-B true-A examples. Push toward A.
        misclass_mask = (y == a) & (base_preds == b)
        n_mis = int(misclass_mask.sum().item())
        if n_mis == 0:
            continue
        x_mis = x[misclass_mask]
        feat_mis = feat[misclass_mask] if feat is not None else None
        y_mis = y[misclass_mask]
        dir_ab = directions[a] - directions[b]
        best = None
        with torch.no_grad():
            for alpha in alphas:
                preds = torch.zeros_like(y_mis)
                for i in range(0, x_mis.shape[0], batch_size):
                    xb = x_mis[i : i + batch_size]
                    fb = feat_mis[i : i + batch_size] if feat_mis is not None else None
                    out = steered_forward(model, xb, fb, dir_ab, alpha)
                    preds[i : i + batch_size] = out["logits"].argmax(dim=-1)
                rec = (preds == a).float().mean().item()
                collateral_pred_b = (preds == b).float().mean().item()
                row = {
                    "class_a": int(a),
                    "class_b": int(b),
                    "class_a_name": STATE_NAMES[a],
                    "class_b_name": STATE_NAMES[b],
                    "n_misclassified": n_mis,
                    "alpha": alpha,
                    "recovery_rate": rec,
                    "still_predicting_b": collateral_pred_b,
                    "baseline_confusion": conf_p,
                }
                results.append(row)
                if best is None or rec > best["recovery_rate"]:
                    best = row
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default="config.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--activations_dir", type=Path, default=None,
                        help="Re-use the phase4 activations cache if present.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    p6 = cfg.get("phase6", {})
    base_out = Path(p6.get("output_dir", "results/phase6_steering"))
    base_out.mkdir(parents=True, exist_ok=True)
    act_dir = args.activations_dir or Path(cfg.get("phase4_sae", {}).get(
        "output_dir", "results/phase4_sae")) / "activations"

    _ensure_activations(args.checkpoint, args.config, act_dir, args.device)

    bottleneck = np.load(act_dir / "bottleneck.npy")
    labels = np.load(act_dir / "labels.npy")
    vec = compute_steering_vectors(bottleneck, labels, n_classes=18)

    np.savez(base_out / "steering_vectors.npz",
             centroids=vec.centroids,
             global_centroid=vec.global_centroid,
             directions=vec.directions,
             unit_directions=vec.unit_directions,
             counts=vec.counts)

    # Load val batch into RAM for sweeps
    per_class_val = int(p6.get("balanced_val_per_class", 5000))
    print(f"Loading {per_class_val}/class val samples ...")
    x, feat, y = load_val_batch_in_ram(cfg, per_class_val, args.device)
    print(f"Loaded {x.shape[0]:,} val samples.")

    model = build_model(cfg).to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    alphas = list(p6.get("alpha_grid", [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]))
    directions_t = torch.from_numpy(vec.directions).float().to(args.device)

    sweep = sweep_alpha(model, x, feat, y, directions_t, alphas)
    with open(base_out / "steering_sweep.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
        writer.writeheader()
        writer.writerows(sweep)

    contrastive = confused_pair_improvement(
        model, x, feat, y, directions_t, alphas,
        top_k_pairs=int(p6.get("contrastive_top_k_pairs", 10)),
    )
    if contrastive:
        with open(base_out / "contrastive_pairs.csv", "w") as f:
            writer = csv.DictWriter(f, fieldnames=list(contrastive[0].keys()))
            writer.writeheader()
            writer.writerows(contrastive)

    # Headline
    base_acc = sweep[0]["baseline_overall_acc"]
    best_per_class = {}
    for row in sweep:
        c = row["target_class"]
        if c not in best_per_class or row["target_class_acc"] > best_per_class[c]["target_class_acc"]:
            best_per_class[c] = row

    best_pair_recovery = {}
    for row in contrastive:
        key = (row["class_a"], row["class_b"])
        if key not in best_pair_recovery or row["recovery_rate"] > best_pair_recovery[key]["recovery_rate"]:
            best_pair_recovery[key] = row

    headline = {
        "baseline_overall_acc": base_acc,
        "per_class_best_target_acc": {
            STATE_NAMES[c]: {"alpha": r["alpha"], "target_acc": r["target_class_acc"],
                              "overall_acc": r["overall_acc"]}
            for c, r in best_per_class.items()
        },
        "contrastive_recovery": [
            {"a": r["class_a_name"], "b": r["class_b_name"],
             "alpha": r["alpha"], "recovery_rate": r["recovery_rate"],
             "still_b": r["still_predicting_b"], "n": r["n_misclassified"]}
            for r in best_pair_recovery.values()
        ],
    }
    (base_out / "summary.json").write_text(json.dumps(headline, indent=2))
    print(f"Wrote steering results to {base_out}")
    print(f"Baseline overall acc: {base_acc:.4f}")


if __name__ == "__main__":
    main()
