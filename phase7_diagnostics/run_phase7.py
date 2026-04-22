"""Phase 7 orchestrator: saliency + calibration + RC consistency.

Writes to `results/phase7_diagnostics/`:
    saliency_per_class.npy       (n_classes, n_per_class, 200, 4)
    saliency_per_class_sum.npy   (n_classes, 200)   per-base mean importance
    saliency_examples.json       one example sequence per class for reference
    calibration.json             temperature + ECE before/after
    consistency.json             RC consistency report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
from phase7_diagnostics.calibration import fit_temperature  # noqa: E402
from phase7_diagnostics.consistency import reverse_complement_consistency  # noqa: E402
from phase7_diagnostics.saliency import smoothgrad_saliency  # noqa: E402


def _load_val_loader(cfg: dict, per_class: int) -> DataLoader:
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
    return DataLoader(ds, batch_size=1024, num_workers=4, shuffle=False,
                      collate_fn=collate_chromatin)


def run(args):
    with open(args.config) as f:
        cfg = json.load(f)
    p7 = cfg.get("phase7", {})
    out_dir = Path(p7.get("output_dir", "results/phase7_diagnostics"))
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg).to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    use_feat = bool(cfg["phase3"].get("use_engineered_features", True))

    # ----- Calibration & RC consistency (requires a val pass) -----
    val_loader = _load_val_loader(cfg, per_class=2000)

    logits_cat, labels_cat = [], []
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(args.device, non_blocking=True)
            feat = batch.get("feat").to(args.device) if (use_feat and batch.get("feat") is not None) else None
            out = model(x, engineered=feat)
            logits_cat.append(out["logits"].cpu())
            labels_cat.append(batch["y"])
    logits_all = torch.cat(logits_cat, dim=0)
    labels_all = torch.cat(labels_cat, dim=0)
    T, ece_before, ece_after = fit_temperature(logits_all, labels_all, device="cpu")
    (out_dir / "calibration.json").write_text(json.dumps({
        "temperature": T,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "n_samples": int(labels_all.shape[0]),
    }, indent=2))
    print(f"Calibration: T={T:.3f}  ECE {ece_before:.4f} -> {ece_after:.4f}")

    val_loader = _load_val_loader(cfg, per_class=2000)  # fresh iterator
    rc_stats = reverse_complement_consistency(
        model, val_loader, device=args.device, use_feat=use_feat
    )
    (out_dir / "consistency.json").write_text(json.dumps(rc_stats, indent=2))
    print(f"RC consistency: overall {rc_stats['overall_consistency']:.4f}")

    # ----- Saliency per class -----
    n_per_class = int(p7.get("saliency_n_per_class", 64))
    n_noise = int(p7.get("smoothgrad_n_samples", 20))
    noise_std = float(p7.get("smoothgrad_noise_std", 0.1))

    # Collect n_per_class correctly-classified examples per class
    cls_pool = {c: [] for c in range(18)}
    with torch.no_grad():
        val_loader = _load_val_loader(cfg, per_class=n_per_class)
        for batch in val_loader:
            x = batch["x"]
            y = batch["y"].numpy()
            feat = batch.get("feat")
            x_dev = x.to(args.device)
            feat_dev = feat.to(args.device) if (use_feat and feat is not None) else None
            preds = model(x_dev, engineered=feat_dev)["logits"].argmax(dim=-1).cpu().numpy()
            ok = preds == y
            for i in range(x.shape[0]):
                c = int(y[i])
                if not ok[i] or len(cls_pool[c]) >= n_per_class:
                    continue
                cls_pool[c].append({
                    "x": x[i].clone(),
                    "feat": feat[i].clone() if feat is not None else None,
                    "y": c,
                })
            if all(len(v) >= n_per_class for v in cls_pool.values()):
                break

    saliency_all = np.zeros((18, n_per_class, 200, 4), dtype=np.float32)
    for c in range(18):
        if not cls_pool[c]:
            continue
        batch_x = torch.stack([s["x"] for s in cls_pool[c]]).to(args.device)
        batch_feat = (
            torch.stack([s["feat"] for s in cls_pool[c]]).to(args.device)
            if use_feat and cls_pool[c][0]["feat"] is not None else None
        )
        target = torch.full((batch_x.shape[0],), c, dtype=torch.long, device=args.device)
        sal = smoothgrad_saliency(
            model, batch_x, target, engineered=batch_feat,
            n_samples=n_noise, noise_std=noise_std,
        )
        saliency_all[c, : sal.shape[0]] = sal.detach().cpu().numpy()
        print(f"  saliency {STATE_NAMES[c]}: mean {sal.mean().item():.4g}")

    np.save(out_dir / "saliency_per_class.npy", saliency_all)
    summary = saliency_all.sum(axis=-1).mean(axis=1)  # (n_classes, 200)
    np.save(out_dir / "saliency_per_class_sum.npy", summary)
    (out_dir / "saliency_meta.json").write_text(json.dumps({
        "n_per_class": n_per_class,
        "smoothgrad_n_samples": n_noise,
        "noise_std": noise_std,
        "class_names": STATE_NAMES,
    }, indent=2))
    print(f"Saved saliency_per_class.npy (shape {saliency_all.shape}) to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default="config.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true",
                        help="Unused placeholder for pipeline symmetry.")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
