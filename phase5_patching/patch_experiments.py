"""Activation patching across hook layers for the 18-class chromatin CNN.

Given a trained phase-3 checkpoint we:

1. Identify the top-K confusable class pairs from the model's predictions on
   a balanced val subset (confusion matrix off-diagonal).
2. For each pair (A, B) we pick `n_pairs` pairs of (x_A, x_B) such that the
   model classifies each correctly with high confidence.
3. Run a CLEAN forward on x_A, recording activations at every hook.
4. Run a CORRUPTED forward on x_B, recording activations.
5. For each hook layer, patch the corrupted activation into the clean
   forward pass and measure the logit shift:

       delta_logit_B = logit_B_patched - logit_B_clean
       delta_logit_A = logit_A_clean - logit_A_patched
       recovery     = (delta_logit_B + delta_logit_A) / 2 / full_gap

   where `full_gap` is the difference in logits between an unpatched clean
   forward and an unpatched corrupted forward. Recovery near 1.0 means the
   hooked layer "carries the entire class distinction".

Outputs (to `{output_dir}`):

    patching_results.csv    one row per (hook, pair_A, pair_B, sample_pair)
    circuit_summary.json    aggregated per-hook recovery per class pair
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chromatin_lib import (  # noqa: E402
    FAMILY_NAMES,
    LABEL_TO_FAMILY,
    LABEL_TO_SUBCLUSTER,
    STATE_NAMES,
    StreamingChromatinDataset,
    collate_chromatin,
    merged_split_paths,
)
from phase3_model.model import build_model  # noqa: E402
from phase5_patching.hooks import HOOK_NAMES, ModelWithHooks  # noqa: E402


def load_val_samples(
    cfg: Dict, n_per_class: int, device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """Return (x, feat, y, indices) cached in RAM for one pass."""
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
    loader = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=4,
                        collate_fn=collate_chromatin)

    xs, feats, ys = [], [], []
    per_class_counts = np.zeros(18, dtype=np.int64)
    target = n_per_class
    for batch in loader:
        y = batch["y"].numpy()
        keep = np.zeros(len(y), dtype=bool)
        for c in range(18):
            mask = y == c
            if per_class_counts[c] >= target:
                continue
            slots = target - per_class_counts[c]
            rows = np.flatnonzero(mask)[:slots]
            keep[rows] = True
            per_class_counts[c] += rows.size
        if keep.any():
            xs.append(batch["x"][keep])
            if "feat" in batch:
                feats.append(batch["feat"][keep])
            ys.append(batch["y"][keep])
        if (per_class_counts >= target).all():
            break

    x = torch.cat(xs, dim=0).to(device)
    feat = torch.cat(feats, dim=0).to(device) if feats else None
    y = torch.cat(ys, dim=0).to(device)
    return x, feat, y, idx


def find_confused_pairs(
    model: torch.nn.Module,
    x: torch.Tensor,
    feat: torch.Tensor,
    y: torch.Tensor,
    top_k: int,
    device: str,
) -> List[Tuple[int, int, float]]:
    """Run the model and return the top-K off-diagonal confusion entries.

    Returns a sorted list of (class_a, class_b, normalized_confusion_mass).
    """
    model.eval()
    bs = 1024
    n_classes = 18
    conf = torch.zeros(n_classes, n_classes, device=device)
    with torch.no_grad():
        for i in range(0, x.shape[0], bs):
            xb = x[i : i + bs]
            fb = feat[i : i + bs] if feat is not None else None
            yb = y[i : i + bs]
            logits = model(xb, engineered=fb)["logits"]
            preds = logits.argmax(dim=-1)
            for a, b in zip(yb.tolist(), preds.tolist()):
                conf[a, b] += 1
    conf = conf.cpu().numpy()
    row_norm = conf.sum(axis=1, keepdims=True).clip(min=1)
    rel = conf / row_norm  # per-class P(pred=b | true=a)
    pairs: List[Tuple[int, int, float]] = []
    for a in range(n_classes):
        for b in range(n_classes):
            if a == b:
                continue
            pairs.append((a, b, float(rel[a, b])))
    pairs.sort(key=lambda t: -t[2])
    return pairs[:top_k]


def run_patching(
    checkpoint: Path,
    config_path: Path,
    output_dir: Path,
    n_pairs_per_confusion: int,
    n_confusion_pairs: int,
    device: str,
    hooks_to_patch: List[str],
) -> None:
    with open(config_path) as f:
        cfg = json.load(f)

    p5 = cfg.get("phase5_patching", {})
    hooks_to_patch = hooks_to_patch or p5.get("layers_to_patch", HOOK_NAMES)
    hooks_to_patch = [h for h in hooks_to_patch if h in HOOK_NAMES]
    if not hooks_to_patch:
        hooks_to_patch = HOOK_NAMES

    model = build_model(cfg).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    wrapped = ModelWithHooks(model)

    per_class = int(p5.get("balanced_eval_per_class", 2000))
    x, feat, y, val_idx = load_val_samples(cfg, per_class, device)
    print(f"Loaded {x.shape[0]:,} val samples.")

    # Find confused class pairs
    confusion = find_confused_pairs(model, x, feat, y, top_k=n_confusion_pairs, device=device)
    print("Top confusions:")
    for a, b, p in confusion[:10]:
        print(f"  {STATE_NAMES[a]:>14s} -> {STATE_NAMES[b]:<14s}  {p:.3%}")

    # Pre-compute per-sample confidence and group by class
    with torch.no_grad():
        bs = 1024
        logits_all = torch.zeros(x.shape[0], 18, device=device)
        for i in range(0, x.shape[0], bs):
            xb = x[i : i + bs]
            fb = feat[i : i + bs] if feat is not None else None
            logits_all[i : i + bs] = model(xb, engineered=fb)["logits"]
    preds_all = logits_all.argmax(dim=-1)
    correct = preds_all == y
    by_class: Dict[int, List[int]] = {c: [] for c in range(18)}
    for i, (c, ok) in enumerate(zip(y.tolist(), correct.tolist())):
        if ok:
            by_class[c].append(i)

    rows: List[Dict] = []
    for pair_rank, (a, b, conf_p) in enumerate(confusion):
        a_idx = by_class[a]
        b_idx = by_class[b]
        if len(a_idx) < n_pairs_per_confusion or len(b_idx) < n_pairs_per_confusion:
            print(f"  [skip {STATE_NAMES[a]}→{STATE_NAMES[b]}] not enough correct samples")
            continue
        rng = np.random.default_rng(20260408 + pair_rank)
        a_sel = rng.choice(a_idx, size=n_pairs_per_confusion, replace=False)
        b_sel = rng.choice(b_idx, size=n_pairs_per_confusion, replace=False)

        # Clean forward on a_sel: record all hook activations
        x_a, f_a = x[a_sel], (feat[a_sel] if feat is not None else None)
        x_b, f_b = x[b_sel], (feat[b_sel] if feat is not None else None)

        with torch.no_grad():
            with wrapped.record(hooks_to_patch) as cache_a:
                logits_clean_a = model(x_a, engineered=f_a)["logits"]
            with wrapped.record(hooks_to_patch) as cache_b:
                logits_clean_b = model(x_b, engineered=f_b)["logits"]

        # Baseline gap: how much better does class A score for x_a vs x_b?
        # gap_ab[i] = logit_a(x_a_i) - logit_a(x_b_i)   (should be positive)
        gap_ab = (logits_clean_a[:, a] - logits_clean_b[:, a]).cpu().numpy()
        gap_ba = (logits_clean_b[:, b] - logits_clean_a[:, b]).cpu().numpy()

        for hook in hooks_to_patch:
            patch_a_from_b = {hook: cache_b[hook]}
            patch_b_from_a = {hook: cache_a[hook]}
            with torch.no_grad():
                with wrapped.patch(patch_a_from_b):
                    logits_a_patched = model(x_a, engineered=f_a)["logits"]
                with wrapped.patch(patch_b_from_a):
                    logits_b_patched = model(x_b, engineered=f_b)["logits"]

            # Recovery metric: how much did patching move the target class?
            # For A patched with B's activation: logit_a(x_a) should drop,
            # ideally approaching logit_a(x_b).
            drop_a = (logits_clean_a[:, a] - logits_a_patched[:, a]).cpu().numpy()
            drop_b = (logits_clean_b[:, b] - logits_b_patched[:, b]).cpu().numpy()
            rec_a = drop_a / np.where(gap_ab > 0, gap_ab, np.nan)
            rec_b = drop_b / np.where(gap_ba > 0, gap_ba, np.nan)

            rows.append({
                "pair_rank": pair_rank,
                "class_a": a,
                "class_b": b,
                "class_a_name": STATE_NAMES[a],
                "class_b_name": STATE_NAMES[b],
                "confusion": conf_p,
                "hook": hook,
                "n": int(n_pairs_per_confusion),
                "mean_drop_logit_a": float(drop_a.mean()),
                "mean_drop_logit_b": float(drop_b.mean()),
                "recovery_a": float(np.nanmean(rec_a)),
                "recovery_b": float(np.nanmean(rec_b)),
                "recovery_mean": float(np.nanmean([np.nanmean(rec_a),
                                                    np.nanmean(rec_b)])),
                "baseline_gap_ab": float(gap_ab.mean()),
                "baseline_gap_ba": float(gap_ba.mean()),
            })
        print(
            f"  [{STATE_NAMES[a]}→{STATE_NAMES[b]}] done. "
            f"Best hook: "
            f"{max([r for r in rows if r['class_a']==a and r['class_b']==b], key=lambda r: r['recovery_mean'])['hook']}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    # CSV
    with open(output_dir / "patching_results.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Aggregated per-hook
    per_hook: Dict[str, List[float]] = {h: [] for h in hooks_to_patch}
    for r in rows:
        per_hook[r["hook"]].append(r["recovery_mean"])
    summary = {
        "per_hook_mean_recovery": {h: float(np.mean(v)) for h, v in per_hook.items() if v},
        "per_hook_median_recovery": {h: float(np.median(v)) for h, v in per_hook.items() if v},
        "n_class_pairs": len({(r["class_a"], r["class_b"]) for r in rows}),
        "rows": len(rows),
    }
    (output_dir / "circuit_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {output_dir}/patching_results.csv and circuit_summary.json")
    print("Per-hook mean recovery (higher = more causal for class distinction):")
    for h, v in sorted(summary["per_hook_mean_recovery"].items(), key=lambda kv: -kv[1]):
        print(f"  {h:>14s}: {v:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default="config.json")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_pairs", type=int, default=None,
                        help="override phase5_patching.n_pairs_per_confusion")
    parser.add_argument("--n_confusion_pairs", type=int, default=10)
    parser.add_argument("--hooks", nargs="*", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    p5 = cfg.get("phase5_patching", {})
    output_dir = args.output_dir or Path(p5.get("output_dir", "results/phase5_patching"))
    n_pairs = args.n_pairs or int(p5.get("n_pairs_per_confusion", 200))

    run_patching(
        args.checkpoint, args.config, output_dir, n_pairs,
        args.n_confusion_pairs, args.device, args.hooks,
    )


if __name__ == "__main__":
    main()
