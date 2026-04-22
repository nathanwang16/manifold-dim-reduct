"""Phase 8 orchestrator: stem motifs + ISM + hypotheses.

Pipeline:
    1. Load the trained phase-3 checkpoint + a balanced val loader.
    2. Extract stem-filter PWMs with `stem_motifs.extract_stem_pwms`.
    3. Compute a summarised ISM tensor on a small held-out subset for
       per-class "hotspot" reports.
    4. Build hypothesis records from PWMs + ISM + (optional) saliency.
    5. Write json + markdown summaries.
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
from phase8_motifs.hypotheses import build_hypotheses, write_markdown  # noqa: E402
from phase8_motifs.ism import in_silico_mutagenesis  # noqa: E402
from phase8_motifs.stem_motifs import extract_stem_pwms  # noqa: E402


def _loader(cfg: dict, per_class: int, batch_size: int = 512) -> DataLoader:
    phase1_dir = Path(cfg["phase1"]["output_dir"])
    seed = int(cfg["phase1"].get("seed", 20260408))
    candidates = sorted(phase1_dir.glob(f"val_balanced_*perclass_seed{seed}.npy"))
    if not candidates:
        raise FileNotFoundError("Run phase1 `build_subsamples.py` first.")
    cand = max(candidates, key=lambda p: int(p.stem.split("_")[2].replace("perclass", "")))
    idx = np.load(cand)
    # Subsample so we actually have ~per_class samples per class.
    idx = np.random.RandomState(seed).permutation(idx)[: per_class * 18]
    val_paths = merged_split_paths("val")
    ds = StreamingChromatinDataset(
        sequences_path=val_paths["sequences"],
        labels_path=val_paths["labels"],
        sample_indices=idx,
        sequence_length=int(cfg["dataset"].get("sequence_length", 200)),
        compute_features=bool(cfg["phase3"].get("use_engineered_features", True)),
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=4,
                      shuffle=False, collate_fn=collate_chromatin)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default="config.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    p8 = cfg.get("phase8", {})
    out_dir = Path(p8.get("output_dir", "results/phase8_motifs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg).to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    # ----- Stem motifs -----
    motifs_path = out_dir / "stem_motifs.npz"
    if args.force or not motifs_path.exists():
        loader = _loader(cfg, per_class=400, batch_size=512)
        print("Extracting stem-filter PWMs ...")
        res = extract_stem_pwms(
            model, loader, device=args.device,
            top_k_per_filter=int(p8.get("motif_top_activations", 500)),
            batch_limit=int(p8.get("motif_batch_limit", 40)),
        )
        np.savez(motifs_path, **res)
        print(f"  wrote {motifs_path}  pwms={res['pwms'].shape}")
    else:
        res = dict(np.load(motifs_path))
        print(f"loaded {motifs_path}")

    # ----- ISM summary -----
    ism_path = out_dir / "ism_per_class.npy"
    if args.force or not ism_path.exists():
        n_per = int(p8.get("ism_n_per_class", 16))
        print(f"Collecting {n_per}/class for ISM ...")
        # Gather correctly-classified examples per class
        pool = {c: [] for c in range(18)}
        use_feat = bool(cfg["phase3"].get("use_engineered_features", True))
        loader = _loader(cfg, per_class=n_per * 4, batch_size=512)
        with torch.no_grad():
            for batch in loader:
                x = batch["x"]
                y = batch["y"].numpy()
                feat = batch.get("feat")
                x_dev = x.to(args.device)
                feat_dev = feat.to(args.device) if (use_feat and feat is not None) else None
                preds = model(x_dev, engineered=feat_dev)["logits"].argmax(dim=-1).cpu().numpy()
                ok = preds == y
                for i in range(x.shape[0]):
                    c = int(y[i])
                    if not ok[i] or len(pool[c]) >= n_per:
                        continue
                    pool[c].append({
                        "x": x[i].clone(),
                        "feat": feat[i].clone() if feat is not None else None,
                    })
                if all(len(v) >= n_per for v in pool.values()):
                    break

        ism_per_class = np.zeros((18, n_per, 200, 4), dtype=np.float32)
        for c in range(18):
            if not pool[c]:
                continue
            xb = torch.stack([s["x"] for s in pool[c]]).to(args.device)
            fb = (torch.stack([s["feat"] for s in pool[c]]).to(args.device)
                  if use_feat and pool[c][0]["feat"] is not None else None)
            tgt = torch.full((xb.shape[0],), c, dtype=torch.long, device=args.device)
            d = in_silico_mutagenesis(model, xb, tgt, engineered=fb,
                                      mutation_batch=int(p8.get("ism_mutation_batch", 800)))
            ism_per_class[c, : d.shape[0]] = d.detach().cpu().numpy()
            print(f"  ISM {STATE_NAMES[c]}: mean|Δ|={np.abs(ism_per_class[c]).mean():.4g}")
        np.save(ism_path, ism_per_class)
    else:
        print(f"loaded {ism_path}")

    # ----- Hypotheses -----
    print("Building hypothesis records ...")
    hyps = build_hypotheses(
        pwms=res["pwms"],
        info_content=res["info_content"],
        class_counts=res["class_counts"],
        min_info_bits=float(p8.get("motif_min_info_content", 0.3)),
        top_class_min_frac=float(p8.get("motif_top_class_min_frac", 0.15)),
        top_n_filters=int(p8.get("motif_top_n_report", 60)),
    )
    (out_dir / "hypotheses.json").write_text(json.dumps(hyps, indent=2))
    write_markdown(hyps, out_dir / "hypotheses.md")
    print(f"  wrote {len(hyps)} hypotheses to {out_dir}/hypotheses.{{json,md}}")


if __name__ == "__main__":
    main()
