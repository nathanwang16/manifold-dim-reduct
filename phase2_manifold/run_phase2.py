"""Phase 2 driver: manifold learning on a balanced train subsample.

Pipeline:
  1. Materialize a stratified train subsample (reuses phase1 index .npy)
  2. Extract 6-mer / positional / dinucleotide features
  3. Run PCA / UMAP / PHATE dimensionality reduction
  4. Cluster analysis + silhouette / ARI metrics
  5. Static visualizations (per-label, per-family, per-subcluster colorings)

Artifacts land under `results/phase2/...` per `config.json`.

Typical usage:
    python phase2_manifold/run_phase2.py                       # full run
    python phase2_manifold/run_phase2.py --subsample 10000     # quick dev
    python phase2_manifold/run_phase2.py --skip_reduction      # replot only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chromatin_lib import (  # noqa: E402
    FAMILY_NAMES,
    LABEL_TO_FAMILY,
    LABEL_TO_SUBCLUSTER,
    merged_split_paths,
)
from logger import get_logger, configure_logging, LogTimer  # noqa: E402

logger = get_logger(__name__)


def materialize_subsample(
    indices_path: Path,
    sequences_path: Path,
    labels_path: Path,
    out_dir: Path,
    subsample: int | None = None,
) -> dict[str, Path]:
    """Given a row-index .npy, emit aligned sequences + labels + family + subcluster CSVs."""
    indices = np.load(indices_path)
    if subsample is not None and indices.size > subsample:
        rng = np.random.default_rng(0)
        indices = rng.choice(indices, size=subsample, replace=False)
    indices = np.sort(indices)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Materializing {indices.size:,} sequences from {sequences_path.name}")
    keep = set(indices.tolist())
    seq_out = out_dir / "viz_sequences.csv"
    lab_out = out_dir / "viz_labels.csv"
    fam_out = out_dir / "viz_family_labels.csv"
    sub_out = out_dir / "viz_subcluster_labels.csv"

    lab_values: list[int] = []
    with open(sequences_path) as fs, open(labels_path) as fl, open(seq_out, "w") as so, open(lab_out, "w") as lo:
        for i, (seq, lab) in enumerate(zip(fs, fl)):
            if i in keep:
                so.write(seq)
                lo.write(lab)
                lab_values.append(int(lab.strip()) - 1)

    lab_arr = np.asarray(lab_values, dtype=np.int64)
    fam_arr = LABEL_TO_FAMILY[lab_arr]
    sub_arr = LABEL_TO_SUBCLUSTER[lab_arr]
    pd.DataFrame(fam_arr).to_csv(fam_out, header=False, index=False)
    pd.DataFrame(sub_arr).to_csv(sub_out, header=False, index=False)

    logger.info(f"Wrote {seq_out}, {lab_out}, {fam_out}, {sub_out}")
    return {
        "sequences": seq_out,
        "labels": lab_out,
        "family": fam_out,
        "subcluster": sub_out,
    }


def run_step(script: Path, args: list[str], name: str) -> None:
    cmd = [sys.executable, str(script)] + args
    logger.info(f"Running {name}: {' '.join(cmd)}")
    with LogTimer(logger, name):
        subprocess.run(cmd, check=True)


def plot_per_coloring(
    embeddings_dir: Path,
    labels_path: Path,
    coloring_name: str,
    output_dir: Path,
) -> None:
    """Re-plot the already-computed 2D embeddings colored by a different label array."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    labels = np.loadtxt(labels_path, dtype=np.int64)
    n_classes = int(labels.max()) + 1
    cmap = plt.get_cmap("tab20", max(n_classes, 20))

    for emb_path in sorted(embeddings_dir.glob("*.npy")):
        if "labels" in emb_path.name or "variance" in emb_path.name or "indices" in emb_path.name:
            continue
        emb = np.load(emb_path)
        if emb.ndim != 2 or emb.shape[1] < 2:
            continue
        fig, ax = plt.subplots(figsize=(7, 6), dpi=130)
        scatter = ax.scatter(emb[:, 0], emb[:, 1], c=labels[: emb.shape[0]], cmap=cmap, s=2, alpha=0.6)
        ax.set_title(f"{emb_path.stem} • colored by {coloring_name}")
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        cbar = plt.colorbar(scatter, ax=ax, ticks=range(n_classes))
        cbar.set_label(coloring_name)
        out = output_dir / f"{emb_path.stem}_{coloring_name}.png"
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        logger.info(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--skip_materialize", action="store_true")
    parser.add_argument("--skip_extraction", action="store_true")
    parser.add_argument("--skip_reduction", action="store_true")
    parser.add_argument("--skip_analysis", action="store_true")
    parser.add_argument("--skip_visualization", action="store_true")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Cap the viz sample to this many total rows (debug).")
    parser.add_argument("--log-dir", default="results/phase2/logs")
    args = parser.parse_args()

    configure_logging(log_dir=args.log_dir)

    with open(args.config) as f:
        cfg = json.load(f)

    phase1_out = Path(cfg["phase1"]["output_dir"])
    per_class = int(cfg["phase1"]["viz_subsample_per_class"])
    seed = int(cfg["phase1"]["seed"])
    idx_file = phase1_out / f"train_viz_{per_class}perclass_seed{seed}.npy"
    if not idx_file.exists():
        raise FileNotFoundError(
            f"Missing phase1 subsample index {idx_file}. Run `python phase1_filter/run_phase1.py`."
        )

    paths = merged_split_paths("train")
    phase2_cfg = cfg["phase2"]
    features_dir = Path(phase2_cfg["features_dir"])
    embeddings_dir = Path(phase2_cfg["embeddings_dir"])
    analysis_dir = Path(phase2_cfg["analysis_dir"])
    figures_dir = Path(phase2_cfg["figures_dir"])
    for d in (features_dir, embeddings_dir, analysis_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    viz_dir = features_dir.parent / "viz_materialized"
    if not args.skip_materialize:
        viz_paths = materialize_subsample(
            idx_file, paths["sequences"], paths["labels"], viz_dir, args.subsample
        )
    else:
        viz_paths = {
            "sequences": viz_dir / "viz_sequences.csv",
            "labels": viz_dir / "viz_labels.csv",
            "family": viz_dir / "viz_family_labels.csv",
            "subcluster": viz_dir / "viz_subcluster_labels.csv",
        }

    scripts_dir = Path(__file__).parent

    # Step 1: Feature extraction
    if not args.skip_extraction:
        run_step(
            scripts_dir / "feature_extraction.py",
            [
                "--input", str(viz_paths["sequences"]),
                "--labels", str(viz_paths["labels"]),
                "--config", args.config,
                "--output", str(features_dir),
                "--log-dir", args.log_dir,
                "--n-jobs", str(phase2_cfg.get("n_jobs", -1)),
                "--batch-size", str(phase2_cfg.get("feature_extraction_batch_size", 2000)),
            ],
            "Feature extraction",
        )

    # Step 2: Dimensionality reduction
    feat_path = features_dir / f"kmer_{phase2_cfg['kmer_k']}_features.npy"
    label_path = features_dir / "labels.npy"
    if not args.skip_reduction:
        run_step(
            scripts_dir / "dimensionality_reduction.py",
            [
                "--features", str(feat_path),
                "--labels", str(label_path),
                "--config", args.config,
                "--output", str(embeddings_dir),
                "--log-dir", args.log_dir,
            ],
            "Dimensionality reduction",
        )

    # Step 3: Cluster analysis
    if not args.skip_analysis:
        run_step(
            scripts_dir / "cluster_analysis.py",
            [
                "--embeddings", str(embeddings_dir),
                "--labels", str(embeddings_dir / "labels.npy"),
                "--config", args.config,
                "--output", str(analysis_dir),
                "--log-dir", args.log_dir,
            ],
            "Cluster analysis",
        )

    # Step 4: Static visualizations (label-colored)
    if not args.skip_visualization:
        run_step(
            scripts_dir / "static_visualizations.py",
            [
                "--embeddings", str(embeddings_dir),
                "--analysis", str(analysis_dir),
                "--labels", str(embeddings_dir / "labels.npy"),
                "--output", str(figures_dir),
                "--log-dir", args.log_dir,
            ],
            "Static visualizations (label)",
        )

        # Extra: family- and subcluster-colored scatter plots.
        plot_per_coloring(
            embeddings_dir,
            viz_paths["family"],
            "family",
            figures_dir / "family",
        )
        plot_per_coloring(
            embeddings_dir,
            viz_paths["subcluster"],
            "subcluster",
            figures_dir / "subcluster",
        )

    logger.info("Phase 2 complete.")


if __name__ == "__main__":
    sys.exit(main() or 0)
