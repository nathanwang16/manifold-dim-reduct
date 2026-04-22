# Chromatin State Prediction — Mechanistic Interpretability Pipeline

End-to-end pipeline for training an interpretable CNN + attention model
on the Roadmap 18-state full dataset (1M balanced train, ~64M val,
~63M test, 200 bp sequences) and then turning the trained model into a
*microscope* on chromatin sequence patterns via mechanistic
interpretability techniques (Sparse Autoencoders, activation patching,
activation steering, saliency, and in-silico mutagenesis).

Hardware target: 5× RTX 3090 (24 GB) · CUDA 12.4+ · PyTorch 2.6+.
Environment: `conda activate biohack` (see `environment.yml`).

## Pipeline Overview

```
phase0_aggregate → phase1_filter → phase2_manifold → phase3_model
     │                                                    │
     ▼                                                    ▼
  raw ChromHMM                            ┌──────────────┴──────────────┐
  → hg38 corpora                          ▼                             ▼
                                 phase4_sae  phase5_patching  phase6_steering
                                     │          │                  │
                                     └──────────┴──────────────────┘
                                                ▼
                                    phase7_diagnostics
                                                ▼
                                    phase8_motifs (hypotheses)
```

## Phase Directory

| Dir | Purpose | Key entrypoint |
|-----|---------|----------------|
| `phase0_aggregate/` | Roadmap download + liftOver + merged train/val/test corpora | `scripts/run_phase0_aggregate.py` |
| `phase1_filter/`    | Hierarchy labels + class-balanced subsample indices                | `run_phase1.py` |
| `phase2_manifold/`  | k-mer features, PCA, UMAP/PHATE of the label landscape             | `run_phase2.py` |
| `phase3_model/`     | `ChromatinCNNAttentionV2` DDP training on 5 GPUs                   | `launch.sh` (wraps `train_ddp.py`) |
| `phase4_sae/`       | TopK / L1 Sparse Autoencoder on the 384-d bottleneck + feature dict | `run_phase4.py` |
| `phase5_patching/`  | Activation patching + layer-wise direct logit attribution          | `run_phase5.py` |
| `phase6_steering/`  | Per-class steering vectors + alpha sweep + contrastive steering    | `run_phase6.py` |
| `phase7_diagnostics/` | SmoothGrad saliency, temperature calibration, RC consistency      | `run_phase7.py` |
| `phase8_motifs/`    | Stem-filter PWMs, in-silico mutagenesis, hypothesis records         | `run_phase8.py` |
| `chromatin_lib/`    | Shared utilities: data loaders, one-hot, hierarchy, paths           | — |

## Quick Start

```bash
conda activate biohack

# Phase 0: one-time download + preprocess (requires network + disk)
python phase0_aggregate/scripts/run_phase0_aggregate.py \
    --config phase0_aggregate/config/roadmap_18state_full.json

# Phase 1: build balanced subsample indices + hierarchy labels
python phase1_filter/run_phase1.py --splits train val

# Phase 2: manifold visualisation (fast, single-GPU)
python phase2_manifold/run_phase2.py

# Phase 3: pre-bake mmap caches, then DDP train on 5x 3090
python phase3_model/precompute_cache.py --splits train
bash   phase3_model/launch.sh

# Phases 4-8: single-GPU, consume phase-3 checkpoint
python run_pipeline.py --checkpoint results/phase3/checkpoints/best.pt --phases 4,5,6,7,8
# or individually:
python phase4_sae/run_phase4.py        --checkpoint results/phase3/checkpoints/best.pt
python phase5_patching/run_phase5.py   --checkpoint results/phase3/checkpoints/best.pt
python phase6_steering/run_phase6.py   --checkpoint results/phase3/checkpoints/best.pt
python phase7_diagnostics/run_phase7.py --checkpoint results/phase3/checkpoints/best.pt
python phase8_motifs/run_phase8.py     --checkpoint results/phase3/checkpoints/best.pt
```

## Global Configuration

A single `config.json` at the repo root contains every phase's knobs
under top-level keys: `dataset`, `phase1`, `phase2`, `phase3`,
`phase4_sae`, `phase5_patching`, `phase6`, `phase7`, `phase8`. Paths are
resolved relative to the repo root.

## Outputs

All run artifacts live under `results/`:

```
results/phase1_filter/      balanced indices + hierarchy label caches
results/phase2/             k-mer features + UMAP/PHATE embeddings + figures
results/phase3/             DDP checkpoints + training history
results/phase4_sae/         sae.pt + activation caches + features.json
results/phase5_patching/    patching_results.csv + circuit_summary.json + layer_dla.json
results/phase6_steering/    steering_vectors.npz + alpha sweep + contrastive pairs
results/phase7_diagnostics/ saliency_per_class.npy + calibration.json + consistency.json
results/phase8_motifs/      stem_motifs.npz + ism_per_class.npy + hypotheses.{json,md}
```

## What's "discovered" vs. "known"

The previous iteration of this project treated the 18 ChromHMM labels
as unknown states and tried to re-identify them post-hoc (phase 8's
original "label identification" objective). The new roadmap corpus
already exposes ground-truth `state_name`, `family`, and `subcluster`
via the meta CSVs, so phase 8 has been repurposed: instead of
identifying labels we use the trained model to *generate biologically
testable hypotheses about motifs that drive each state* via
SAE features (phase 4), circuit localization (phase 5), steering
directions (phase 6), saliency (phase 7), and ISM + stem-filter PWMs
(phase 8).

## Hardware & runtime budget

- Phase 3 DDP training: ~18–22 h on 5× RTX 3090 for 40 epochs.
- Phase 4 SAE: ~10 min on one GPU.
- Phase 5 patching: ~15 min on one GPU.
- Phase 6 steering sweep: ~10 min.
- Phase 7 diagnostics: ~15 min.
- Phase 8 motif discovery: ~30 min (dominated by ISM).

## Revision history

- **2026-04-21**: Pipeline v2 overhaul. Added `chromatin_lib/`,
  rewrote phases 3/4/5/6, added phases 7 (diagnostics) and 8 (motifs).
  Reworked all data loaders for the 12 GB val/test CSVs via
  `StreamingChromatinDataset` + mmap-backed `CachedMmapDataset`.
  DDP hardened for 5× 3090.
- 2026-04-08: Added `phase0_aggregate` (Roadmap 18-state full).
