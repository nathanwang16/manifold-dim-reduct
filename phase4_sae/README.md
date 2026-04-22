# Phase 4 — Sparse Autoencoder on the Bottleneck

Trains an SAE on the 384-d `bottleneck` activations of the phase-3 model,
then compiles a per-feature "dictionary" that phases 5/6/8 consume.

## Pipeline

```
[trained phase3 .pt] → extract_activations → activations/bottleneck.npy
                                             activations/{labels,family,subcluster,indices}.npy
                          ↓
                   train_sae → sae.pt + history.json + feature_stats.npz
                          ↓
              feature_analysis → feature_analysis/features.json
                                 feature_analysis/features_summary.csv
```

## Files

- `sae.py` — `BaseSAE`, `TopKSAE` (Anthropic-style), `L1SAE`. Includes the
  "pre-subtract decoder bias" parameterisation and unit-norm dictionary
  renormalisation after each step.
- `extract_activations.py` — runs a trained phase-3 checkpoint over a
  balanced subsample of the val split (`balanced_per_class`/class),
  writes `bottleneck.npy` (N × D) and aligned label tensors.
- `train_sae.py` — trains the SAE (top-K or L1) on the cached activations;
  reports reconstruction R², L0, dead-feature fraction.
- `feature_analysis.py` — for each feature, computes:
  top-activating example indices, class/family distribution over top-k
  activations, specificity score, firing rate. Writes both JSON (for
  phase 5/8) and CSV (for quick human inspection).
- `run_phase4.py` — orchestrator that stitches all three.

## Config (`config.json` → `phase4_sae`)

```json
{
  "activation_layer": "bottleneck",   // hook point on the phase-3 model
  "activation_dim": 384,              // == phase3.bottleneck_dim
  "feature_dim": 4096,                // 10.7x expansion
  "sae_type": "topk",                 // "topk" | "l1"
  "topk_k": 32,                       // L0 target for topk
  "l1_weight": 2e-3,                  // used for sae_type="l1"
  "lr": 1e-3,
  "num_epochs": 20,
  "batch_size": 4096,
  "balanced_per_class": 12000,        // 18 * 12k = 216k activation samples
  "output_dir": "results/phase4_sae"
}
```

## Usage

```bash
# 1) extract activations + train SAE + build feature dictionary:
python phase4_sae/run_phase4.py \
    --checkpoint results/phase3/checkpoints/best.pt

# 2) or run each step manually:
python phase4_sae/extract_activations.py \
    --checkpoint results/phase3/checkpoints/best.pt \
    --output_dir results/phase4_sae/activations
python phase4_sae/train_sae.py \
    --activations_dir results/phase4_sae/activations \
    --output_dir results/phase4_sae
python phase4_sae/feature_analysis.py \
    --activations_dir results/phase4_sae/activations
```

## Outputs

- `results/phase4_sae/sae.pt` — state dict + activation mean + config.
- `results/phase4_sae/final_metrics.json` — reconstruction R², dead
  fraction, mean L0.
- `results/phase4_sae/feature_analysis/features.json` — the "feature
  dictionary" consumed by downstream MI phases. Each entry is a dict with
  `feature_idx`, `firing_rate`, `specificity_score`, `top_classes`,
  `top_families`, `example_indices`.
