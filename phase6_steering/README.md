# Phase 6 — Representation Engineering (Steering Vectors)

Computes per-class steering vectors on the 384-d bottleneck of the trained
phase-3 model and measures how well the representation can be "pushed"
toward a target class or "unconfused" for known class-pair confusions.

## Why

If the bottleneck is a clean family/state representation, then centroid-
based steering directions should monotonically increase target-class
accuracy as `alpha` grows, with graceful trade-offs for collateral
classes. Failures (e.g., flat or chaotic sweeps) indicate the representation
has not actually disentangled that axis — information that feeds phase 8
(motif discovery).

## Files

- `steering.py` — `compute_steering_vectors` (centroid − global centroid),
  `steered_forward` (forward hook that adds `alpha * direction` to the
  bottleneck output).
- `run_phase6.py` — orchestrator. Re-uses `phase4_sae/extract_activations.py`
  to cache bottleneck activations, computes steering vectors, sweeps a
  grid of alphas, and evaluates contrastive steering for the top-K
  confused class pairs.

## Usage

```bash
python phase6_steering/run_phase6.py \
    --checkpoint results/phase3/checkpoints/best.pt
```

## Config (`config.json` → `phase6`)

```json
{
  "balanced_val_per_class": 5000,
  "confidence_threshold": 0.6,
  "default_alpha": 0.5,
  "alpha_grid": [0.0, 0.25, 0.5, 0.75, 1.0, 1.5],
  "contrastive_top_k_pairs": 10,
  "contrastive_confusion_threshold": 0.05,
  "output_dir": "results/phase6_steering"
}
```

## Outputs

- `steering_vectors.npz` — directions (18 × 384), unit directions, and
  class centroids.
- `steering_sweep.csv` — per-(class, alpha) overall & target-class
  accuracy.
- `contrastive_pairs.csv` — for each top-K confused pair (A→B), the
  recovery rate (fraction of mispredicted true-A samples that come back
  to A) at each alpha when applying `alpha * (dir_A − dir_B)`.
- `summary.json` — headline per-class best alpha and contrastive recovery.
