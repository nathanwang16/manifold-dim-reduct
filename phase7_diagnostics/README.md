# Phase 7 — Diagnostics (Saliency, Calibration, RC Consistency)

Three cheap, single-GPU probes of the trained phase-3 model. Feeds the
writeup and is a precondition for phase 8 (motif discovery).

## Probes

| Probe | Metric | Why |
|-------|--------|-----|
| SmoothGrad saliency | per-base \|∂ logit / ∂ x\| averaged over Gaussian perturbations | identifies which bases drive each class prediction — sanity check for motif discovery in phase 8 |
| Calibration | Expected Calibration Error before / after temperature scaling | should the logits be down-weighted before being used for hypothesis generation? |
| RC consistency | fraction of val samples with argmax(model(x)) == argmax(model(RC(x))) | required invariance for DNA models |

## Files

- `saliency.py`    — `smoothgrad_saliency(model, x, target, ...)`
- `calibration.py` — `expected_calibration_error`, `fit_temperature`
- `consistency.py` — `reverse_complement_consistency`
- `run_phase7.py`  — orchestrator that runs all three

## Usage

```bash
python phase7_diagnostics/run_phase7.py \
    --checkpoint results/phase3/checkpoints/best.pt
```

## Config (`config.json` → `phase7`)

```json
{
  "saliency_n_per_class": 64,
  "smoothgrad_n_samples": 20,
  "smoothgrad_noise_std": 0.1,
  "ism_positions": "all",
  "output_dir": "results/phase7_diagnostics"
}
```

## Outputs

- `saliency_per_class.npy`     — (18, n_per_class, 200, 4) float32
- `saliency_per_class_sum.npy` — (18, 200) position-wise mean importance
- `saliency_meta.json`         — class names + hyperparameters
- `calibration.json`           — temperature T + ECE before/after
- `consistency.json`           — overall & per-class RC consistency, mean KL
