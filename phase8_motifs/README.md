# Phase 8 — Motif Discovery & Hypothesis Generation

With ground-truth `state_name` now available from `meta.csv`, the old
"label identification" puzzle is moot. Phase 8 instead turns the trained
phase-3 model into a *microscope* on sequence patterns and produces
human-readable, testable hypotheses about DNA motifs that drive each of
the 18 chromatin states.

## Probes

- **Stem-filter PWMs** (`stem_motifs.py`). The 256 `stem_conv` filters
  act as motif scanners; for each filter we pool the top-K 19-bp windows
  from a balanced val subset, build a PWM, compute its information
  content (bits), and record which classes it is enriched in. Output:
  `stem_motifs.npz` with `pwms (F, 19, 4)`, `info_content (F,)`, and
  `class_counts (F, 18)`.

- **In-silico mutagenesis** (`ism.py`). For each of N correctly-
  classified examples per class, evaluate `logit(c)` under every
  single-base substitution at every position. Gives a per-position,
  per-base `Δlogit` tensor that is a richer signal than unsigned
  SmoothGrad. Output: `ism_per_class.npy (18, N, 200, 4)`.

- **Hypothesis consolidation** (`hypotheses.py`). Ranks the motifs by
  information content × class specificity, emits `hypotheses.json`
  (machine-readable) and `hypotheses.md` (human-readable).

## Files

- `stem_motifs.py` — `extract_stem_pwms(model, loader, ...)`
- `ism.py`         — `in_silico_mutagenesis(model, x, target, ...)`
- `hypotheses.py`  — `build_hypotheses`, `write_markdown`
- `run_phase8.py`  — orchestrator

## Usage

```bash
python phase8_motifs/run_phase8.py \
    --checkpoint results/phase3/checkpoints/best.pt
```

## Config (`config.json` → `phase8`)

```json
{
  "motif_n_filters": 256,
  "motif_top_activations": 500,
  "motif_min_info_content": 0.3,
  "motif_top_n_report": 60,
  "motif_top_class_min_frac": 0.15,
  "motif_batch_limit": 40,
  "ism_n_per_class": 16,
  "ism_mutation_batch": 800,
  "output_dir": "results/phase8_motifs"
}
```

## Outputs

- `stem_motifs.npz` — PWMs + info content + class counts.
- `ism_per_class.npy` — per-class ISM delta tensor.
- `hypotheses.json` / `hypotheses.md` — ranked hypothesis list.
