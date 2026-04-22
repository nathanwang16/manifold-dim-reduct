# Phase 5 — Activation Patching & Circuit Localization

Finds *where* in the phase-3 model class-discriminative information lives.

## What it measures

For every (class A, class B) confusable pair we:

1. Run a **clean forward** on a correctly-classified example of A and cache
   all hook activations.
2. Run a **corrupt forward** on a correctly-classified example of B and
   cache its hook activations.
3. At one hook at a time, **replace** A's activation with B's and re-run
   the forward. Measure how much class A's logit *drops* for the A example
   (and symmetrically).

A high *recovery* score at a given hook means the class distinction is
carried by that layer's representation.

## Hook points

```
stem, res_block1, conv_expand1, res_block2, conv_expand2,
res_block3, attn_out, pooled, bottleneck
```

## Files

- `hooks.py` — `ModelWithHooks` context manager wrapper around
  `ChromatinCNNAttentionV2`. Two context managers:
  - `record(names)`  → yields a dict of recorded activations.
  - `patch({name: tensor})`  → replaces the named module's output with a
    pre-recorded tensor during a forward pass.
- `patch_experiments.py` — main patching logic; writes
  `patching_results.csv` (per-pair, per-hook) and `circuit_summary.json`
  (per-hook aggregated recovery).
- `layer_dla.py` — layer-wise **direct logit attribution**: projects each
  intermediate layer's activation onto the class direction pulled back
  through the remaining linear layers. Writes `layer_dla.json`.
- `run_phase5.py` — runs both steps if outputs are missing.

## Usage

```bash
# single-GPU, ~5–10 min depending on n_pairs:
python phase5_patching/run_phase5.py \
    --checkpoint results/phase3/checkpoints/best.pt
```

## Config (`config.json` → `phase5_patching`)

```json
{
  "n_pairs_per_confusion": 200,
  "layers_to_patch": ["stem", "res_block1", "conv_expand1", "res_block2",
                       "conv_expand2", "res_block3", "attn_out",
                       "pooled", "bottleneck"],
  "balanced_eval_per_class": 2000,
  "output_dir": "results/phase5_patching"
}
```

## Interpreting results

- `circuit_summary.json` shows per-hook *mean recovery*. Larger ⇒ hook is
  more causal for distinguishing confusable classes in general.
- `patching_results.csv` is the long-form table; useful for plotting
  per-class-pair heatmaps (hook × class pair).
- `layer_dla.json` shows where along the depth class-direction projection
  magnitude peaks. Typically the bottleneck dominates but earlier layers
  can also show discriminative structure once projected back through the
  (approximately linear) bottleneck transformation.

Feeds into phase 6 (steering along the bottleneck direction) and phase 8
(motif discovery around the most causal layer).
