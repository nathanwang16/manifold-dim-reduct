# Phase 3 — ChromatinCNNAttentionV2 (DDP)

Interpretable CNN + attention classifier trained on the full roadmap_18state
1M-sample balanced train set across 5× RTX 3090.

## Architecture (`model.py`)

```
Input (B, 200, 4) one-hot
 → Stem Conv1d(4→256, k=19)  + BN + ReLU                [200]
 → ResBlock(256, k=7)        MaxPool(2)                 [100]
 → Conv(256→384, k=5)        + BN + ReLU
 → ResBlock(384, k=5)        MaxPool(2)                 [ 50]
 → Conv(384→512, k=3)        + BN + ReLU
 → ResBlock(512, k=3)                                   [ 50]
 → 2 × TransformerBlock(512, 8 heads)                   [ 50]
 → GlobalAvgPool → concat 5 engineered features → Bottleneck(384)
 → heads: class (18), family (6), subcluster (7)
```

≈ 9.5 M parameters. The **`bottleneck` activation (384-d)** is the canonical
extraction point consumed by phases 4 (SAE), 5 (patching), 6 (steering),
7 (diagnostics), and 8 (motif discovery).

## Files

- `model.py` — `ChromatinCNNAttentionV2` + `build_model(cfg)`.
- `precompute_cache.py` — one-shot script that pre-bakes a mmap-friendly
  `.npy` cache (one-hot + 5-dim features + labels) per split so all DDP
  ranks share physical pages via the page cache.
- `train_ddp.py` — DDP training entrypoint (invoked through `torchrun`).
- `launch.sh` — convenience wrapper that launches `torch.distributed.run`
  with per-rank log redirection.

## Data flow per step

1. 5 DDP ranks mmap the same `train_onehot_cache.npy` (3.2 GB shared).
2. Each rank draws `batch_size_per_gpu` rows via `DistributedSampler`.
3. Forward through the model (optionally concatenated with the RC view if
   `rc_consistency_weight > 0`), cross-entropy + hierarchical aux losses.
4. Mixed-precision backward (bf16 on 3090 with fallback to fp16+scaler).
5. Cosine LR schedule w/ linear warmup.

Validation uses the balanced index from phase 1
(`phase1_filter/outputs/val_balanced_*perclass_seed*.npy`) via
`StreamingChromatinDataset` (mmap-backed, no per-rank RAM blow-up on the
12 GB val CSV). Macro-accuracy + RC-consistency are reported every epoch;
`best.pt` is saved when macro improves.

## Running

Prereqs: phase 1 subsample build (`phase1_filter/run_phase1.py`) and the
mmap caches (one-off):

```bash
# One-shot pre-bake (5 min) — only needs the train split for training.
python phase3_model/precompute_cache.py --splits train
```

Launch training on all 5 GPUs:

```bash
bash phase3_model/launch.sh                        # fresh run
bash phase3_model/launch.sh --resume results/phase3/checkpoints/last.pt
```

Per-rank logs stream to `/tmp/torchrun_logs/<run_id>/attempt_0/<rank>/`.
Checkpoints land in `results/phase3/checkpoints/` (`best.pt`, `last.pt`).
Training curves are appended to `results/phase3/logs/history.json`.

## Configuration (`config.json` → `phase3`)

Key knobs:
- `batch_size_per_gpu`, `num_workers_per_gpu`
- `learning_rate`, `warmup_epochs`, `num_epochs`, `lr_min`
- `family_loss_weight`, `subcluster_loss_weight` — hierarchical aux heads
- `rc_consistency_weight` — KL(orig‖rc) regulariser (concat-batched)
- `label_smoothing`, `dropout`, `weight_decay`, `gradient_clip_max_norm`

The model hyperparameters (`stem_filters`, `mid_filters`, `wide_filters`,
`attn_heads`, `attn_layers`, `bottleneck_dim`, ...) are also read from
`phase3` so you can sweep without touching code.
