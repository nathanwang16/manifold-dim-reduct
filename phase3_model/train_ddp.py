"""DistributedDataParallel training for ChromatinCNNAttentionV2.

Launch with `torchrun --nproc_per_node=5 phase3_model/train_ddp.py [--args]`
or via the helper `phase3_model/launch.sh`.

Design notes:
  * Each rank loads the full in-memory train set (~3 GB one-hot) and a
    balanced validation subset (~360 MB) via `chromatin_lib`.
  * Cross-entropy on the 18-way label + optional auxiliary losses on the
    family (6-way) and subcluster (7-way) heads.
  * RC-consistency loss: for a random half of batches we also forward the
    reverse-complement and add KL(softmax(orig) || softmax(rc)).
  * Mixed-precision via `torch.cuda.amp`, NCCL backend.
  * Rank-0 saves `best_val_acc` (balanced macro-F1 proxy) + a rolling
    `last.pt` checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chromatin_lib import (  # noqa: E402
    CachedMmapDataset,
    InMemoryChromatinDataset,
    LABEL_TO_FAMILY,
    StreamingChromatinDataset,
    build_line_offset_index,
    collate_chromatin,
    merged_split_paths,
    reverse_complement_onehot,
)
from phase3_model.model import ChromatinCNNAttentionV2, build_model  # noqa: E402


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed() -> Dict[str, int]:
    if "LOCAL_RANK" not in os.environ:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    print(f"[rank {rank}/{world_size}] pid={os.getpid()} cuda:{local_rank} "
          f"{torch.cuda.get_device_name(local_rank)} -- initialising process group...",
          flush=True)
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        print(f"[rank {rank}/{world_size}] pg ready", flush=True)
    return {"rank": rank, "world_size": world_size, "local_rank": local_rank}


def is_main_process() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def all_reduce_mean(value: float | torch.Tensor) -> float:
    if not dist.is_initialized():
        return float(value)
    t = torch.as_tensor(value, dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / dist.get_world_size())


def all_gather_tensor(t: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return t
    world = dist.get_world_size()
    bufs = [torch.empty_like(t) for _ in range(world)]
    dist.all_gather(bufs, t.contiguous())
    return torch.cat(bufs, dim=0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def build_datasets(cfg: Dict[str, Any]):
    """Return (train_ds, balanced_val_ds)."""
    data_cfg = cfg["dataset"]
    p3 = cfg["phase3"]

    train_paths = merged_split_paths("train")
    merged_dir = train_paths["sequences"].parent
    train_onehot_cache = merged_dir / "train_onehot_cache.npy"
    train_feat_cache = merged_dir / "train_features_cache.npy"
    train_label_cache = merged_dir / "train_labels_cache.npy"

    use_feat = bool(p3.get("use_engineered_features", True))
    jitter_prob = float(p3.get("jitter_prob", 0.3))
    noise_prob = float(p3.get("noise_prob", 0.02))

    if train_onehot_cache.exists() and train_label_cache.exists():
        if is_main_process():
            print(f"Using mmap cache from {merged_dir}")
        train_ds = CachedMmapDataset(
            onehot_cache_path=train_onehot_cache,
            features_cache_path=train_feat_cache if use_feat else None,
            labels_cache_path=train_label_cache,
            jitter_prob=jitter_prob,
            jitter_min_len=int(p3.get("jitter_min_len", 170)),
            noise_prob=noise_prob,
            rc_prob=0.0,
            sequence_length=int(data_cfg.get("sequence_length", 200)),
        )
    else:
        if is_main_process():
            print(
                "No mmap cache found. Falling back to per-rank in-memory decode. "
                "Run `python phase3_model/precompute_cache.py --splits train` to speed "
                "up startup and share RAM across ranks."
            )
        train_ds = InMemoryChromatinDataset(
            sequences_path=train_paths["sequences"],
            labels_path=train_paths["labels"],
            jitter_prob=jitter_prob,
            jitter_min_len=int(p3.get("jitter_min_len", 170)),
            noise_prob=noise_prob,
            rc_prob=0.0,
            sequence_length=int(data_cfg.get("sequence_length", 200)),
            cache_onehot=False,  # save RAM: encode on demand
            compute_features=use_feat,
        )

    # Balanced val subset: re-use phase1 index
    phase1_dir = Path(cfg["phase1"]["output_dir"])
    per_class = int(p3["eval"].get("val_balanced_per_class", 5000))
    seed = int(cfg["phase1"]["seed"])
    candidates = [
        phase1_dir / f"val_balanced_{per_class}perclass_seed{seed}.npy",
        phase1_dir / f"val_balanced_{int(cfg['phase1']['eval_subsample_per_class'])}perclass_seed{seed}.npy",
    ]
    val_idx_path: Optional[Path] = None
    for candidate in candidates:
        if candidate.exists():
            val_idx_path = candidate
            break
    if val_idx_path is None:
        raise FileNotFoundError(
            f"No balanced val index found (looked for: {candidates}). "
            "Run phase1 with `--splits val` first."
        )
    val_idx = np.load(val_idx_path)
    if per_class * 18 < val_idx.size:
        rng = np.random.default_rng(seed)
        selected = rng.choice(val_idx, size=min(per_class * 18, val_idx.size), replace=False)
        val_idx = np.sort(selected)
    val_paths = merged_split_paths("val")
    val_ds = StreamingChromatinDataset(
        sequences_path=val_paths["sequences"],
        labels_path=val_paths["labels"],
        sample_indices=val_idx,
        sequence_length=int(data_cfg.get("sequence_length", 200)),
        compute_features=use_feat,
    )
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def build_scheduler(optimizer: torch.optim.Optimizer, p3: Dict[str, Any], steps_per_epoch: int):
    total_epochs = int(p3.get("num_epochs", 30))
    warmup_epochs = int(p3.get("warmup_epochs", 3))
    lr_min = float(p3.get("lr_min", 1e-6))
    start_factor = float(p3.get("warmup_start_factor", 1e-2))
    warmup_iters = max(1, warmup_epochs * max(1, steps_per_epoch))
    cosine_iters = max(1, total_epochs * max(1, steps_per_epoch) - warmup_iters)
    warmup = LinearLR(optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_iters)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_iters, eta_min=lr_min)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_iters])


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def prepare_batch(batch: Dict[str, torch.Tensor], device: torch.device, use_feat: bool):
    x = batch["x"].to(device, non_blocking=True)
    y = batch["y"].to(device, non_blocking=True)
    fam = batch["family"].to(device, non_blocking=True)
    sub = batch["subcluster"].to(device, non_blocking=True)
    feat = batch["feat"].to(device, non_blocking=True) if use_feat else None
    return x, feat, y, fam, sub


def compute_losses(
    out: Dict[str, torch.Tensor],
    y: torch.Tensor,
    fam: torch.Tensor,
    sub: torch.Tensor,
    p3: Dict[str, Any],
    label_smoothing: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    ce = F.cross_entropy(out["logits"], y, label_smoothing=label_smoothing)
    loss = float(p3.get("main_ce_weight", 1.0)) * ce
    meters = {"ce": ce.detach().item()}

    if "logits_family" in out and p3.get("family_loss_weight", 0.0):
        fam_loss = F.cross_entropy(out["logits_family"], fam)
        loss = loss + float(p3["family_loss_weight"]) * fam_loss
        meters["family"] = fam_loss.detach().item()
    if "logits_subcluster" in out and p3.get("subcluster_loss_weight", 0.0):
        sub_loss = F.cross_entropy(out["logits_subcluster"], sub)
        loss = loss + float(p3["subcluster_loss_weight"]) * sub_loss
        meters["subcluster"] = sub_loss.detach().item()
    return loss, meters


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_feat: bool,
    amp_dtype: torch.dtype,
) -> Dict[str, float]:
    model.eval()
    total = 0
    correct_class = 0
    per_class_total = torch.zeros(18, device=device)
    per_class_correct = torch.zeros(18, device=device)
    rc_match = 0
    rc_total = 0
    for batch in loader:
        x, feat, y, fam, sub = prepare_batch(batch, device, use_feat)
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            out = model(x, engineered=feat)
            logits = out["logits"]
            preds = logits.argmax(dim=-1)
            # RC consistency probe (half-batch)
            if x.shape[0] >= 2:
                x_rc = reverse_complement_onehot(x)
                out_rc = model(x_rc, engineered=feat)
                preds_rc = out_rc["logits"].argmax(dim=-1)
                rc_match += (preds_rc == preds).sum().item()
                rc_total += preds.shape[0]
        correct_class += (preds == y).sum().item()
        total += y.shape[0]
        for c in range(18):
            mask = y == c
            if mask.any():
                per_class_total[c] += mask.sum()
                per_class_correct[c] += (preds[mask] == c).sum()

    if dist.is_initialized():
        t = torch.tensor([correct_class, total, rc_match, rc_total], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        dist.all_reduce(per_class_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(per_class_correct, op=dist.ReduceOp.SUM)
        correct_class, total, rc_match, rc_total = int(t[0]), int(t[1]), int(t[2]), int(t[3])

    pct = per_class_total.cpu().numpy()
    pcc = per_class_correct.cpu().numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc = np.where(pct > 0, pcc / pct, 0.0)
    macro_acc = float(per_class_acc.mean())

    return {
        "overall_acc": float(correct_class / max(1, total)),
        "macro_acc": macro_acc,
        "rc_consistency": float(rc_match / max(1, rc_total)),
        "per_class_acc": per_class_acc.tolist(),
    }


def train(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    dist_info = setup_distributed()
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]
    local_rank = dist_info["local_rank"]
    device = torch.device(f"cuda:{local_rank}")

    seed = int(cfg["phase1"].get("seed", 20260408)) + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True

    p3 = cfg["phase3"]

    # Rank 0 builds any missing val offset caches before all ranks touch the
    # streaming dataset, to avoid multi-GB redundant scans.
    if is_main_process():
        val_paths = merged_split_paths("val")
        for key in ("sequences", "labels"):
            cache = val_paths[key].with_suffix(val_paths[key].suffix + ".offsets.npy")
            if not cache.exists():
                print(f"[rank 0] Building offset index for {val_paths[key]} ...", flush=True)
                build_line_offset_index(val_paths[key])
    if dist.is_initialized():
        dist.barrier()

    train_ds, val_ds = build_datasets(cfg)
    use_feat = bool(p3.get("use_engineered_features", True))

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    ) if world_size > 1 else None
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    ) if world_size > 1 else None

    bs = int(p3.get("batch_size_per_gpu", 1024))
    nw = int(p3.get("num_workers_per_gpu", 6))
    train_loader = DataLoader(
        train_ds, batch_size=bs, sampler=train_sampler,
        shuffle=(train_sampler is None), num_workers=nw,
        pin_memory=True, drop_last=True, persistent_workers=(nw > 0),
        collate_fn=collate_chromatin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(bs, 2048), sampler=val_sampler, shuffle=False,
        num_workers=max(2, nw // 2), pin_memory=True, drop_last=False,
        persistent_workers=(nw > 0),
        collate_fn=collate_chromatin,
    )

    model = build_model(cfg).to(device)
    if is_main_process():
        n = sum(p.numel() for p in model.parameters())
        print(f"[rank {rank}] Model built: {n/1e6:.2f}M params")

    # Data-driven stem init (rank 0 only, then broadcast).
    if is_main_process():
        with torch.no_grad():
            init_n = min(4096, len(train_ds))
            samples = []
            for i in range(init_n):
                item = train_ds[i]
                samples.append(item["x"].numpy())
            stack = torch.from_numpy(np.stack(samples)).transpose(1, 2).contiguous().to(device)
            model.init_stem_from_data(stack)
    if dist.is_initialized():
        for param in model.parameters():
            dist.broadcast(param.data, src=0)

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    lr = float(p3.get("learning_rate", 3e-3))
    wd = float(p3.get("weight_decay", 1e-3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = build_scheduler(optimizer, p3, steps_per_epoch)

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16 and bool(p3.get("use_amp", True))))

    ckpt_dir = Path(p3.get("checkpoint_dir", "results/phase3/checkpoints"))
    log_dir = Path(p3.get("log_dir", "results/phase3/logs"))
    if is_main_process():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "config_used.json").write_text(json.dumps(cfg, indent=2))

    num_epochs = int(p3.get("num_epochs", 30))
    label_smoothing = float(p3.get("label_smoothing", 0.05))
    rc_consistency_weight = float(p3.get("rc_consistency_weight", 0.0))
    patience = int(p3.get("early_stopping_patience", 10))
    clip = float(p3.get("gradient_clip_max_norm", 2.0))

    # Resume
    start_epoch = 1
    best_macro = 0.0
    no_improve = 0
    resume = args.resume or p3.get("resume_from")
    if resume:
        if is_main_process():
            print(f"Resuming from {resume}")
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        core = model.module if world_size > 1 else model
        core.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_macro = float(ckpt.get("best_macro", 0.0))

    history: list[Dict[str, Any]] = []

    for epoch in range(start_epoch, num_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        running = {"loss": 0.0, "ce": 0.0, "family": 0.0, "subcluster": 0.0, "rc": 0.0}
        correct = 0
        seen = 0
        for step, batch in enumerate(train_loader):
            x, feat, y, fam, sub = prepare_batch(batch, device, use_feat)

            optimizer.zero_grad(set_to_none=True)
            do_rc = rc_consistency_weight > 0 and random.random() < 0.5
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                if do_rc:
                    # Concatenate original + RC into one forward so BN running
                    # stats are updated exactly once per step.
                    x_rc = reverse_complement_onehot(x)
                    x_cat = torch.cat([x, x_rc], dim=0)
                    feat_cat = torch.cat([feat, feat], dim=0) if feat is not None else None
                    out_cat = model(x_cat, engineered=feat_cat)
                    bsz = x.shape[0]
                    out = {
                        "logits": out_cat["logits"][:bsz],
                        "bottleneck": out_cat["bottleneck"][:bsz],
                    }
                    if "logits_family" in out_cat:
                        out["logits_family"] = out_cat["logits_family"][:bsz]
                    if "logits_subcluster" in out_cat:
                        out["logits_subcluster"] = out_cat["logits_subcluster"][:bsz]
                    logits_rc = out_cat["logits"][bsz:]

                    loss, meters = compute_losses(out, y, fam, sub, p3, label_smoothing)
                    p_orig = F.log_softmax(out["logits"], dim=-1)
                    p_rc = F.softmax(logits_rc, dim=-1)
                    rc_loss = F.kl_div(p_orig, p_rc, reduction="batchmean")
                    loss = loss + rc_consistency_weight * rc_loss
                    meters["rc"] = rc_loss.detach().item()
                else:
                    out = model(x, engineered=feat)
                    loss, meters = compute_losses(out, y, fam, sub, p3, label_smoothing)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            scheduler.step()

            preds = out["logits"].argmax(dim=-1)
            correct += (preds == y).sum().item()
            seen += y.shape[0]
            for k, v in meters.items():
                running[k] += v
            running["loss"] += loss.item()

            if is_main_process() and (step % 25 == 0):
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"[ep {epoch:02d} step {step:04d}/{steps_per_epoch}] "
                    f"loss {loss.item():.4f}  ce {meters['ce']:.4f}  "
                    f"lr {lr_now:.2e}  acc_batch {(preds == y).float().mean().item():.3f}"
                )

        train_acc = correct / max(1, seen)
        train_acc = all_reduce_mean(train_acc)
        for k in running:
            running[k] /= max(1, steps_per_epoch)
            running[k] = all_reduce_mean(running[k])

        # Validation
        val_metrics = evaluate(model, val_loader, device, use_feat, amp_dtype)

        if is_main_process():
            dt = time.time() - t0
            row = {
                "epoch": epoch,
                "train_acc": train_acc,
                **{f"train_{k}": v for k, v in running.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "epoch_seconds": dt,
            }
            history.append(row)
            print(
                f"== Epoch {epoch:03d} done in {dt:.1f}s | train_acc {train_acc:.4f}"
                f" | val overall {val_metrics['overall_acc']:.4f}"
                f"  macro {val_metrics['macro_acc']:.4f}"
                f"  rc_cons {val_metrics['rc_consistency']:.4f} =="
            )
            (log_dir / "history.json").write_text(json.dumps(history, indent=2))

            core = model.module if world_size > 1 else model
            state = {
                "epoch": epoch,
                "model": core.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_macro": best_macro,
                "config": cfg,
                "model_config": asdict(core.cfg),
            }
            torch.save(state, ckpt_dir / "last.pt")
            if val_metrics["macro_acc"] > best_macro:
                best_macro = val_metrics["macro_acc"]
                state["best_macro"] = best_macro
                torch.save(state, ckpt_dir / "best.pt")
                print(f"  -> new best macro_acc {best_macro:.4f} (saved best.pt)")
                no_improve = 0
            else:
                no_improve += 1
        # Broadcast no_improve/best_macro to all ranks
        flag = torch.tensor([no_improve, int(best_macro * 1e6)], dtype=torch.int64, device=device)
        if dist.is_initialized():
            dist.broadcast(flag, src=0)
        no_improve = int(flag[0].item())
        if no_improve >= patience:
            if is_main_process():
                print(f"Early stop: {no_improve} epochs without macro_acc improvement.")
            break

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--resume", default=None, help="Optional checkpoint path to resume from.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    train(cfg, args)


if __name__ == "__main__":
    main()
