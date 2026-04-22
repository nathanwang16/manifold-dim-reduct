"""Train an SAE on cached bottleneck activations.

Inputs (produced by `extract_activations.py`):
    {activations_dir}/bottleneck.npy  float32  (N, D)
    {activations_dir}/labels.npy      int64    (N,)

Outputs (written to {output_dir}):
    sae.pt               torch state dict
    history.json         loss / L0 / dead-feature trajectory
    config_used.json     the config block actually used
    feature_stats.npz    per-feature activation histogram snapshot

Single-GPU training. For D=384 → F=4096 and N=216k the whole thing fits in
~6 GB of GPU RAM and trains in a few minutes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from phase4_sae.sae import build_sae  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json", type=Path)
    parser.add_argument("--activations_dir", required=True, type=Path)
    parser.add_argument("--output_dir", default=None, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--override_sae_type", default=None,
                        choices=[None, "topk", "l1"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    p4 = cfg.get("phase4_sae", {})
    if args.override_sae_type:
        p4 = {**p4, "sae_type": args.override_sae_type}

    output_dir = args.output_dir or Path(p4.get("output_dir", "results/phase4_sae"))
    output_dir.mkdir(parents=True, exist_ok=True)

    bottleneck = np.load(args.activations_dir / "bottleneck.npy")
    labels = np.load(args.activations_dir / "labels.npy")
    n, d = bottleneck.shape
    print(f"Activations: {n:,} x {d} (loaded from {args.activations_dir})")

    # Rescale: subtract mean so the SAE's pre-sub-b_dec trick is better centered.
    mean = bottleneck.mean(axis=0, keepdims=True)
    bottleneck = bottleneck - mean

    x_all = torch.from_numpy(bottleneck).float()
    y_all = torch.from_numpy(labels).long()
    ds = TensorDataset(x_all, y_all)
    loader = DataLoader(ds, batch_size=int(p4.get("batch_size", 4096)),
                        shuffle=True, num_workers=2, pin_memory=True)

    sae = build_sae({"phase4_sae": {**p4, "activation_dim": d}}).to(args.device)
    optimizer = torch.optim.AdamW(sae.parameters(), lr=float(p4.get("lr", 1e-3)))

    # Initialise b_dec from the data mean (classic Anthropic trick).
    with torch.no_grad():
        sae.b_dec.copy_(torch.zeros(d, device=args.device))  # already zero-centered
    sae.normalize_decoder()

    num_epochs = int(p4.get("num_epochs", 20))
    history: list = []
    feature_sum = torch.zeros(sae.feature_dim, device=args.device)
    feature_hits = torch.zeros(sae.feature_dim, device=args.device)
    total_seen = 0

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        running = {"loss": 0.0, "recon": 0.0, "l0": 0.0, "l1": 0.0}
        n_steps = 0
        feature_sum.zero_()
        feature_hits.zero_()
        total_seen = 0

        for xb, _ in loader:
            xb = xb.to(args.device, non_blocking=True)
            out = sae(xb)
            loss = out["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Project out the gradient component along each dictionary direction
            # (Anthropic trick: enforce unit-norm dictionary while using Adam).
            with torch.no_grad():
                # Remove radial component of grad on W_dec (since norm is constrained)
                proj = (sae.W_dec.grad * sae.W_dec).sum(dim=0, keepdim=True)
                sae.W_dec.grad.sub_(proj * sae.W_dec)
            optimizer.step()
            sae.normalize_decoder()

            running["loss"] += float(out["loss"])
            running["recon"] += float(out["recon"])
            running["l0"] += float(out["l0"])
            running["l1"] += float(out.get("l1", torch.tensor(0.0)))
            n_steps += 1

            h = out["h"].detach()
            feature_sum += h.sum(dim=0)
            feature_hits += (h > 0).float().sum(dim=0)
            total_seen += xb.shape[0]

        for k in running:
            running[k] /= max(1, n_steps)
        dead = (feature_hits == 0).float().mean().item()
        mean_act = float(feature_sum.sum().item() / max(1, total_seen))
        row = {
            "epoch": epoch,
            **running,
            "dead_frac": dead,
            "mean_activation_sum": mean_act,
            "seconds": time.time() - t0,
        }
        history.append(row)
        print(
            f"[sae ep {epoch:02d}] loss {running['loss']:.4f}  recon {running['recon']:.4f}"
            f"  L0 {running['l0']:.1f}  dead {dead:.3%}  ({row['seconds']:.1f}s)"
        )

    # Final evaluation on the full set (reconstruction R^2 per-class).
    sae.eval()
    with torch.no_grad():
        h_all = torch.zeros(n, sae.feature_dim)
        x_hat_all = torch.zeros(n, d)
        chunk = 8192
        for i in range(0, n, chunk):
            xb = x_all[i : i + chunk].to(args.device)
            out = sae(xb)
            h_all[i : i + chunk] = out["h"].cpu()
            x_hat_all[i : i + chunk] = out["x_hat"].cpu()
    recon_err = ((x_hat_all - x_all) ** 2).sum(dim=1).mean().item()
    total_var = (x_all - x_all.mean(0, keepdim=True)).pow(2).sum(dim=1).mean().item()
    r2 = 1.0 - recon_err / max(1e-8, total_var)
    print(f"Final R^2 = {r2:.4f}")

    # Save everything
    torch.save({
        "state_dict": sae.state_dict(),
        "activation_mean": mean,
        "config": p4,
        "activation_dim": d,
        "feature_dim": sae.feature_dim,
        "sae_type": p4.get("sae_type", "topk"),
    }, output_dir / "sae.pt")

    per_feature_mean_act = (feature_sum / max(1, total_seen)).cpu().numpy()
    firing_rate = (feature_hits / max(1, total_seen)).cpu().numpy()
    np.savez(
        output_dir / "feature_stats.npz",
        per_feature_mean_act=per_feature_mean_act,
        firing_rate=firing_rate,
        dead_mask=(firing_rate == 0),
    )

    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    (output_dir / "config_used.json").write_text(json.dumps(p4, indent=2))
    (output_dir / "final_metrics.json").write_text(json.dumps({
        "R2": r2,
        "recon_err": recon_err,
        "dead_frac": float((firing_rate == 0).mean()),
        "mean_L0": float(history[-1]["l0"]) if history else 0.0,
    }, indent=2))
    print(f"Saved SAE to {output_dir}/sae.pt")


if __name__ == "__main__":
    main()
