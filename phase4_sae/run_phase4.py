"""Phase 4 end-to-end orchestrator.

Chains:
    1. extract_activations.py   -> results/phase4_sae/activations/
    2. train_sae.py             -> results/phase4_sae/sae.pt + history.json
    3. feature_analysis.py      -> results/phase4_sae/feature_analysis/

Each step is skipped if its outputs already exist (unless --force).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> None:
    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Trained phase3 checkpoint (best.pt or last.pt)")
    parser.add_argument("--config", default="config.json", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sae_type", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    base_out = Path(cfg.get("phase4_sae", {}).get("output_dir", "results/phase4_sae"))
    base_out.mkdir(parents=True, exist_ok=True)

    act_dir = base_out / "activations"
    bottleneck_path = act_dir / "bottleneck.npy"
    if args.force or not bottleneck_path.exists():
        run([
            sys.executable, "phase4_sae/extract_activations.py",
            "--checkpoint", str(args.checkpoint),
            "--config", str(args.config),
            "--output_dir", str(act_dir),
            "--device", args.device,
        ])
    else:
        print(f"skip extract: {bottleneck_path} exists")

    sae_path = base_out / "sae.pt"
    if args.force or not sae_path.exists():
        cmd = [
            sys.executable, "phase4_sae/train_sae.py",
            "--config", str(args.config),
            "--activations_dir", str(act_dir),
            "--output_dir", str(base_out),
            "--device", args.device,
        ]
        if args.sae_type:
            cmd.extend(["--override_sae_type", args.sae_type])
        run(cmd)
    else:
        print(f"skip train: {sae_path} exists")

    feat_dir = base_out / "feature_analysis"
    if args.force or not (feat_dir / "features.json").exists():
        run([
            sys.executable, "phase4_sae/feature_analysis.py",
            "--config", str(args.config),
            "--sae_path", str(sae_path),
            "--activations_dir", str(act_dir),
            "--output_dir", str(feat_dir),
            "--device", args.device,
        ])
    else:
        print(f"skip feature analysis: {feat_dir}/features.json exists")


if __name__ == "__main__":
    main()
