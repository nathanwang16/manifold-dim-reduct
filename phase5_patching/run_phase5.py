"""Phase 5 orchestrator: patching + layer DLA."""

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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="config.json", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    base_out = Path(cfg.get("phase5_patching", {}).get(
        "output_dir", "results/phase5_patching"))
    base_out.mkdir(parents=True, exist_ok=True)

    if args.force or not (base_out / "patching_results.csv").exists():
        run([
            sys.executable, "phase5_patching/patch_experiments.py",
            "--checkpoint", str(args.checkpoint),
            "--config", str(args.config),
            "--output_dir", str(base_out),
            "--device", args.device,
        ])
    else:
        print("skip patching: patching_results.csv exists")

    if args.force or not (base_out / "layer_dla.json").exists():
        run([
            sys.executable, "phase5_patching/layer_dla.py",
            "--checkpoint", str(args.checkpoint),
            "--config", str(args.config),
            "--output_dir", str(base_out),
            "--device", args.device,
        ])
    else:
        print("skip layer DLA: layer_dla.json exists")


if __name__ == "__main__":
    main()
