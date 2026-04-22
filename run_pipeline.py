"""Top-level orchestrator for phases 4 through 8.

Phase 0 (download ~100 GB), phase 1 (balanced indices), phase 2 (manifold
learning) and phase 3 (DDP training) are long-running enough that they
are kept as separate entrypoints — this orchestrator assumes a trained
`results/phase3/checkpoints/best.pt` already exists, and chains the
single-GPU MI phases 4-8 end-to-end.

Usage:
    python run_pipeline.py --checkpoint results/phase3/checkpoints/best.pt
    python run_pipeline.py --checkpoint ... --phases 4,5
    python run_pipeline.py --checkpoint ... --force   # re-run even if outputs exist
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

PHASES = {
    4: ("phase4_sae/run_phase4.py",         "Sparse Autoencoder on bottleneck"),
    5: ("phase5_patching/run_phase5.py",    "Activation patching + DLA"),
    6: ("phase6_steering/run_phase6.py",    "Steering vectors + sweeps"),
    7: ("phase7_diagnostics/run_phase7.py", "Saliency + calibration + RC consistency"),
    8: ("phase8_motifs/run_phase8.py",      "Stem motifs + ISM + hypotheses"),
}


def _run(script: Path, args: list[str]) -> None:
    cmd = [sys.executable, str(script)] + args
    print("\n" + "=" * 72)
    print("$", " ".join(cmd))
    print("=" * 72, flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to results/phase3/checkpoints/best.pt")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--phases", default="4,5,6,7,8",
                        help="Comma-separated list of phase numbers (4-8).")
    parser.add_argument("--force", action="store_true",
                        help="Forward --force to each phase orchestrator.")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    want = [int(x) for x in args.phases.split(",") if x.strip()]
    for p in want:
        if p not in PHASES:
            raise ValueError(f"Unknown phase {p}; valid: {sorted(PHASES)}")
        script, desc = PHASES[p]
        print(f"\n### Phase {p}: {desc}")
        shared = [
            "--checkpoint", str(args.checkpoint),
            "--config", str(args.config),
            "--device", args.device,
        ]
        if args.force:
            shared.append("--force")
        _run(REPO_ROOT / script, shared)

    print("\nAll phases completed.")


if __name__ == "__main__":
    main()
