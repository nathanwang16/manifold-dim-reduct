#!/usr/bin/env bash
# Launch DDP training across all 5 RTX 3090s in the `biohack` conda env.
# Usage:
#   bash phase3_model/launch.sh                # start fresh
#   bash phase3_model/launch.sh --resume results/phase3/checkpoints/last.pt
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS=4
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export CUDA_DEVICE_ORDER=PCI_BUS_ID

NUM_GPUS=${NUM_GPUS:-5}

PYBIN="/home/lemon/.conda/envs/biohack/bin/python"
mkdir -p /tmp/torchrun_logs
"$PYBIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    --redirects=3 \
    --tee=3 \
    --log-dir=/tmp/torchrun_logs \
    phase3_model/train_ddp.py "$@"
