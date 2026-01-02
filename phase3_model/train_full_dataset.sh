#!/bin/bash

# Phase 3: Train ChromatinCNN on Full Dataset
# This script runs training with nohup for background execution

# Configuration
CONFIG_FILE="config.json"
NUM_EPOCHS=50
BATCH_SIZE=512
DEVICE="auto"  # auto, cuda, mps, or cpu
MODE="both"     # train, inference, or both

# Log files
LOG_DIR="logs"
NOHUP_OUT="nohup_phase3_resume_fix.out"

# Create log directory if it doesn't exist
mkdir -p $LOG_DIR

echo "=========================================="
echo "Phase 3 Training - Full Dataset"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Mode: $MODE"
echo "Epochs: $NUM_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Device: $DEVICE"
echo "Output log: $NOHUP_OUT"
echo "=========================================="

# Run training with nohup
nohup python -m phase3_model.run_phase3 \
    --config $CONFIG_FILE \
    --mode $MODE \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --device $DEVICE \
    > $NOHUP_OUT 2>&1 &

# Get the process ID
PID=$!

echo ""
echo "Training started in background!"
echo "Process ID: $PID"
echo ""
echo "Monitor training:"
echo "  tail -f $NOHUP_OUT"
echo ""
echo "Check logs:"
echo "  ls -lth $LOG_DIR/*phase3*"
echo ""
echo "Stop training if needed:"
echo "  kill $PID"
echo ""

