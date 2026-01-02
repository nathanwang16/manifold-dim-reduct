# Training on Full Dataset - Quick Guide

## MPS Support ✅

All code now supports **MPS (Metal Performance Shaders)** for Mac M1/M2/M3 acceleration!

**Device Detection (auto):**
- CUDA (NVIDIA GPU) → MPS (Apple Silicon) → CPU (fallback)

**Verify MPS is available:**
```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
# Should print: MPS: True
```

## Configuration Changes

I've already updated `config.json` to use the full dataset:

### Changes Made:
```json
// Before (demo data):
{
  "train_sequences": "data/demo_train_sequences.csv",
  "train_labels": "data/demo_train_labels.csv",
  "val_sequences": "data/demo_val_sequences.csv",
  "val_labels": "data/demo_val_labels.csv",
  "test_sequences": "data/demo_test_sequences.csv"
}

// After (full dataset):
{
  "train_sequences": "data/train_sequences.csv",
  "train_labels": "data/train_labels.csv",
  "val_sequences": "data/val_sequences.csv",
  "val_labels": "data/val_labels.csv",
  "test_sequences": "data/testsequences.csv"
}
```

### To Switch Back to Demo Data:
Change paths back to:
```json
"train_sequences": "data/demo_train_sequences.csv",
"train_labels": "data/demo_train_labels.csv",
"val_sequences": "data/demo_val_sequences.csv",
"val_labels": "data/demo_val_labels.csv",
"test_sequences": "data/demo_test_sequences.csv"
```

---

## Run Training with Nohup

### Method 1: Using the Provided Script (Recommended)

```bash
# Make script executable (first time only)
chmod +x phase3_model/train_full_dataset.sh

# Run training
./phase3_model/train_full_dataset.sh
```

### Method 2: Direct Command

```bash
# Auto-detect device (MPS on Mac M1/M2/M3, CUDA on NVIDIA GPU)
nohup python -m phase3_model.run_phase3 \
    --config config.json \
    --mode both \
    --num_epochs 50 \
    --batch_size 256 \
    --device auto \
    > nohup_phase3_training.out 2>&1 &

# Force MPS (Mac M1/M2/M3)
nohup python -m phase3_model.run_phase3 \
    --config config.json \
    --mode both \
    --num_epochs 50 \
    --batch_size 256 \
    --device mps \
    > nohup_phase3_training.out 2>&1 &
```

---

## Monitoring Training

### Real-time Output:
```bash
# Tail the nohup output file
tail -f nohup_phase3_training.out
```

### Check Detailed Logs:
```bash
# List all log files
ls -lth logs/*phase3*

# View current phase3 log
tail -f logs/$(date +%Y%m%d)_phase3.log
```

### Check Training Metrics:
```bash
# View metrics (JSON format)
tail -f logs/$(date +%Y%m%d)_metrics.jsonl
```

---

## Managing the Training Process

### Check if Training is Running:
```bash
# Find Python processes
ps aux | grep "phase3_model.run_phase3"

# Or check for the specific PID (if you saved it)
ps -p <PID>
```

### Stop Training:
```bash
# Find the PID
ps aux | grep "phase3_model.run_phase3"
# Kill the process
kill <PID>

# Force kill if needed
kill -9 <PID>
```

### Resume Training:
The script doesn't support automatic resumption, but you can:
1. Let it complete training (saves best checkpoint automatically)
2. Use the checkpoint for inference:
   ```bash
   python -m phase3_model.run_phase3 \
       --config config.json \
       --mode inference \
       --checkpoint phase3_model/checkpoints/best_model.pt
   ```

---

## Expected Training Time

**Full Dataset Sizes:**
- Train: ~286,164 sequences (after RC augmentation: ~572,328)
- Val: ~71,541 sequences
- Test: ~100,008 sequences

**Estimated Time:**

| Device | Time per Epoch | 50 Epochs Total |
|--------|----------------|-------------------|
| CPU | ~10-15 min | ~8-12 hours |
| MPS (M1) | ~2-3 min | ~2-2.5 hours |
| MPS (M1 Pro/Max) | ~1-2 min | ~1-1.5 hours |
| CUDA (RTX 3080) | ~30-60 sec | ~30-60 min |

**With `--device auto` on Mac M1/M2/M3:**
- Automatically uses MPS (4-5x speedup vs CPU)
- Estimated: ~2-2.5 hours for 50 epochs
- Checkpoints saved to: `phase3_model/checkpoints/best_model.pt`

---

## Output Files

### Training Outputs:
- `phase3_model/checkpoints/best_model.pt`: Best model checkpoint
- `phase3_model/training_metrics.jsonl`: Training history
- `logs/YYYYMMDD_phase3.log`: Detailed training log
- `logs/YYYYMMDD_metrics.jsonl`: Structured metrics

### Inference Outputs:
- `predictions.csv`: Test set predictions (100,008 predictions)

---

## Troubleshooting

### Out of Memory:
Reduce batch size in nohup command:
```bash
--batch_size 128
# or
--batch_size 64
```

### Slow Training:
Increase workers in `config.json`:
```json
{
  "phase3": {
    "training": {
      "num_workers": 8  // or 16 for Mac M1 Pro/Max
    }
  }
}
```

### Checkpoint Not Saving:
- Ensure `phase3_model/checkpoints/` directory exists
- Check nohup output for errors
- Verify disk space: `df -h`

### Training Stopped Unexpectedly:
Check nohup output:
```bash
tail -n 100 nohup_phase3_training.out
```

Look for common errors:
- `CUDA out of memory` → Reduce batch size
- `File not found` → Check data paths in config.json
- `Permission denied` → Check file permissions

---

## Quick Reference Commands

```bash
# Start training
nohup python -m phase3_model.run_phase3 \
    --config config.json \
    --mode both \
    --num_epochs 50 \
    --batch_size 256 \
    --device auto \
    > nohup_phase3_training.out 2>&1 &

# Monitor progress
tail -f nohup_phase3_training.out

# Check logs
tail -f logs/$(date +%Y%m%d)_phase3.log

# Stop training
ps aux | grep phase3_model
kill <PID>

# Check checkpoint
ls -lh phase3_model/checkpoints/best_model.pt

# Inference only (after training)
python -m phase3_model.run_phase3 \
    --config config.json \
    --mode inference \
    --checkpoint phase3_model/checkpoints/best_model.pt
```

---

## After Training Completes

### Verify Predictions:
```bash
# Check predictions file exists
ls -lh predictions.csv

# Check number of predictions (should be 100008)
wc -l predictions.csv

# Preview first few predictions
head predictions.csv

# Check label distribution
cut -d',' -f1 predictions.csv | sort | uniq -c
```

### Prepare for Submission:
```bash
# Zip predictions file
zip predictions.zip predictions.csv

# Verify zip file
unzip -l predictions.zip
```

---

## Custom Parameters

You can modify these parameters in the nohup command:

- `--num_epochs 50`: Number of training epochs
- `--batch_size 256`: Batch size (reduce if OOM)
- `--device auto`: Device selection (auto, cuda, cpu)
- `--mode both`: Pipeline mode (train, inference, both)

Example with custom epochs:
```bash
nohup python -m phase3_model.run_phase3 \
    --config config.json \
    --mode both \
    --num_epochs 100 \
    --batch_size 256 \
    > nohup_phase3_training.out 2>&1 &
```

