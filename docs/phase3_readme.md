# Phase 3: Interpretability-Friendly CNN Architecture

This module implements the interpretability-focused CNN architecture for chromatin state prediction as described in Phase 3 of the guide.

## Overview

The ChromatinCNN is designed with mechanistic interpretability in mind:
- **Minimal polysemanticity**: Wider layers to reduce feature mixing
- **Preserved spatial information**: Global pooling only at final layer
- **Modular structure**: Clear separation between motif detection and decision making
- **RC equivariance**: Architecture treats forward and reverse complement symmetrically

## Architecture

```
Input: (batch, 200, 4) one-hot encoded

═══════════════════════════════════════════════════
MOTIF DETECTION BLOCK (Interpretable)
═══════════════════════════════════════════════════
[Conv1D] 128 filters, kernel_size=19, padding='same', ReLU
[BatchNorm]
[Conv1D] 256 filters, kernel_size=11, padding='same', ReLU
[BatchNorm]

═══════════════════════════════════════════════════
SPARSE BOTTLENECK (SAE Attachment Point)
═══════════════════════════════════════════════════
[Conv1D] 512 filters, kernel_size=1, ReLU
[BatchNorm]

═══════════════════════════════════════════════════
SPATIAL AGGREGATION
═══════════════════════════════════════════════════
[Global Max Pooling] → (batch, 512)
[Global Average Pooling] → (batch, 512)
[Concatenate] → (batch, 1024)

═══════════════════════════════════════════════════
DECISION BLOCK
═══════════════════════════════════════════════════
[Dense] 512 units, ReLU, Dropout(0.3)
[Dense] 256 units, ReLU, Dropout(0.3)
[Dense] 18 units, Softmax
```

## Files

- `model.py`: ChromatinCNN architecture and configuration
- `dataset.py`: Data loaders with biological augmentations (RC, jitter, noise)
- `train.py`: Training script with comprehensive logging
- `inference.py`: Inference script with RC averaging
- `run_phase3.py`: Main pipeline orchestrator

## Usage

### Quick Start

```bash
# Run complete pipeline (training + inference)
python -m phase3_model.run_phase3 \
    --config config.json \
    --mode both \
    --num_epochs 50 \
    --batch_size 256

# Run training only
python -m phase3_model.run_phase3 \
    --config config.json \
    --mode train \
    --num_epochs 50

# Run inference only (using existing checkpoint)
python -m phase3_model.run_phase3 \
    --config config.json \
    --mode inference \
    --checkpoint phase3_model/checkpoints/best_model.pt
```

### Using Demo Data

```bash
# Run with demo data for quick testing
python -m phase3_model.run_phase3 \
    --config config.json \
    --use_demo_data \
    --mode both \
    --num_epochs 10
```

## Configuration

Parameters are managed through `config.json` under the `phase3` section:

```json
{
  "phase3": {
    "n_classes": 18,
    "conv1_filters": 128,
    "conv2_filters": 256,
    "bottleneck_filters": 512,
    "kernel1": 19,
    "kernel2": 11,
    "dropout_rate": 0.3,
    "use_l1_regularization": true,
    "l1_weight": 1e-5,
    "label_smoothing": 0.05,
    "checkpoint_dir": "phase3_model/checkpoints",
    "training": {
      "num_workers": 4,
      "rc_augment": true,
      "jitter_prob": 0.3,
      "jitter_min_len": 180,
      "noise_prob": 0.01,
      "cache_data": true
    }
  }
}
```

### Key Configuration Parameters

**Model Architecture:**
- `conv1_filters`: Number of filters in first conv layer (motif detectors)
- `conv2_filters`: Number of filters in second conv layer
- `bottleneck_filters`: Number of bottleneck filters (SAE attachment point)
- `kernel1`: Kernel size for first conv layer (motif length)
- `kernel2`: Kernel size for second conv layer

**Regularization:**
- `use_l1_regularization`: Apply L1 penalty to first conv layer (encourages sparse motifs)
- `l1_weight`: L1 regularization coefficient
- `label_smoothing`: Label smoothing factor (prevents overconfidence)
- `dropout_rate`: Dropout probability in dense layers

**Training:**
- `rc_augment`: Apply reverse complement augmentation during training
- `jitter_prob`: Probability of position jittering
- `noise_prob`: Probability of noise injection per base
- `cache_data`: Cache one-hot encodings in memory

## Data Augmentation

The training data uses several biological augmentations:

1. **Reverse Complement (RC) Augmentation**:
   - Doubles training set size
   - Ensures model treats `GATTACA` and `TGTAATC` as biologically identical
   - Applied with 50% probability during training

2. **Position Jittering**:
   - Randomly crops 180-190bp windows from 200bp sequences
   - Forces model to recognize motifs regardless of position
   - Applied with 30% probability (configurable)

3. **Noise Injection**:
   - With low probability (1%), randomly mutates a base
   - Simulates sequencing errors
   - Improves generalization

## Training Protocol

The training follows the protocol specified in Phase 3.4 of the guide:

- **Optimizer**: AdamW with weight decay 1e-4
- **Learning Rate Schedule**:
  - Warmup: Linear increase from 1e-5 to 1e-3 over first 5 epochs
  - Main: Cosine annealing from 1e-3 to 1e-6 over remaining epochs
- **Batch Size**: 256 (adjustable)
- **Early Stopping**: Patience=10 on validation loss
- **Gradient Clipping**: Max norm=1.0

## Interpretability Features

The model includes several features designed for mechanistic interpretability:

1. **Extract First Conv Filters**:
   ```python
   filters = model.get_first_conv_filters()  # Shape: (128, 4, 19)
   # These can be visualized as sequence logos for motif discovery
   ```

2. **Intermediate Activations**:
   ```python
   logits, activations = model(x, return_activations=True)
   # Returns dict with conv1, conv2, bottleneck, dense1, dense2 activations
   ```

3. **Position Information**:
   ```python
   logits, _, positions = model(x, return_positions=True)
   # Returns positions of max activations for each filter
   ```

4. **RC Averaging for Inference**:
   ```python
   predictions = model.predict_with_rc_consistency(x)
   # Averages predictions from forward and reverse complement strands
   ```

## Output Files

### Checkpoints
- `phase3_model/checkpoints/best_model.pt`: Best model checkpoint (by validation accuracy)
- Contains model weights, optimizer state, scheduler state, and configuration

### Training Metrics
- `phase3_model/training_metrics.jsonl`: Training history with losses and accuracies
- `logs/YYYYMMDD_metrics.jsonl`: Structured metrics log

### Logs
- `logs/YYYYMMDD_phase3.log`: Detailed training log with timestamps
- Color-coded console output for easy monitoring

### Predictions
- `predictions.csv`: Test set predictions (one prediction per line, 0-17)
- Can be zipped for submission to Codabench

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config or command line
- Set `cache_data: false` in config to reduce memory usage

### Slow Training
- Increase `num_workers` for data loading (up to number of CPU cores)
- Use GPU (`--device cuda`) if available

### Poor Performance
- Increase training epochs
- Adjust learning rate schedule
- Try different augmentation parameters
- Consider tuning model architecture (filter counts, kernel sizes)

## Next Steps

After training, proceed to:
- **Phase 4**: Mechanistic Interpretability (filter visualization, DeepLIFT, etc.)
- **Phase 5**: Sparse Autoencoders for feature decomposition
- **Phase 6**: Steering and alignment techniques

## Notes

- All computational intensive code utilizes Mac M1 multi-core CPU or MPS for parallel computing
- Use demo datasets in `data/` folder for quick testing and debugging
- The bottleneck layer is designed for SAE attachment in Phase 5
- Global max + average pooling provides complementary information for interpretability













