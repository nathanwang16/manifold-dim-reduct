# Chromatin State Prediction Challenge: "The Discovery Engine"

This repository contains the solution for the **Chromatin State Prediction Challenge** at the UCLA MiniHack. The goal is to predict **ChromHMM chromatin state annotations** (18 classes) for **200bp DNA sequences**.

Because the labels are provided as abstract integers (1–18), this solution employs a **"Discovery Engine"** approach: we treat the labels as unknown biological states and use manifold learning, interpretable deep learning, and feature analysis to "rediscover" their biological meaning (e.g., Promoter, Enhancer, Heterochromatin).

## Repository Structure

```
manifold-dim-reduct/
├── phase1_filter/       # Data engineering & augmentation
├── phase2_manifold/     # Manifold learning (UMAP/PHATE) & visualization
├── phase3_model/        # Main CNN model (training & inference)
├── phase6_steering/     # Steering vectors & alignment analysis
├── phase8/              # Label-to-Biological State identification
├── data/                # Dataset folder (input CSVs)
├── guide.md             # Detailed research master plan
└── README.md            # This file
```

## Key Components

### 1. Data Engineering (`phase1_filter/`)
- **Reverse Complement Augmentation**: Ensures the model treats forward/reverse strands symmetrically.
- **Hierarchy Extraction**: Infers super-families (e.g., "Active", "Repressed") from label similarities.

### 2. Manifold Analysis (`phase2_manifold/`)
- Maps the 18 states into a low-dimensional functional landscape.
- Uses **PHATE** and **UMAP** on k-mer frequencies to visualize label relationships.

### 3. The "Discovery Engine" Model (`phase3_model/`)
- **Architecture**: `ChromatinCNNAttention`
  - 1D-CNN backbone for motif detection.
  - Self-attention mechanism for long-range dependencies.
  - Global pooling for position invariance.
- **Training**:
  - RC-consistency loss.
  - Hierarchical multitask learning (optional).
  - Data-driven filter initialization (Mechanistic Interpretability fix).

### 4. Steering & Alignment (`phase6_steering/`)
- **Steering Vectors**: Calculate directions in activation space that shift predictions from one state to another.
- **Inference-Time Intervention**: Use steering to resolve confusion between similar states.

### 5. Biological Mapping (`phase8/`)
- Matches the abstract labels (1-18) to biological states using feature profiles (GC content, CpG ratio, Repeats).
- Uses **Hungarian Assignment** to optimally map learned clusters to known ChromHMM states.

## Usage

### Training
```bash
python phase3_model/run_phase3.py --config config.json
```

### Inference
```bash
python phase3_model/inference.py
```

### Analysis
```bash
python phase6_steering/analysis.py
```

## Competition Specs
- **Input**: 200bp DNA (A, C, G, T).
- **Output**: Integer label 1–18.
- **Metric**: Accuracy.
- **Constraints**: No external data, no specialized DNA software (standard ML libs only).
