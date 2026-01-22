# Chromatin State Label Identification: Strategic Plan

## Executive Summary

**Objective**: Identify which competition labels (1-18) correspond to which known biological chromatin states to leverage domain knowledge for improved predictions.

**Feasibility Assessment**: **HIGH** - Each chromatin state has well-characterized sequence signatures that can be detected from DNA sequence alone.

---

Notes: Do not use python script execution as part of the script or use it anywhere else. It saves me time to simply copy and paste code that should have been grouped in a cell. Simply give me a clear breakline so that I know to stop and paste into the next cell. Simply write a python script that in the middle, use comments telling to stop and paste into the next cell, and I will know.

## Rationale

### Why This Approach Can Work

The competition uses an 18-state ChromHMM model, which is almost certainly the **Roadmap Epigenomics 18-state model** (Kundaje et al., Nature 2015). This model has been extensively studied, and each state has documented sequence characteristics:

```
State  | Name        | Key Sequence Features
-------|-------------|--------------------------------------------------
1      | TssA        | CpG islands (>60%), TATA box, GC-box, Inr
2      | TssFlnk     | High GC (~55%), adjacent to CpG islands
3      | TssFlnkU    | Upstream promoter signatures
4      | TssFlnkD    | Transition into gene body
5      | Tx          | Gene body composition, splice signals
6      | TxWk        | Similar to Tx, weaker
7      | EnhG1       | AP-1/ETS motifs, genic context
8      | EnhG2       | Similar to EnhG1
9      | EnhA1       | Strong AP-1, ETS, GATA motifs
10     | EnhA2       | Similar TF binding sites
11     | EnhWk       | Weaker enhancer signatures
12     | ZnfRpts     | KRAB-ZNF motifs, high repeat content
13     | Het         | AT-rich (>65%), satellite repeats, low complexity
14     | TssBiv      | CpG islands + developmental gene context
15     | EnhBiv      | Poised enhancer signatures
16     | ReprPC      | CpG-rich, Polycomb targets
17     | ReprPCWk    | Similar but weaker
18     | Quies       | Background (~40% GC, low information)
```

### Expected Discriminative Power

Based on literature, we can expect clear separation for:

1. **High confidence** (distinct signatures):
   - TssA vs Het (GC-rich vs AT-rich: ~65% vs ~35%)
   - Quies vs TssA (background vs promoter)
   - ZnfRpts (unique repeat signature)

2. **Medium confidence** (overlapping but distinguishable):
   - TssA vs TssFlnk (gradient of CpG/GC)
   - Enh states (specific TF motifs)
   - Tx vs TxWk (intensity differences)

3. **Lower confidence** (may be confused):
   - EnhA1 vs EnhA2
   - TssFlnkU vs TssFlnkD
   - ReprPC vs TssBiv (both CpG-rich)

---

## Implementation Plan

### Phase 1: Feature Extraction (2-3 hours)

**Key features to compute for each sequence:**

```python
# Core compositional features
- gc_content: (G+C) / length
- cpg_ratio: CpG observed/expected (>0.6 = CpG island)
- cpg_frequency: Raw CpG dinucleotide count
- at_content: (A+T) / length

# Repeat/complexity features
- repeat_density: Fraction of sequence in simple repeats
- homopolymer_runs: Count of runs ≥4bp (poly-A, poly-T)

# Motif features
- TATA_box_count: Matches to TATA[AT]A[AT]
- GC_box_count: Matches to GGGCGG
- AP1_motif_count: Matches to TGA[CG]TCA
- ETS_motif_count: Matches to [AC]GGA[AT]G
- CTCF_motif_count: Matches to CTCF consensus
```

### Phase 2: Profile Computation (1 hour)

For each label (1-18), compute:
- Mean and standard deviation of each feature
- Create a feature profile vector

### Phase 3: State Matching (1 hour)

Compare label profiles against expected state profiles:
1. Normalize features to z-scores
2. Compute similarity (e.g., negative Euclidean distance)
3. Use Hungarian algorithm for optimal assignment

### Phase 4: Validation (2 hours)

1. **Cross-validation**: Split data, compute profiles on subset, validate on remainder
2. **Biological sanity checks**:
   - TssA should have highest CpG ratio
   - Het should have lowest GC content
   - ZnfRpts should have highest repeat density

---

## Expected Outputs

### 1. Label Mapping File (`label_state_mapping.json`)
```json
{
  "1": "TssA",
  "2": "Het",
  "3": "EnhA1",
  ...
}
```

### 2. Confidence Matrix
- 18×18 matrix showing similarity between each label and each state
- Identifies ambiguous assignments

### 3. Feature Profile Table
- Detailed per-label feature statistics for future reference

---

## How to Use the Identified Labels

### Strategy 1: Class-Weighted Training
If we know label 13 = Het (AT-rich heterochromatin), we can:
- Apply specific augmentation strategies for that class
- Adjust loss weights based on class difficulty

### Strategy 2: Hierarchical Classification
Group states into super-classes based on biology:
```
Promoter-related: TssA, TssFlnk, TssFlnkU, TssFlnkD, TssBiv
Enhancer-related: EnhG1, EnhG2, EnhA1, EnhA2, EnhWk, EnhBiv
Transcribed: Tx, TxWk
Repressed: Het, ReprPC, ReprPCWk
Other: ZnfRpts, Quies
```
Train a hierarchical model: super-class → specific class

### Strategy 3: Biological Feature Integration
Once we know the mapping, we can:
- Add explicit biological features (CpG island score, etc.)
- Use motif scanning as input features
- Apply state-specific preprocessing

### Strategy 4: Error Analysis
Knowing the biological meaning helps interpret:
- Why certain labels are confused (e.g., TssFlnkU vs TssFlnkD - spatially adjacent)
- Which classes need more training data or specific architectures

---

## Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Labels are scrambled | Low | Biological patterns persist regardless of label number |
| Different cell type with different patterns | Medium | Core state signatures are conserved across cell types |
| Non-standard 18-state model | Low | All published 18-state models have similar structure |
| Sequence features insufficient | Medium | Use multiple features, validate with UMAP clusters |

---

## Next Steps

1. **Run label_identification_plan.py** with your training data
2. **Validate** the mapping with UMAP visualization (color by identified states)
3. **Integrate** into model training:
   - Use state names in logging/visualization
   - Apply hierarchical classification
   - Add biological feature channels

---

## Code Usage

```bash
# Run identification pipeline
python label_identification_plan.py \
    --sequences data/trainsequences.csv \
    --labels data/trainlabels.csv \
    --output label_identification_output

# Outputs:
# - label_identification_output/label_state_mapping.json
# - label_identification_output/identification_report.txt
# - label_identification_output/label_feature_profiles.csv
# - label_identification_output/similarity_matrix.npy
```

---

## Coding & Execution Guidelines for Colab (Instructions for Generation)

When generating code or scripts for this phase, strictly adhere to the following guidelines to ensure compatibility with the Google Colab environment and existing project structure.

### 1. Drive Mounting & Directory Structure
- **Always** include a safe Drive mounting block at the beginning of the script.
- **Use `try-except`** to handle local vs. Colab execution seamlessly.
- **Base Directory**: `/content/drive/MyDrive` for Colab, or `.` (current directory) for local.

```python
try:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DIR = '/content/drive/MyDrive'
except ImportError:
    BASE_DIR = '.'  # Fallback for local execution
    print("Drive not mounted, using local directory.")
```

- **Data Paths**:
  - Sequences: `f'{BASE_DIR}/chromatin_data/train_sequences.csv'`
  - Labels: `f'{BASE_DIR}/chromatin_data/train_labels.csv'`
  - Validation/Test paths follow the same pattern in `chromatin_data/`.

- **Checkpoint/Output Paths**:
  - Create a dedicated folder for this phase: `f'{BASE_DIR}/chromatin_phase8'`
  - Ensure directories exist using `os.makedirs(..., exist_ok=True)`.

### 2. Coding Style & Libraries
- **Format**: Produce **Python scripts (`.py`)** that can be pasted into a Colab cell. Do not assume Jupyter Notebook magic commands (`%`) unless necessary (like `%pip install`).
- **Imports**: Standardize imports.
  - `import torch`
  - `import torch.nn as nn`
  - `import torch.nn.functional as F`
  - `import numpy as np`
  - `import pandas as pd`
  - `import matplotlib.pyplot as plt`
  - `from tqdm.auto import tqdm` (use `tqdm.auto` for notebook compatibility)
- **Device Handling**: Always define `DEVICE` early and use it consistently.
  ```python
  DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
  ```
- **Structure**:
  - **No `if __name__ == "__main__":` blocks** that wrap the *entire* script if it's meant to be run cell-by-cell or as a single script in Colab. Instead, define functions and call a `main()` function at the end.
  - **Self-Contained**: The script should be standalone. If it relies on a custom model class, **re-define the class** within the script or ensure the import path is correctly set up if using a package structure (though standalone is safer for Colab).

### 3. Training & Validation Loop Standards
- **Progress Bars**: Use `tqdm` for loops.
- **Metrics**: Track `accuracy`, `loss`, and `f1-score` (if applicable).
- **Checkpointing**:
  - Save the **best model** based on validation accuracy.
  - Save `last_model.pt` for resumption.
  - Save `config` dict inside the checkpoint for reproducibility.
- **Resumption**: Include logic to check if a checkpoint exists and resume training automatically if desired (or via a flag).

### 4. Specific Phase 8 Requirements
- **Hybrid Model**: Ensure the model class (`HybridChromatinModel` or similar) is explicitly defined in the script.
- **K-mer Generation**: Include the K-mer counting functions directly in the script (no external dependency if possible).
- **Loss Function**: Explicitly define `FocalLoss` class.

### 5. Reproducibility
- Set random seeds at the start:
  ```python
  def set_seed(seed=42):
      torch.manual_seed(seed)
      np.random.seed(seed)
      random.seed(seed)
      if torch.cuda.is_available():
          torch.cuda.manual_seed_all(seed)
  set_seed(42)
  ```

### 6. Example Usage Comment
At the top of the generated script, include a docstring with example usage or expected input files:
```python
"""
Phase 8 Training Script
Usage:
1. Upload data to Drive: /chromatin_data/train_sequences.csv, etc.
2. Run this script in Colab.
3. Checkpoints saved to: /chromatin_model_checkpoints_phase8/
"""
```

---

## References

1. Kundaje et al. "Integrative analysis of 111 reference human epigenomes" Nature 2015
2. Ernst & Kellis "ChromHMM: automating chromatin-state discovery" Nature Methods 2012
3. ENCODE Consortium chromatin state annotations

---

## Appendix: Quick Test

You can quickly validate the approach by checking if:

```python
# Expected: Label with highest mean CpG ratio should be TssA or TssBiv
# Expected: Label with lowest mean GC content should be Het
# Expected: Label with highest repeat density should be ZnfRpts or Het

# Run these checks to validate the identification is working
for label in range(1, 19):
    print(f"Label {label}: GC={gc[label]:.3f}, CpG_ratio={cpg[label]:.3f}")
```

If the highest GC/CpG label maps to something other than TssA/TssBiv, or the lowest GC label maps to something other than Het, the identification may need refinement.
