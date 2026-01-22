"""
Simple Stratified Split: 4/5 Train, 1/5 Validation

Splits trainsequences.csv and trainlabels.csv into:
- 80% training set (4/5)
- 20% validation set (1/5)

Preserves class balance across splits (stratified by 18 labels).
Maintains correspondence between sequences and labels.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Setup paths
DATA_DIR = Path('/Users/xiaoyuwang/Desktop/manifold-dim-reduct/data')
SEQUENCES_PATH = DATA_DIR / 'trainsequences.csv'
LABELS_PATH = DATA_DIR / 'trainlabels.csv'

print("=" * 70)
print("SIMPLE STRATIFIED SPLIT: 4/5 TRAIN, 1/5 VAL")
print("=" * 70)

# Load data
print("\nLoading data...")
sequences_df = pd.read_csv(SEQUENCES_PATH, header=None, names=['sequence'])
labels_df = pd.read_csv(LABELS_PATH, header=None, names=['label'])

sequences = sequences_df['sequence'].values
labels = labels_df['label'].values

print(f"Loaded {len(sequences)} sequences")
print(f"Loaded {len(labels)} labels")

# Check class distribution
unique_labels, counts = np.unique(labels, return_counts=True)
print(f"\nClass distribution:")
for label, count in zip(unique_labels, counts):
    print(f"  Label {label}: {count} samples")

# Perform stratified split
print("\n" + "=" * 70)
print("Performing stratified split (80% train, 20% val)...")
print("=" * 70)

train_sequences, val_sequences, train_labels, val_labels = train_test_split(
    sequences,
    labels,
    test_size=0.2,  # 1/5 for validation
    stratify=labels,  # Ensure balanced split across classes
    random_state=42
)

print(f"\nSplit results:")
print(f"  Train: {len(train_sequences)} samples ({len(train_sequences)/len(sequences)*100:.1f}%)")
print(f"  Val: {len(val_sequences)} samples ({len(val_sequences)/len(sequences)*100:.1f}%)")

# Verify class balance in splits
print("\n" + "=" * 70)
print("Verifying class balance in splits...")
print("=" * 70)

for split_name, split_labels in [("Train", train_labels), ("Val", val_labels)]:
    unique, counts = np.unique(split_labels, return_counts=True)
    print(f"\n{split_name} set:")
    print(f"  Total: {len(split_labels)} samples")
    print(f"  Classes: {len(unique)} (should be 18)")
    print(f"  Samples per class: min={counts.min()}, max={counts.max()}, avg={counts.mean():.1f}")

# Save splits
print("\n" + "=" * 70)
print("Saving split files to data folder...")
print("=" * 70)

# Save train sequences
train_sequences_df = pd.DataFrame(train_sequences, columns=['sequence'])
train_sequences_output = DATA_DIR / 'train_sequences.csv'
train_sequences_df.to_csv(train_sequences_output, index=False, header=False)
print(f"Saved train_sequences.csv ({len(train_sequences)} rows)")

# Save train labels
train_labels_df = pd.DataFrame(train_labels, columns=['label'])
train_labels_output = DATA_DIR / 'train_labels.csv'
train_labels_df.to_csv(train_labels_output, index=False, header=False)
print(f"Saved train_labels.csv ({len(train_labels)} rows)")

# Save val sequences
val_sequences_df = pd.DataFrame(val_sequences, columns=['sequence'])
val_sequences_output = DATA_DIR / 'val_sequences.csv'
val_sequences_df.to_csv(val_sequences_output, index=False, header=False)
print(f"Saved val_sequences.csv ({len(val_sequences)} rows)")

# Save val labels
val_labels_df = pd.DataFrame(val_labels, columns=['label'])
val_labels_output = DATA_DIR / 'val_labels.csv'
val_labels_df.to_csv(val_labels_output, index=False, header=False)
print(f"Saved val_labels.csv ({len(val_labels)} rows)")

# Save split indices for reproducibility
train_indices, val_indices = train_test_split(
    np.arange(len(sequences)),
    test_size=0.2,
    stratify=labels,
    random_state=42
)

np.savez(DATA_DIR / 'split_indices.npz', train=train_indices, val=val_indices)
print(f"Saved split_indices.npz")

print("\n" + "=" * 70)
print("SPLIT COMPLETED SUCCESSFULLY")
print("=" * 70)
print("\nFiles created in data/:")
print("  - train_sequences.csv")
print("  - train_labels.csv")
print("  - val_sequences.csv")
print("  - val_labels.csv")
print("  - split_indices.npz")







