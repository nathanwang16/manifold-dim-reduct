"""
Phase 1: Data Engineering & Stratified Splitting

Implements stratified 5-fold cross-validation framework according to guide.md:
- Fold 5: Held-out "interpretability test set" (never used for training)
- Folds 1-3: Training set
- Fold 4: Validation set

All splits preserve the 1/18 class balance across all folds.
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os
from pathlib import Path

# Add parent directory to path to import logger
import sys
sys.path.append(str(Path(__file__).parent.parent))
from logger.logger import setup_logger

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Setup logger
logger = setup_logger('phase1_filter')
logger.info("=" * 70)
logger.info("PHASE 1: Data Engineering & Stratified Splitting")
logger.info("=" * 70)


def load_data():
    """Load training sequences and labels from CSV files."""
    logger.info("Loading data...")
    
    sequences_path = config['data']['train_sequences']
    labels_path = config['data']['train_labels']
    
    # Load sequences
    logger.info(f"Loading sequences from {sequences_path}")
    sequences_df = pd.read_csv(sequences_path, header=None, names=['sequence'])
    sequences = sequences_df['sequence'].values
    logger.info(f"Loaded {len(sequences)} sequences")
    
    # Load labels (assuming integer labels 1-18)
    logger.info(f"Loading labels from {labels_path}")
    labels_df = pd.read_csv(labels_path, header=None, names=['label'])
    labels = labels_df['label'].values
    logger.info(f"Loaded {len(labels)} labels")
    
    # Verify data integrity
    assert len(sequences) == len(labels), "Mismatch between sequences and labels count"
    
    # Check label distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    logger.info(f"Unique labels: {unique_labels}")
    logger.info(f"Label distribution: {dict(zip(unique_labels, counts))}")
    
    # Verify balanced classes
    n_classes = config['data']['n_classes']
    expected_count = len(labels) // n_classes
    tolerance = expected_count * 0.01  # 1% tolerance
    
    for label, count in zip(unique_labels, counts):
        if abs(count - expected_count) > tolerance:
            logger.warning(f"Label {label} has {count} samples (expected ~{expected_count})")
    
    logger.info(f"Data balance verified: {len(unique_labels)} classes, ~{expected_count} samples each")
    
    return sequences, labels


def perform_stratified_split(sequences, labels, n_splits=5, random_state=42):
    """
    Perform stratified k-fold split preserving class balance.
    
    Args:
        sequences: Array of DNA sequences
        labels: Array of integer labels (1-18)
        n_splits: Number of folds (default 5 per guide.md)
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary mapping fold number to indices
    """
    logger.info("\n" + "=" * 70)
    logger.info(f"Performing {n_splits}-fold stratified split...")
    logger.info("=" * 70)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_indices = {}
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(sequences, labels), 1):
        fold_indices[fold_idx] = {
            'train': train_idx,  # Indices for this fold's training set
            'test': test_idx     # Indices for this fold's test/validation set
        }
        
        # Check class balance in this fold's test set
        test_labels = labels[test_idx]
        unique, counts = np.unique(test_labels, return_counts=True)
        logger.info(f"Fold {fold_idx} test set: {len(test_idx)} samples, "
                   f"labels range {unique[0]}-{unique[-1]}, "
                   f"counts per class: min={counts.min()}, max={counts.max()}")
    
    return fold_indices


def create_final_splits(fold_indices, labels):
    """
    Create final train/val/MI splits according to guide.md framework:
    - MI (Mechanistic Interpretability): Fold 5 test indices
    - Validation: Fold 4 test indices
    - Training: Folds 1, 2, 3 test indices (since we use all non-test data as train)
    
    Note: In 5-fold CV, each fold's "test" set becomes a candidate for val/MI.
    The actual training set for each fold is the remaining 4 folds' combined data.
    """
    logger.info("\n" + "=" * 70)
    logger.info("Creating final train/val/MI splits...")
    logger.info("=" * 70)
    
    # MI test set: Fold 5's test indices
    mi_indices = fold_indices[5]['test']
    
    # Validation set: Fold 4's test indices
    val_indices = fold_indices[4]['test']
    
    # Training set: All indices not in MI or val
    # This is folds 1, 2, 3 test indices + their corresponding train indices
    # Effectively: all data except fold 4 and fold 5 test sets
    all_indices = np.arange(len(labels))
    excluded_indices = np.concatenate([mi_indices, val_indices])
    train_indices = np.setdiff1d(all_indices, excluded_indices)
    
    logger.info(f"MI (Fold 5): {len(mi_indices)} samples")
    logger.info(f"Validation (Fold 4): {len(val_indices)} samples")
    logger.info(f"Training: {len(train_indices)} samples")
    logger.info(f"Total: {len(train_indices) + len(val_indices) + len(mi_indices)} samples")
    
    # Verify no overlap
    assert len(np.intersect1d(train_indices, val_indices)) == 0, "Train and val overlap!"
    assert len(np.intersect1d(train_indices, mi_indices)) == 0, "Train and MI overlap!"
    assert len(np.intersect1d(val_indices, mi_indices)) == 0, "Val and MI overlap!"
    
    # Verify class balance in each split
    logger.info("\nClass balance check:")
    for split_name, indices in [("Train", train_indices), ("Val", val_indices), ("MI", mi_indices)]:
        split_labels = labels[indices]
        unique, counts = np.unique(split_labels, return_counts=True)
        logger.info(f"  {split_name}: {len(indices)} samples, "
                   f"classes {len(unique)}, "
                   f"counts min={counts.min()}, max={counts.max()}")
    
    return {
        'train': train_indices,
        'val': val_indices,
        'mi': mi_indices
    }


def save_splits(sequences, labels, splits, output_dir='data/splits'):
    """
    Save the splits to CSV files for use in subsequent phases.
    
    Args:
        sequences: Array of DNA sequences
        labels: Array of integer labels
        splits: Dictionary with 'train', 'val', 'mi' indices
        output_dir: Directory to save split files
    """
    logger.info("\n" + "=" * 70)
    logger.info(f"Saving splits to {output_dir}...")
    logger.info("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each split
    for split_name in ['train', 'val', 'mi']:
        indices = splits[split_name]
        
        # Save sequences
        sequences_split = sequences[indices]
        sequences_df = pd.DataFrame(sequences_split, columns=['sequence'])
        sequences_path = os.path.join(output_dir, f'{split_name}_sequences.csv')
        sequences_df.to_csv(sequences_path, index=False, header=False)
        logger.info(f"  Saved {len(sequences_split)} sequences to {sequences_path}")
        
        # Save labels
        labels_split = labels[indices]
        labels_df = pd.DataFrame(labels_split, columns=['label'])
        labels_path = os.path.join(output_dir, f'{split_name}_labels.csv')
        labels_df.to_csv(labels_path, index=False, header=False)
        logger.info(f"  Saved {len(labels_split)} labels to {labels_path}")
    
    # Save split indices as numpy arrays for easy loading
    indices_path = os.path.join(output_dir, 'split_indices.npz')
    np.savez(indices_path, **splits)
    logger.info(f"  Saved split indices to {indices_path}")
    
    logger.info("\nAll splits saved successfully!")


def main():
    """Main execution function."""
    try:
        # Load data
        sequences, labels = load_data()
        
        # Perform stratified 5-fold split
        fold_indices = perform_stratified_split(
            sequences, 
            labels,
            n_splits=5,
            random_state=config['phase2'].get('random_state', 42)
        )
        
        # Create final train/val/MI splits
        splits = create_final_splits(fold_indices, labels)
        
        # Save splits
        save_splits(sequences, labels, splits, output_dir='data')
        
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 1 COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info("\nSummary:")
        logger.info(f"  Train sequences: data/splits/train_sequences.csv")
        logger.info(f"  Train labels: data/splits/train_labels.csv")
        logger.info(f"  Val sequences: data/splits/val_sequences.csv")
        logger.info(f"  Val labels: data/splits/val_labels.csv")
        logger.info(f"  MI sequences: data/splits/mi_sequences.csv")
        logger.info(f"  MI labels: data/splits/mi_labels.csv")
        logger.info(f"  Indices: data/splits/split_indices.npz")
        
    except Exception as e:
        logger.error(f"Error in Phase 1: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

