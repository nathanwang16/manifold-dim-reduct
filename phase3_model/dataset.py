"""
Data loaders for ChromatinCNN with biological augmentations.

Implements:
- One-hot encoding of DNA sequences
- Reverse complement augmentation
- Position jittering
- Noise injection
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict
import json

from logger import get_logger

logger = get_logger(__name__)


def reverse_complement(sequence: str) -> str:
    """
    Compute reverse complement of a DNA sequence.

    Args:
        sequence: DNA sequence (A, C, G, T)

    Returns:
        Reverse complement sequence
    """
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join([complement[base] for base in reversed(sequence)])


def one_hot_encode(sequence: str) -> np.ndarray:
    """
    Convert DNA sequence to one-hot encoded array.

    Args:
        sequence: DNA sequence (A, C, G, T)

    Returns:
        One-hot encoded array of shape (200, 4)
        Channels: A=0, C=1, G=2, T=3
    """
    # Base to index mapping
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    # Initialize array
    one_hot = np.zeros((200, 4), dtype=np.float32)

    for i, base in enumerate(sequence):
        if base in base_to_idx:
            one_hot[i, base_to_idx[base]] = 1.0
        else:
            # Handle unexpected characters (set to uniform)
            one_hot[i, :] = 0.25

    return one_hot


class ChromatinDataset(Dataset):
    """
    Dataset for chromatin state prediction with biological augmentations.

    Supports:
    - Reverse complement augmentation (on-the-fly)
    - Position jittering
    - Noise injection
    """

    def __init__(
        self,
        sequences_file: str,
        labels_file: Optional[str] = None,
        rc_augment: bool = False,
        jitter_prob: float = 0.3,
        jitter_min_len: int = 180,
        noise_prob: float = 0.01,
        sequence_length: int = 200,
        cache_data: bool = True,
        pin_memory: bool = False,  # Default to False for MPS to avoid file descriptor limits
    ):
        """
        Initialize ChromatinDataset.

        Args:
            sequences_file: Path to sequences CSV
            labels_file: Path to labels CSV (None for test data)
            rc_augment: Apply reverse complement augmentation
            jitter_prob: Probability of position jittering
            jitter_min_len: Minimum length for jittered crop
            noise_prob: Probability of noise injection per base
            sequence_length: Expected sequence length
            cache_data: Cache one-hot encoded sequences in memory
            pin_memory: Whether to use pinned memory for faster GPU/MPS transfers
        """
        self.sequences_file = Path(sequences_file)
        self.labels_file = Path(labels_file) if labels_file else None
        self.rc_augment = rc_augment
        self.jitter_prob = jitter_prob
        self.jitter_min_len = jitter_min_len
        self.noise_prob = noise_prob
        self.sequence_length = sequence_length
        self.cache_data = cache_data
        self.pin_memory = pin_memory

        # Load sequences
        logger.info(f"Loading sequences from {self.sequences_file}")
        self.sequences_df = pd.read_csv(self.sequences_file, header=None)

        # Handle different CSV formats
        if self.sequences_df.shape[1] == 1:
            # Format: single column with sequences
            self.sequences = self.sequences_df[0].values
        else:
            # Format: first column is ID, second is sequence
            self.sequences = self.sequences_df.iloc[:, 1].values

        logger.info(f"Loaded {len(self.sequences)} sequences")

        # Load labels if provided
        self.labels = None
        if self.labels_file is not None:
            logger.info(f"Loading labels from {self.labels_file}")
            labels_df = pd.read_csv(self.labels_file, header=None)

            if labels_df.shape[1] == 1:
                self.labels = labels_df[0].values
            else:
                self.labels = labels_df.iloc[:, 1].values

            # Convert labels from 1-18 to 0-17 for PyTorch
            self.labels = self.labels - 1
            logger.info(f"Loaded {len(self.labels)} labels (converted from 1-18 to 0-17)")

        # Cache one-hot encodings
        self._cached_sequences = None
        if cache_data:
            logger.info("Caching one-hot encodings in memory...")
            self._cached_sequences = [
                one_hot_encode(seq) for seq in self.sequences
            ]
            logger.info("Cache complete")

        self._validate_sequences()

    def _validate_sequences(self):
        """Validate sequence lengths."""
        for i, seq in enumerate(self.sequences):
            if len(seq) != self.sequence_length:
                logger.warning(f"Sequence {i} has length {len(seq)}, expected {self.sequence_length}")

    def __len__(self) -> int:
        """Return number of sequences in dataset."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get a single item from the dataset.

        Returns:
            Tuple of (sequence_tensor, label_tensor)
            label_tensor is None for test data
        """
        sequence = self.sequences[idx]

        # Get one-hot encoding
        if self._cached_sequences is not None:
            one_hot = self._cached_sequences[idx].copy()
        else:
            one_hot = one_hot_encode(sequence)

        # Apply augmentations
        one_hot = self._apply_augmentations(one_hot)

        # Convert to tensor
        sequence_tensor = torch.from_numpy(one_hot)

        # Get label if available
        label_tensor = None
        if self.labels is not None:
            label = self.labels[idx]
            label_tensor = torch.tensor(label, dtype=torch.long)
        else:
            # Return dummy label for test data (will be ignored)
            label_tensor = torch.tensor(0, dtype=torch.long)

        return sequence_tensor, label_tensor

    def _apply_augmentations(self, one_hot: np.ndarray) -> np.ndarray:
        """Apply biological augmentations to one-hot encoded sequence."""
        augmented = one_hot.copy()

        # Reverse complement augmentation
        if self.rc_augment and np.random.random() < 0.5:
            augmented = augmented[::-1, [3, 2, 1, 0]]  # Reverse and swap A<->T, C<->G

        # Position jittering
        if np.random.random() < self.jitter_prob:
            crop_length = np.random.randint(self.jitter_min_len, self.sequence_length + 1)
            start_pos = np.random.randint(0, self.sequence_length - crop_length + 1)
            end_pos = start_pos + crop_length

            # Crop and pad
            cropped = augmented[start_pos:end_pos, :]
            padded = np.zeros((self.sequence_length, 4), dtype=np.float32)

            # Pad with random bases (simulating flanking sequence)
            pad_before = start_pos
            if pad_before > 0:
                padded[:pad_before, :] = 0.25  # Uniform distribution

            padded[pad_before:pad_before + crop_length, :] = cropped
            pad_after = self.sequence_length - pad_before - crop_length
            if pad_after > 0:
                padded[pad_before + crop_length:, :] = 0.25

            augmented = padded

        # Noise injection
        if np.random.random() < self.noise_prob:
            # Randomly mutate a base
            pos = np.random.randint(0, self.sequence_length)
            new_base = np.random.randint(0, 4)
            augmented[pos, :] = 0.0
            augmented[pos, new_base] = 1.0

        return augmented


class ChromatinDataModule:
    """
    Data module for managing train/val/test splits and dataloaders.
    """

    def __init__(
        self,
        train_sequences: str,
        train_labels: str,
        val_sequences: str,
        val_labels: str,
        test_sequences: str,
        batch_size: int = 256,
        num_workers: int = 4,
        rc_augment: bool = True,
        jitter_prob: float = 0.3,
        noise_prob: float = 0.01,
        sequence_length: int = 200,
        cache_data: bool = True,
        pin_memory: bool = True,
    ):
        """
        Initialize data module.

        Args:
            train_sequences: Path to training sequences CSV
            train_labels: Path to training labels CSV
            val_sequences: Path to validation sequences CSV
            val_labels: Path to validation labels CSV
            test_sequences: Path to test sequences CSV
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes for dataloaders
            rc_augment: Apply reverse complement augmentation during training
            jitter_prob: Probability of position jittering
            noise_prob: Probability of noise injection
            sequence_length: Expected sequence length
            cache_data: Cache one-hot encodings in memory
            pin_memory: Whether to use pinned memory (True for GPU/MPS, False for CPU)
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        logger.info("Initializing datasets...")

        # Training dataset with augmentations
        self.train_dataset = ChromatinDataset(
            sequences_file=train_sequences,
            labels_file=train_labels,
            rc_augment=rc_augment,
            jitter_prob=jitter_prob,
            noise_prob=noise_prob,
            sequence_length=sequence_length,
            cache_data=cache_data,
            pin_memory=pin_memory,
        )

        # Validation dataset (no augmentations)
        self.val_dataset = ChromatinDataset(
            sequences_file=val_sequences,
            labels_file=val_labels,
            rc_augment=False,
            jitter_prob=0.0,
            noise_prob=0.0,
            sequence_length=sequence_length,
            cache_data=cache_data,
            pin_memory=pin_memory,
        )

        # Test dataset (no labels, no augmentations)
        self.test_dataset = ChromatinDataset(
            sequences_file=test_sequences,
            labels_file=None,
            rc_augment=False,
            jitter_prob=0.0,
            noise_prob=0.0,
            sequence_length=sequence_length,
            cache_data=cache_data,
            pin_memory=pin_memory,
        )

        logger.info("Datasets initialized")

        # Initialize dataloaders once to reuse across epochs (prevents file descriptor leak)
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def get_train_dataloader(self) -> DataLoader:
        """Get training dataloader with shuffling. Creates once and reuses."""
        if self.train_loader is None:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=True,
                persistent_workers=False,  # Explicitly set to prevent leaks
            )
        return self.train_loader

    def get_val_dataloader(self) -> DataLoader:
        """Get validation dataloader without shuffling. Creates once and reuses."""
        if self.val_loader is None:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=False,  # Explicitly set to prevent leaks
            )
        return self.val_loader

    def get_test_dataloader(self) -> DataLoader:
        """Get test dataloader without shuffling. Creates once and reuses."""
        if self.test_loader is None:
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=False,  # Explicitly set to prevent leaks
            )
        return self.test_loader

    def cleanup(self):
        """Clean up dataloader worker processes to free file descriptors."""
        if self.train_loader is not None:
            # Trigger cleanup of worker processes
            del self.train_loader
            self.train_loader = None
        if self.val_loader is not None:
            del self.val_loader
            self.val_loader = None
        if self.test_loader is not None:
            del self.test_loader
            self.test_loader = None

    def get_dataset_sizes(self) -> Dict[str, int]:
        """Return sizes of train/val/test datasets."""
        return {
            'train': len(self.train_dataset),
            'val': len(self.val_dataset),
            'test': len(self.test_dataset),
        }

