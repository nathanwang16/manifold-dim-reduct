"""
Phase 2.1: Feature Extraction for Manifold Visualization

Extracts three types of features from DNA sequences:
1. K-mer Frequency Vectors (5-mer or 6-mer)
2. Positional K-mer Profiles (binned k-mer frequencies)
3. Dinucleotide Transition Frequencies

Usage:
    python feature_extraction.py --input trainsequences.csv --labels trainlabels.csv
"""

import sys
import json
import argparse
from pathlib import Path
from itertools import product
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count

# Add parent directory to path for logger import
sys.path.insert(0, str(Path(__file__).parent.parent))
from logger import get_logger, LogTimer, log_metrics, configure_logging

# Initialize logger
logger = get_logger(__name__)


def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def generate_kmer_vocabulary(k: int) -> dict:
    """Generate all possible k-mers and their indices."""
    bases = ['A', 'C', 'G', 'T']
    kmers = [''.join(p) for p in product(bases, repeat=k)]
    return {kmer: idx for idx, kmer in enumerate(kmers)}


def compute_kmer_frequencies(sequence: str, k: int, vocab: dict) -> np.ndarray:
    """
    Compute normalized k-mer frequency vector for a single sequence.

    Args:
        sequence: DNA sequence string
        k: k-mer length
        vocab: dictionary mapping k-mer to index

    Returns:
        Normalized frequency vector of shape (4^k,)
    """
    counts = np.zeros(len(vocab), dtype=np.float32)
    n_kmers = len(sequence) - k + 1

    if n_kmers <= 0:
        return counts

    for i in range(n_kmers):
        kmer = sequence[i:i+k]
        if kmer in vocab:  # Skip k-mers with N or other ambiguous bases
            counts[vocab[kmer]] += 1

    # Normalize to relative frequencies
    total = counts.sum()
    if total > 0:
        counts /= total

    return counts


def compute_positional_kmer_profiles(
    sequence: str,
    k: int,
    vocab: dict,
    n_bins: int = 10
) -> np.ndarray:
    """
    Compute position-aware k-mer frequencies by binning the sequence.

    Args:
        sequence: DNA sequence string
        k: k-mer length
        vocab: dictionary mapping k-mer to index
        n_bins: number of positional bins

    Returns:
        Feature vector of shape (n_bins * 4^k,)
    """
    seq_len = len(sequence)
    bin_size = seq_len // n_bins
    vocab_size = len(vocab)

    profiles = np.zeros((n_bins, vocab_size), dtype=np.float32)

    for bin_idx in range(n_bins):
        start = bin_idx * bin_size
        end = start + bin_size if bin_idx < n_bins - 1 else seq_len
        bin_seq = sequence[start:end]
        profiles[bin_idx] = compute_kmer_frequencies(bin_seq, k, vocab)

    return profiles.flatten()


def compute_dinucleotide_frequencies(sequence: str) -> np.ndarray:
    """
    Compute all 16 dinucleotide transition frequencies.

    Args:
        sequence: DNA sequence string

    Returns:
        Frequency vector of shape (16,)
    """
    dinuc_vocab = generate_kmer_vocabulary(2)
    return compute_kmer_frequencies(sequence, 2, dinuc_vocab)


def process_single_sequence(
    seq: str,
    k: int,
    vocab: dict,
    n_bins: int
) -> tuple:
    """
    Process a single sequence and return its features.
    
    Args:
        seq: DNA sequence string
        k: k-mer length
        vocab: k-mer vocabulary dictionary
        n_bins: number of positional bins
    
    Returns:
        Tuple of (kmer_features, positional_features, dinuc_features, is_valid)
    """
    if not seq or len(seq) < k:
        # Return zero features for invalid sequences
        vocab_size = len(vocab)
        dinuc_vocab_size = 16  # 4^2 dinucleotides
        return (
            np.zeros(vocab_size, dtype=np.float32),
            np.zeros(n_bins * vocab_size, dtype=np.float32),
            np.zeros(dinuc_vocab_size, dtype=np.float32),
            False
        )
    
    kmer_feat = compute_kmer_frequencies(seq, k, vocab)
    positional_feat = compute_positional_kmer_profiles(seq, k, vocab, n_bins)
    dinuc_feat = compute_dinucleotide_frequencies(seq)
    
    return (kmer_feat, positional_feat, dinuc_feat, True)


def extract_all_features(
    sequences: list,
    config: dict,
    output_dir: Path,
    n_jobs: int = -1,
    batch_size: int = 1000
) -> dict:
    """
    Extract all feature types from sequences and save to disk using parallel processing.

    Args:
        sequences: list of DNA sequence strings
        config: configuration dictionary
        output_dir: directory to save feature files
        n_jobs: number of parallel jobs (-1 = all cores)
        batch_size: number of sequences to process per worker

    Returns:
        Dictionary with feature arrays
    """
    k = config['phase2']['kmer_k']
    n_bins = config['phase2']['n_positional_bins']

    with LogTimer(logger, f"Generating {k}-mer vocabulary"):
        vocab = generate_kmer_vocabulary(k)
        vocab_size = len(vocab)
        logger.info(f"Vocabulary size: {vocab_size}")

    n_samples = len(sequences)
    logger.info(f"Processing {n_samples} sequences with n_jobs={n_jobs}, batch_size={batch_size}")

    # Pre-allocate arrays
    kmer_features = np.zeros((n_samples, vocab_size), dtype=np.float32)
    positional_features = np.zeros((n_samples, n_bins * vocab_size), dtype=np.float32)
    dinuc_features = np.zeros((n_samples, 16), dtype=np.float32)

    # Process sequences in parallel batches
    with LogTimer(logger, "Feature extraction (parallel)"):
        # Create partial function with fixed parameters
        process_func = partial(process_single_sequence, k=k, vocab=vocab, n_bins=n_bins)
        
        # Process in parallel
        results = Parallel(
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=1 if logger.level <= 20 else 0  # Show progress if INFO or DEBUG
        )(
            delayed(process_func)(seq) for seq in sequences
        )
        
        # Assign results to arrays
        skipped_sequences = 0
        for i, (kmer_feat, positional_feat, dinuc_feat, is_valid) in enumerate(results):
            if not is_valid:
                skipped_sequences += 1
            else:
                kmer_features[i] = kmer_feat
                positional_features[i] = positional_feat
                dinuc_features[i] = dinuc_feat

    if skipped_sequences > 0:
        logger.warning(f"Skipped {skipped_sequences} invalid sequences")

    # Save features
    output_dir.mkdir(parents=True, exist_ok=True)

    with LogTimer(logger, "Saving features to disk"):
        np.save(output_dir / f"kmer_{k}_features.npy", kmer_features)
        np.save(output_dir / "positional_kmer_features.npy", positional_features)
        np.save(output_dir / "dinucleotide_features.npy", dinuc_features)

        # Save vocabulary for reference
        with open(output_dir / f"kmer_{k}_vocab.json", 'w') as f:
            json.dump(vocab, f)

    # Log metrics
    log_metrics(logger, {
        "n_samples": n_samples,
        "kmer_k": k,
        "kmer_features_shape": list(kmer_features.shape),
        "positional_features_shape": list(positional_features.shape),
        "dinuc_features_shape": list(dinuc_features.shape),
        "skipped_sequences": skipped_sequences
    }, message="Feature extraction metrics")

    logger.info(f"K-mer features shape: {kmer_features.shape}")
    logger.info(f"Positional features shape: {positional_features.shape}")
    logger.info(f"Dinucleotide features shape: {dinuc_features.shape}")

    return {
        'kmer': kmer_features,
        'positional': positional_features,
        'dinucleotide': dinuc_features,
        'vocab': vocab
    }


def main():
    parser = argparse.ArgumentParser(description="Extract features for manifold learning")
    parser.add_argument("--input", type=str, required=True, help="Path to sequences CSV")
    parser.add_argument("--labels", type=str, default=None, help="Path to labels CSV")
    parser.add_argument("--config", type=str, default="config.json", help="Config file path")
    parser.add_argument("--output", type=str, default="phase2_manifold/features", help="Output directory")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel jobs (-1 = all cores)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for parallel processing")
    args = parser.parse_args()

    # Configure logging
    configure_logging(log_dir=args.log_dir)

    config = load_config(args.config)
    output_dir = Path(args.output)

    # Determine actual n_jobs
    if args.n_jobs == -1:
        actual_n_jobs = cpu_count()
        logger.info(f"Using all {actual_n_jobs} CPU cores for parallel feature extraction")
    else:
        actual_n_jobs = args.n_jobs
        logger.info(f"Using {actual_n_jobs} CPU cores for parallel feature extraction")

    # Load sequences
    with LogTimer(logger, f"Loading sequences from {args.input}"):
        seq_df = pd.read_csv(args.input, header=None, names=['sequence'])
        sequences = seq_df['sequence'].tolist()
        logger.info(f"Loaded {len(sequences)} sequences")

    # Load labels if provided
    if args.labels:
        with LogTimer(logger, f"Loading labels from {args.labels}"):
            labels_df = pd.read_csv(args.labels, header=None, names=['label'])
            labels = labels_df['label'].values
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save(output_dir / "labels.npy", labels)
            logger.info(f"Labels range: {labels.min()} to {labels.max()}")
            log_metrics(logger, {
                "n_labels": len(labels),
                "unique_labels": len(np.unique(labels)),
                "label_min": int(labels.min()),
                "label_max": int(labels.max())
            }, message="Label statistics")

    # Extract features with parallelization
    features = extract_all_features(sequences, config, output_dir, n_jobs=actual_n_jobs, batch_size=args.batch_size)

    logger.info("Feature extraction complete!")


if __name__ == "__main__":
    main()
