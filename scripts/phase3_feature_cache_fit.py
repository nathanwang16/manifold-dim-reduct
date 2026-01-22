import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from tqdm.auto import tqdm


def read_sequences(csv_path: Path) -> np.ndarray:
    """Read sequences from CSV (handles ID,sequence or sequence-only formats)."""
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] == 1:
        return df[0].values
    return df.iloc[:, 1].values


class FeatureExtractor:
    """Standalone feature extractor + SVD reducer for offline caching."""

    def __init__(self, n_components: int = 1024):
        self.n_components = n_components
        self.svd: Optional[TruncatedSVD] = None
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
        self.kmer5_to_idx = self._build_kmer_vocab(5)
        self.kmer4_to_idx = self._build_kmer_vocab(4)
        self.dinuc_to_idx = self._build_kmer_vocab(2)

    @staticmethod
    def _build_kmer_vocab(k: int) -> dict[str, int]:
        vocab = {}
        bases = ['A', 'C', 'G', 'T']

        def backtrack(current, depth):
            if depth == k:
                vocab[''.join(current)] = len(vocab)
                return
            for base in bases:
                current.append(base)
                backtrack(current, depth + 1)
                current.pop()

        backtrack([], 0)
        return vocab

    def _count_kmers(self, sequence: str, k: int, vocab: dict[str, int]) -> np.ndarray:
        counts = np.zeros(len(vocab), dtype=np.float32)
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i + k]
            idx = vocab.get(kmer)
            if idx is not None:
                counts[idx] += 1.0
        total = counts.sum()
        if total > 0:
            counts /= total
        return counts

    def _positional_kmers(self, sequence: str, n_bins: int = 10) -> np.ndarray:
        bin_size = max(len(sequence) // n_bins, 1)
        features = []
        for i in range(n_bins):
            start = i * bin_size
            end = start + bin_size if i < n_bins - 1 else len(sequence)
            bin_seq = sequence[start:end]
            features.append(self._count_kmers(bin_seq, 4, self.kmer4_to_idx))
        return np.concatenate(features)

    def _gc_content(self, sequence: str, n_windows: int = 10) -> np.ndarray:
        window_size = max(len(sequence) // n_windows, 1)
        gc_vals = []
        for i in range(n_windows):
            start = i * window_size
            end = start + window_size if i < n_windows - 1 else len(sequence)
            window = sequence[start:end]
            if len(window) == 0:
                gc_vals.append(0.0)
            else:
                gc = (window.count('G') + window.count('C')) / len(window)
                gc_vals.append(gc)
        return np.array(gc_vals, dtype=np.float32)

    def _dinucleotide_transitions(self, sequence: str) -> np.ndarray:
        counts = np.zeros(len(self.dinuc_to_idx), dtype=np.float32)
        for i in range(len(sequence) - 1):
            dinuc = sequence[i:i + 2]
            idx = self.dinuc_to_idx.get(dinuc)
            if idx is not None:
                counts[idx] += 1.0
        total = counts.sum()
        if total > 0:
            counts /= total
        return counts

    @staticmethod
    def _homopolymer_runs(sequence: str) -> np.ndarray:
        runs = []
        for base in 'ACGT':
            max_run = current = 0
            for nuc in sequence:
                if nuc == base:
                    current += 1
                    max_run = max(max_run, current)
                else:
                    current = 0
            runs.append(float(max_run))
        return np.array(runs, dtype=np.float32)

    def extract_raw(self, sequence: str) -> np.ndarray:
        features = [
            self._count_kmers(sequence, 5, self.kmer5_to_idx),
            self._positional_kmers(sequence, n_bins=10),
            self._gc_content(sequence, n_windows=10),
            self._dinucleotide_transitions(sequence),
            self._homopolymer_runs(sequence),
        ]
        return np.concatenate(features)

    def fit_svd(self, sequences: np.ndarray, sample_size: int | None = None):
        if sample_size is not None and sample_size < len(sequences):
            rng = np.random.default_rng(42)
            indices = rng.choice(len(sequences), sample_size, replace=False)
            sample_sequences = sequences[indices]
        else:
            sample_sequences = sequences

        raw_features = [self.extract_raw(seq) for seq in tqdm(sample_sequences, desc="Extracting sample features")]
        raw_matrix = np.vstack(raw_features)
        self.feature_mean = raw_matrix.mean(axis=0)
        self.feature_std = raw_matrix.std(axis=0) + 1e-8

        normalized = (raw_matrix - self.feature_mean) / self.feature_std
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.svd.fit(normalized)

    def transform(self, sequence: str) -> np.ndarray:
        if self.svd is None or self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("SVD has not been fitted. Call fit_svd first.")
        raw = self.extract_raw(sequence)
        normalized = (raw - self.feature_mean) / self.feature_std
        reduced = self.svd.transform(normalized.reshape(1, -1))
        return reduced.flatten().astype(np.float32)

    def transform_batch(self, sequences: np.ndarray, desc: str) -> np.ndarray:
        if self.svd is None:
            raise RuntimeError("SVD has not been fitted. Call fit_svd first.")
        outputs = np.zeros((len(sequences), self.n_components), dtype=np.float32)
        for idx, seq in enumerate(tqdm(sequences, desc=desc)):
            outputs[idx] = self.transform(seq)
        return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline feature cache builder for ChromatinCNN.")
    parser.add_argument("--train_sequences", required=True, help="Path to train_sequences.csv")
    parser.add_argument("--val_sequences", required=True, help="Path to val_sequences.csv")
    parser.add_argument("--test_sequences", required=True, help="Path to testsequences.csv")
    parser.add_argument("--output_dir", required=True, help="Directory to store cached features (e.g., Drive path)")
    parser.add_argument("--feature_dim", type=int, default=1024, help="Output feature dimension after SVD")
    parser.add_argument("--svd_sample_size", type=int, default=20000, help="Number of sequences to sample when fitting SVD")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_sequences = read_sequences(Path(args.train_sequences))
    val_sequences = read_sequences(Path(args.val_sequences))
    test_sequences = read_sequences(Path(args.test_sequences))

    extractor = FeatureExtractor(n_components=args.feature_dim)
    extractor.fit_svd(train_sequences, sample_size=args.svd_sample_size)

    splits = {
        "train": train_sequences,
        "val": val_sequences,
        "test": test_sequences,
    }

    metadata = {
        "feature_dim": args.feature_dim,
        "svd_sample_size": min(args.svd_sample_size, len(train_sequences)),
        "splits": {name: len(seq) for name, seq in splits.items()},
        "files": {},
    }

    for split_name, seqs in splits.items():
        feats = extractor.transform_batch(seqs, desc=f"Building {split_name} cache")
        feature_path = output_dir / f"{split_name}_features.npy"
        np.save(feature_path, feats)
        metadata["files"][split_name] = feature_path.name

    svd_state_path = output_dir / "svd_state.npz"
    np.savez_compressed(
        svd_state_path,
        components=extractor.svd.components_,
        explained_variance=extractor.svd.explained_variance_ratio_,
        feature_mean=extractor.feature_mean,
        feature_std=extractor.feature_std,
    )
    metadata["files"]["svd_state"] = svd_state_path.name

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    print(f"Feature cache written to {output_dir}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

