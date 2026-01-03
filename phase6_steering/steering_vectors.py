"""
Steering vector computation for representation engineering.

Computes steering vectors from mean activation differences between labels.
v_{X->Y} = mu_Y - mu_X where mu is the mean activation per label.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import json

from .utils import l2_normalize, cosine_similarity
from logger import get_logger, LogTimer

logger = get_logger(__name__)


class SteeringVectorComputer:
    """
    Computes and stores steering vectors for all label pairs.

    Steering vectors enable representation engineering by defining
    directions in activation space that shift model behavior from
    one label toward another.

    Attributes:
        n_classes: Number of classes (default 18)
        n_features: Feature dimension (default 1024 for post-pooled bottleneck)
        label_centroids: Mean activation per label (18, 1024)
        steering_matrix: Full pairwise steering vectors (18, 18, 1024)
    """

    def __init__(self, n_classes: int = 18, n_features: int = 1024):
        """
        Initialize steering vector computer.

        Args:
            n_classes: Number of chromatin state classes
            n_features: Dimension of activation vectors
        """
        self.n_classes = n_classes
        self.n_features = n_features

        # Computed values
        self.label_centroids: Optional[np.ndarray] = None  # (n_classes, n_features)
        self.steering_matrix: Optional[np.ndarray] = None  # (n_classes, n_classes, n_features)
        self.label_counts: Optional[np.ndarray] = None  # (n_classes,)
        self.label_variances: Optional[np.ndarray] = None  # (n_classes, n_features)

        logger.info(f"SteeringVectorComputer: {n_classes} classes, {n_features} features")

    def compute_label_centroids(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        normalize: bool = False
    ) -> np.ndarray:
        """
        Compute mean activation for each label.

        Args:
            activations: Activations array of shape (n_samples, n_features)
            labels: Labels array of shape (n_samples,) with values 0 to n_classes-1
            normalize: Whether to L2-normalize centroids

        Returns:
            centroids: (n_classes, n_features) mean activation per label
        """
        logger.info("Computing label centroids...")

        self.label_centroids = np.zeros((self.n_classes, self.n_features))
        self.label_counts = np.zeros(self.n_classes)
        self.label_variances = np.zeros((self.n_classes, self.n_features))

        for label in range(self.n_classes):
            mask = labels == label
            count = np.sum(mask)

            if count > 0:
                label_acts = activations[mask]
                self.label_centroids[label] = np.mean(label_acts, axis=0)
                self.label_variances[label] = np.var(label_acts, axis=0)
                self.label_counts[label] = count
                logger.debug(f"Label {label}: {count} samples, centroid norm: {np.linalg.norm(self.label_centroids[label]):.4f}")
            else:
                logger.warning(f"Label {label} has no samples")

        if normalize:
            self.label_centroids = l2_normalize(self.label_centroids, axis=1)

        logger.info(f"Computed centroids for {np.sum(self.label_counts > 0)} labels")
        return self.label_centroids

    def compute_steering_vectors(self) -> np.ndarray:
        """
        Compute all pairwise steering vectors: v_{i->j} = mu_j - mu_i

        Returns:
            steering_matrix: (n_classes, n_classes, n_features)
                where steering_matrix[i, j] is the vector to steer from label i to j
        """
        if self.label_centroids is None:
            raise ValueError("Must compute label centroids first")

        logger.info("Computing steering vector matrix...")

        self.steering_matrix = np.zeros(
            (self.n_classes, self.n_classes, self.n_features)
        )

        for i in range(self.n_classes):
            for j in range(self.n_classes):
                # v_{i->j} = mu_j - mu_i
                self.steering_matrix[i, j] = self.label_centroids[j] - self.label_centroids[i]

        logger.info(f"Computed {self.n_classes * self.n_classes} steering vectors")
        return self.steering_matrix

    def get_steering_vector(
        self,
        source_label: int,
        target_label: int,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Get steering vector from source to target label.

        Args:
            source_label: Label to steer from (0-17)
            target_label: Label to steer toward (0-17)
            normalize: Whether to L2-normalize the vector

        Returns:
            Steering vector of shape (n_features,)
        """
        if self.steering_matrix is None:
            raise ValueError("Must compute steering vectors first")

        vector = self.steering_matrix[source_label, target_label].copy()

        if normalize:
            norm = np.linalg.norm(vector)
            if norm > 1e-8:
                vector = vector / norm

        return vector

    def get_centroid(self, label: int) -> np.ndarray:
        """Get centroid for a specific label."""
        if self.label_centroids is None:
            raise ValueError("Must compute label centroids first")
        return self.label_centroids[label].copy()

    def compute_centroid_distances(self) -> np.ndarray:
        """
        Compute pairwise distances between label centroids.

        Returns:
            distance_matrix: (n_classes, n_classes) Euclidean distances
        """
        if self.label_centroids is None:
            raise ValueError("Must compute label centroids first")

        distances = np.zeros((self.n_classes, self.n_classes))
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                distances[i, j] = np.linalg.norm(
                    self.label_centroids[i] - self.label_centroids[j]
                )
        return distances

    def compute_centroid_similarities(self) -> np.ndarray:
        """
        Compute pairwise cosine similarities between label centroids.

        Returns:
            similarity_matrix: (n_classes, n_classes) cosine similarities
        """
        if self.label_centroids is None:
            raise ValueError("Must compute label centroids first")

        similarities = np.zeros((self.n_classes, self.n_classes))
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                similarities[i, j] = cosine_similarity(
                    self.label_centroids[i],
                    self.label_centroids[j]
                )
        return similarities

    def compute_steering_effectiveness(
        self,
        activations: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        """
        Compute how separable each label pair is via steering.

        Measures alignment between individual samples' offset from source
        centroid and the steering vector direction.

        Args:
            activations: Sample activations (n_samples, n_features)
            labels: Sample labels (n_samples,)

        Returns:
            effectiveness_matrix: (n_classes, n_classes)
                Higher values indicate steering is more likely to succeed
        """
        if self.label_centroids is None or self.steering_matrix is None:
            raise ValueError("Must compute centroids and steering vectors first")

        effectiveness = np.zeros((self.n_classes, self.n_classes))

        for source_label in range(self.n_classes):
            source_mask = labels == source_label
            if not np.any(source_mask):
                continue

            source_acts = activations[source_mask]
            source_centroid = self.label_centroids[source_label]

            for target_label in range(self.n_classes):
                if source_label == target_label:
                    continue

                steering_vec = self.steering_matrix[source_label, target_label]
                steering_norm = np.linalg.norm(steering_vec)

                if steering_norm < 1e-8:
                    continue

                steering_dir = steering_vec / steering_norm

                # Compute alignment of samples with steering direction
                offsets = source_acts - source_centroid
                alignments = np.dot(offsets, steering_dir)

                # Effectiveness = variance explained in steering direction
                effectiveness[source_label, target_label] = np.var(alignments)

        return effectiveness

    def find_most_similar_labels(
        self,
        label: int,
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Find labels most similar to a given label in activation space.

        Args:
            label: Query label
            top_k: Number of similar labels to return

        Returns:
            List of (label, similarity) tuples, sorted by similarity
        """
        similarities = self.compute_centroid_similarities()
        label_sims = similarities[label]

        # Exclude self
        label_sims[label] = -np.inf

        # Sort by similarity
        sorted_indices = np.argsort(label_sims)[::-1][:top_k]

        return [(int(idx), float(label_sims[idx])) for idx in sorted_indices]

    def find_most_different_labels(
        self,
        label: int,
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Find labels most different from a given label in activation space.

        Args:
            label: Query label
            top_k: Number of different labels to return

        Returns:
            List of (label, distance) tuples, sorted by distance
        """
        distances = self.compute_centroid_distances()
        label_dists = distances[label]

        # Sort by distance (descending)
        sorted_indices = np.argsort(label_dists)[::-1][:top_k]

        return [(int(idx), float(label_dists[idx])) for idx in sorted_indices]

    def save(self, path: str) -> None:
        """
        Serialize steering vectors to disk.

        Args:
            path: Path to save file (will add .npz extension)
        """
        if self.label_centroids is None:
            raise ValueError("No data to save")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            save_path,
            label_centroids=self.label_centroids,
            steering_matrix=self.steering_matrix,
            label_counts=self.label_counts,
            label_variances=self.label_variances,
            n_classes=self.n_classes,
            n_features=self.n_features,
        )

        logger.info(f"Saved steering vectors to: {save_path}")

    @classmethod
    def load(cls, path: str) -> 'SteeringVectorComputer':
        """
        Load precomputed steering vectors.

        Args:
            path: Path to saved file

        Returns:
            Loaded SteeringVectorComputer instance
        """
        data = np.load(path)

        instance = cls(
            n_classes=int(data['n_classes']),
            n_features=int(data['n_features'])
        )
        instance.label_centroids = data['label_centroids']
        instance.steering_matrix = data['steering_matrix']
        instance.label_counts = data['label_counts']
        instance.label_variances = data['label_variances']

        logger.info(f"Loaded steering vectors from: {path}")
        return instance

    def get_statistics(self) -> Dict:
        """Get statistics about steering vectors."""
        if self.label_centroids is None:
            return {}

        distances = self.compute_centroid_distances()
        similarities = self.compute_centroid_similarities()

        # Get off-diagonal elements
        mask = ~np.eye(self.n_classes, dtype=bool)

        return {
            'n_classes': self.n_classes,
            'n_features': self.n_features,
            'total_samples': int(np.sum(self.label_counts)) if self.label_counts is not None else 0,
            'centroid_distance_mean': float(np.mean(distances[mask])),
            'centroid_distance_std': float(np.std(distances[mask])),
            'centroid_distance_min': float(np.min(distances[mask])),
            'centroid_distance_max': float(np.max(distances[mask])),
            'centroid_similarity_mean': float(np.mean(similarities[mask])),
            'centroid_similarity_std': float(np.std(similarities[mask])),
            'centroid_norm_mean': float(np.mean(np.linalg.norm(self.label_centroids, axis=1))),
        }
