"""
Activation extraction and caching for steering vector computation.

Extracts model activations from ChromatinCNN and stores them efficiently
for steering vector computation and analysis.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, List
from tqdm import tqdm

from phase3_model.model import ChromatinCNN
from .utils import get_device, pool_bottleneck_activations
from logger import get_logger, LogTimer

logger = get_logger(__name__)


class ActivationCache:
    """
    Efficiently extracts and stores activations from ChromatinCNN.

    Uses the model's built-in return_activations feature to extract
    intermediate representations for steering vector computation.

    Attributes:
        model: Trained ChromatinCNN model
        device: Computation device
        cache_dir: Directory for cached activations
    """

    def __init__(
        self,
        model: ChromatinCNN,
        device: str = 'auto',
        cache_dir: str = 'phase6_steering/cache'
    ):
        """
        Initialize with trained model.

        Args:
            model: Trained ChromatinCNN model
            device: Device for computation ('auto', 'mps', 'cuda', 'cpu')
            cache_dir: Directory for storing cached activations
        """
        self.device = get_device() if device == 'auto' else device
        self.model = model.to(self.device)
        self.model.eval()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cached data
        self._activations: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None
        self._predictions: Optional[np.ndarray] = None
        self._logits: Optional[np.ndarray] = None

        logger.info(f"ActivationCache initialized on device: {self.device}")

    def extract_activations(
        self,
        data_loader: torch.utils.data.DataLoader,
        layer_name: str = 'bottleneck',
        apply_pooling: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract activations for all sequences in the data loader.

        Args:
            data_loader: DataLoader with (sequences, labels) batches
            layer_name: Which layer to extract ('conv1', 'conv2', 'bottleneck',
                       'dense1', 'dense2')
            apply_pooling: If True and layer is bottleneck, apply global pooling
                          to get (batch, 1024) features

        Returns:
            Tuple of:
            - activations: (n_samples, n_features) array
            - labels: (n_samples,) array
            - predictions: (n_samples,) array of model predictions
            - logits: (n_samples, 18) array of raw logits
        """
        logger.info(f"Extracting activations from layer: {layer_name}")

        all_activations = []
        all_labels = []
        all_predictions = []
        all_logits = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Extracting {layer_name}"):
                # Unpack batch
                if isinstance(batch, (list, tuple)):
                    sequences = batch[0].to(self.device)
                    labels = batch[1].cpu().numpy() if len(batch) > 1 else None
                else:
                    sequences = batch.to(self.device)
                    labels = None

                # Forward pass with activation extraction
                logits, activations_dict = self.model(
                    sequences,
                    return_activations=True
                )

                # Get target activations
                layer_acts = activations_dict[layer_name]

                # Apply pooling for bottleneck layer if requested
                if layer_name == 'bottleneck' and apply_pooling:
                    layer_acts = pool_bottleneck_activations(layer_acts)

                # Store results
                all_activations.append(layer_acts.cpu().numpy())
                all_logits.append(logits.cpu().numpy())

                predictions = torch.argmax(logits, dim=1).cpu().numpy()
                all_predictions.append(predictions)

                if labels is not None:
                    all_labels.append(labels)

        # Concatenate all batches
        self._activations = np.concatenate(all_activations, axis=0)
        self._logits = np.concatenate(all_logits, axis=0)
        self._predictions = np.concatenate(all_predictions, axis=0)

        if all_labels:
            self._labels = np.concatenate(all_labels, axis=0)
        else:
            self._labels = np.zeros(len(self._predictions), dtype=np.int64)

        logger.info(f"Extracted activations shape: {self._activations.shape}")
        logger.info(f"Total samples: {len(self._predictions)}")

        return self._activations, self._labels, self._predictions, self._logits

    def compute_per_label_activations(
        self,
        activations: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None
    ) -> Dict[int, np.ndarray]:
        """
        Group activations by label for steering vector computation.

        Args:
            activations: Activations array, uses cached if None
            labels: Labels array, uses cached if None

        Returns:
            Dictionary mapping label (0-17) to activations array
        """
        acts = activations if activations is not None else self._activations
        lbls = labels if labels is not None else self._labels

        if acts is None or lbls is None:
            raise ValueError("No activations available. Run extract_activations first.")

        per_label: Dict[int, np.ndarray] = {}
        unique_labels = np.unique(lbls)

        for label in unique_labels:
            mask = lbls == label
            per_label[int(label)] = acts[mask]
            logger.debug(f"Label {label}: {per_label[label].shape[0]} samples")

        return per_label

    def compute_accuracy(
        self,
        predictions: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None
    ) -> float:
        """Compute prediction accuracy."""
        preds = predictions if predictions is not None else self._predictions
        lbls = labels if labels is not None else self._labels

        if preds is None or lbls is None:
            raise ValueError("No predictions available.")

        return float(np.mean(preds == lbls))

    def get_per_class_accuracy(
        self,
        predictions: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None
    ) -> Dict[int, float]:
        """Compute accuracy per class."""
        preds = predictions if predictions is not None else self._predictions
        lbls = labels if labels is not None else self._labels

        if preds is None or lbls is None:
            raise ValueError("No predictions available.")

        per_class = {}
        for label in np.unique(lbls):
            mask = lbls == label
            per_class[int(label)] = float(np.mean(preds[mask] == label))

        return per_class

    def save_cache(self, name: str = 'activations') -> Path:
        """
        Save extracted activations to disk.

        Args:
            name: Base name for cache files

        Returns:
            Path to saved cache file
        """
        if self._activations is None:
            raise ValueError("No activations to save. Run extract_activations first.")

        cache_path = self.cache_dir / f"{name}.npz"

        np.savez_compressed(
            cache_path,
            activations=self._activations,
            labels=self._labels,
            predictions=self._predictions,
            logits=self._logits,
        )

        logger.info(f"Saved activation cache to: {cache_path}")
        return cache_path

    def load_cache(self, name: str = 'activations') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load cached activations from disk.

        Args:
            name: Base name for cache files

        Returns:
            Tuple of (activations, labels, predictions, logits)
        """
        cache_path = self.cache_dir / f"{name}.npz"

        if not cache_path.exists():
            raise FileNotFoundError(f"Cache not found: {cache_path}")

        data = np.load(cache_path)
        self._activations = data['activations']
        self._labels = data['labels']
        self._predictions = data['predictions']
        self._logits = data['logits']

        logger.info(f"Loaded activation cache from: {cache_path}")
        logger.info(f"Activations shape: {self._activations.shape}")

        return self._activations, self._labels, self._predictions, self._logits

    def cache_exists(self, name: str = 'activations') -> bool:
        """Check if cache file exists."""
        return (self.cache_dir / f"{name}.npz").exists()

    @property
    def activations(self) -> Optional[np.ndarray]:
        """Get cached activations."""
        return self._activations

    @property
    def labels(self) -> Optional[np.ndarray]:
        """Get cached labels."""
        return self._labels

    @property
    def predictions(self) -> Optional[np.ndarray]:
        """Get cached predictions."""
        return self._predictions

    @property
    def logits(self) -> Optional[np.ndarray]:
        """Get cached logits."""
        return self._logits

    def get_confident_samples(
        self,
        confidence_threshold: float = 0.9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get samples where the model is highly confident.

        Args:
            confidence_threshold: Minimum confidence (max softmax probability)

        Returns:
            Tuple of (indices, activations, labels) for confident samples
        """
        if self._logits is None:
            raise ValueError("No logits available.")

        probs = np.exp(self._logits) / np.exp(self._logits).sum(axis=1, keepdims=True)
        confidences = np.max(probs, axis=1)
        mask = confidences >= confidence_threshold

        indices = np.where(mask)[0]
        return indices, self._activations[mask], self._labels[mask]

    def get_uncertain_samples(
        self,
        confidence_threshold: float = 0.6
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get samples where the model is uncertain.

        Args:
            confidence_threshold: Maximum confidence for "uncertain"

        Returns:
            Tuple of (indices, activations, labels) for uncertain samples
        """
        if self._logits is None:
            raise ValueError("No logits available.")

        probs = np.exp(self._logits) / np.exp(self._logits).sum(axis=1, keepdims=True)
        confidences = np.max(probs, axis=1)
        mask = confidences < confidence_threshold

        indices = np.where(mask)[0]
        return indices, self._activations[mask], self._labels[mask]

    def get_misclassified_samples(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get samples that were misclassified.

        Returns:
            Tuple of (indices, activations, true_labels, predicted_labels)
        """
        if self._predictions is None or self._labels is None:
            raise ValueError("No predictions/labels available.")

        mask = self._predictions != self._labels
        indices = np.where(mask)[0]

        return (
            indices,
            self._activations[mask],
            self._labels[mask],
            self._predictions[mask]
        )
