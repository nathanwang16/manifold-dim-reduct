"""
Contrastive steering for confusion-based correction.

Identifies frequently confused label pairs and implements bidirectional
steering to resolve model uncertainty between them.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from tqdm import tqdm

from phase3_model.model import ChromatinCNN
from .inference_steering import SteeringInferenceEngine
from .steering_vectors import SteeringVectorComputer
from .utils import get_device, pool_bottleneck_activations, cosine_similarity
from logger import get_logger, LogTimer

logger = get_logger(__name__)


class ContrastiveSteeringEngine:
    """
    Handles confusion-aware steering for frequently misclassified pairs.

    Identifies which label pairs the model confuses most often, then
    uses contrastive steering to help resolve ambiguity at inference time.
    """

    def __init__(
        self,
        steering_engine: SteeringInferenceEngine,
        n_classes: int = 18
    ):
        """
        Initialize with steering engine.

        Args:
            steering_engine: SteeringInferenceEngine instance
            n_classes: Number of classes
        """
        self.steering_engine = steering_engine
        self.model = steering_engine.model
        self.steering_computer = steering_engine.steering_computer
        self.device = steering_engine.device
        self.n_classes = n_classes

        # Cached analysis
        self._confusion_matrix: Optional[np.ndarray] = None
        self._confused_pairs: Optional[List[Tuple[int, int, float]]] = None
        self._bidirectional_alphas: Dict[Tuple[int, int], Dict[str, float]] = {}

        logger.info("ContrastiveSteeringEngine initialized")

    def compute_confusion_matrix(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Compute confusion matrix from predictions and labels.

        Args:
            predictions: Predicted labels (n_samples,)
            labels: True labels (n_samples,)
            normalize: Whether to normalize rows (true label)

        Returns:
            confusion: (n_classes, n_classes) confusion matrix
                confusion[i, j] = P(predict j | true i)
        """
        cm = sklearn_confusion_matrix(
            labels, predictions,
            labels=list(range(self.n_classes))
        )

        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1)  # Avoid division by zero
            cm = cm.astype(float) / row_sums

        self._confusion_matrix = cm
        logger.info(f"Computed confusion matrix, shape: {cm.shape}")

        return cm

    def identify_confused_pairs(
        self,
        confusion_matrix: Optional[np.ndarray] = None,
        threshold: float = 0.05,
        top_k: int = 10,
        symmetric: bool = True
    ) -> List[Tuple[int, int, float]]:
        """
        Find label pairs with high mutual confusion.

        Args:
            confusion_matrix: Confusion matrix, uses cached if None
            threshold: Minimum confusion rate to consider
            top_k: Maximum number of pairs to return
            symmetric: If True, merge (i,j) and (j,i) into single pair

        Returns:
            List of (label_i, label_j, confusion_score) tuples,
            sorted by confusion score descending
        """
        cm = confusion_matrix if confusion_matrix is not None else self._confusion_matrix

        if cm is None:
            raise ValueError("No confusion matrix available")

        pairs = []

        for i in range(self.n_classes):
            for j in range(self.n_classes):
                if i == j:
                    continue

                # Confusion rate: P(predict j | true i)
                conf_ij = cm[i, j]

                if conf_ij >= threshold:
                    pairs.append((i, j, conf_ij))

        # Sort by confusion rate
        pairs.sort(key=lambda x: x[2], reverse=True)

        if symmetric:
            # Merge symmetric pairs
            seen = set()
            merged = []
            for i, j, conf in pairs:
                key = (min(i, j), max(i, j))
                if key not in seen:
                    # Use max confusion of the two directions
                    conf_ji = cm[j, i]
                    max_conf = max(conf, conf_ji)
                    merged.append((key[0], key[1], max_conf))
                    seen.add(key)
            pairs = merged
            pairs.sort(key=lambda x: x[2], reverse=True)

        self._confused_pairs = pairs[:top_k]
        logger.info(f"Identified {len(self._confused_pairs)} confused pairs (threshold={threshold})")

        for i, j, conf in self._confused_pairs[:5]:
            logger.info(f"  Labels {i} <-> {j}: confusion = {conf:.4f}")

        return self._confused_pairs

    def compute_bidirectional_steering(
        self,
        confused_pairs: Optional[List[Tuple[int, int, float]]] = None
    ) -> Dict[Tuple[int, int], Dict[str, np.ndarray]]:
        """
        Compute bidirectional steering vectors for confused pairs.

        Args:
            confused_pairs: List of confused pairs, uses cached if None

        Returns:
            Dict mapping (i, j) to:
                {'i_to_j': vector, 'j_to_i': vector, 'magnitude': float}
        """
        pairs = confused_pairs if confused_pairs is not None else self._confused_pairs

        if pairs is None:
            raise ValueError("No confused pairs available")

        result = {}

        for label_i, label_j, _ in pairs:
            vec_i_to_j = self.steering_computer.get_steering_vector(
                label_i, label_j, normalize=False
            )
            vec_j_to_i = self.steering_computer.get_steering_vector(
                label_j, label_i, normalize=False
            )

            result[(label_i, label_j)] = {
                'i_to_j': vec_i_to_j,
                'j_to_i': vec_j_to_i,
                'magnitude_i_to_j': float(np.linalg.norm(vec_i_to_j)),
                'magnitude_j_to_i': float(np.linalg.norm(vec_j_to_i)),
                'cosine_opposite': float(cosine_similarity(vec_i_to_j, -vec_j_to_i)),
            }

        logger.info(f"Computed bidirectional steering for {len(result)} pairs")
        return result

    def apply_contrastive_correction(
        self,
        sequences: torch.Tensor,
        confidence_threshold: float = 0.6,
        alpha: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply contrastive steering to uncertain predictions between confused pairs.

        When the model is uncertain between two frequently confused labels,
        this method uses steering to help distinguish them.

        Args:
            sequences: Input sequences (batch, 200, 4)
            confidence_threshold: Below this, attempt correction
            alpha: Steering strength

        Returns:
            Dict with:
            - original_predictions: Before correction
            - corrected_predictions: After correction
            - corrections_made: Boolean mask of corrected samples
            - correction_details: Per-sample correction info
        """
        if self._confused_pairs is None:
            raise ValueError("Must identify confused pairs first")

        sequences = sequences.to(self.device)
        batch_size = sequences.shape[0]

        with torch.no_grad():
            # Get original predictions
            original_logits, activations = self.model(
                sequences, return_activations=True
            )
            original_probs = F.softmax(original_logits, dim=1)
            original_preds = torch.argmax(original_probs, dim=1)
            original_confidence = torch.max(original_probs, dim=1)[0]

            # Initialize outputs
            corrected_preds = original_preds.clone()
            corrections_made = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

            # Get pooled bottleneck
            bottleneck = activations['bottleneck']
            pooled = pool_bottleneck_activations(bottleneck)

            # Process uncertain samples
            uncertain_mask = original_confidence < confidence_threshold
            uncertain_indices = torch.where(uncertain_mask)[0]

            for idx in uncertain_indices:
                pred = original_preds[idx].item()
                probs = original_probs[idx]

                # Check if prediction is in a confused pair
                for label_i, label_j, _ in self._confused_pairs:
                    if pred == label_i:
                        other_label = label_j
                    elif pred == label_j:
                        other_label = label_i
                    else:
                        continue

                    # Check if the other label has high probability too
                    if probs[other_label] < 0.1:
                        continue

                    # Apply contrastive steering: try both directions
                    # Steer toward predicted label (reinforce)
                    pooled_sample = pooled[idx:idx+1]

                    # Steer toward pred
                    vec_to_pred = self.steering_computer.get_steering_vector(
                        other_label, pred, normalize=False
                    )
                    steered_to_pred = pooled_sample + alpha * torch.from_numpy(vec_to_pred).float().to(self.device)

                    # Steer toward other
                    vec_to_other = self.steering_computer.get_steering_vector(
                        pred, other_label, normalize=False
                    )
                    steered_to_other = pooled_sample + alpha * torch.from_numpy(vec_to_other).float().to(self.device)

                    # Get logits for both directions
                    logits_to_pred = self._forward_from_pooled(steered_to_pred)
                    logits_to_other = self._forward_from_pooled(steered_to_other)

                    conf_to_pred = F.softmax(logits_to_pred, dim=1)[0, pred].item()
                    conf_to_other = F.softmax(logits_to_other, dim=1)[0, other_label].item()

                    # Choose the direction that increases confidence more
                    if conf_to_other > conf_to_pred and conf_to_other > probs[other_label].item():
                        corrected_preds[idx] = other_label
                        corrections_made[idx] = True
                    elif conf_to_pred > original_confidence[idx].item():
                        # Prediction stays the same but we mark as corrected
                        corrections_made[idx] = True

                    break  # Only process first matching pair

        return {
            'original_predictions': original_preds,
            'corrected_predictions': corrected_preds,
            'corrections_made': corrections_made,
            'original_confidence': original_confidence,
        }

    def _forward_from_pooled(self, pooled: torch.Tensor) -> torch.Tensor:
        """Continue forward pass from pooled bottleneck representation."""
        x = F.relu(self.model.dense1(pooled))
        x = self.model.dropout1(x)
        x = F.relu(self.model.dense2(x))
        x = self.model.dropout2(x)
        return self.model.classifier(x)

    def evaluate_contrastive_improvement(
        self,
        val_loader: torch.utils.data.DataLoader,
        confidence_threshold: float = 0.6,
        alpha: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Evaluate improvement from contrastive steering on validation set.

        Args:
            val_loader: Validation DataLoader
            confidence_threshold: Threshold for uncertain samples
            alpha: Steering strength

        Returns:
            Dict with before/after accuracy and per-pair improvements
        """
        all_labels = []
        all_original_preds = []
        all_corrected_preds = []
        all_corrections = []

        with LogTimer(logger, "Evaluating contrastive steering"):
            for batch in tqdm(val_loader, desc="Evaluating"):
                sequences, labels = batch[0], batch[1]

                result = self.apply_contrastive_correction(
                    sequences,
                    confidence_threshold=confidence_threshold,
                    alpha=alpha
                )

                all_labels.append(labels.numpy())
                all_original_preds.append(result['original_predictions'].cpu().numpy())
                all_corrected_preds.append(result['corrected_predictions'].cpu().numpy())
                all_corrections.append(result['corrections_made'].cpu().numpy())

        labels = np.concatenate(all_labels)
        original_preds = np.concatenate(all_original_preds)
        corrected_preds = np.concatenate(all_corrected_preds)
        corrections = np.concatenate(all_corrections)

        # Overall metrics
        original_accuracy = np.mean(original_preds == labels)
        corrected_accuracy = np.mean(corrected_preds == labels)
        correction_rate = np.mean(corrections)

        # Corrections that helped vs hurt
        corrected_mask = corrections
        if np.any(corrected_mask):
            original_correct = original_preds[corrected_mask] == labels[corrected_mask]
            corrected_correct = corrected_preds[corrected_mask] == labels[corrected_mask]
            helped = np.sum(~original_correct & corrected_correct)
            hurt = np.sum(original_correct & ~corrected_correct)
            neutral = np.sum(original_correct == corrected_correct)
        else:
            helped = hurt = neutral = 0

        # Per-pair analysis
        pair_stats = {}
        if self._confused_pairs:
            for label_i, label_j, _ in self._confused_pairs:
                # Samples in either class
                pair_mask = (labels == label_i) | (labels == label_j)
                if not np.any(pair_mask):
                    continue

                pair_orig_acc = np.mean(original_preds[pair_mask] == labels[pair_mask])
                pair_corr_acc = np.mean(corrected_preds[pair_mask] == labels[pair_mask])

                # Use string key instead of tuple for JSON serialization
                pair_key = f"{label_i}_{label_j}"
                pair_stats[pair_key] = {
                    'label_i': label_i,
                    'label_j': label_j,
                    'original_accuracy': float(pair_orig_acc),
                    'corrected_accuracy': float(pair_corr_acc),
                    'improvement': float(pair_corr_acc - pair_orig_acc),
                    'n_samples': int(np.sum(pair_mask)),
                }

        return {
            'original_accuracy': float(original_accuracy),
            'corrected_accuracy': float(corrected_accuracy),
            'improvement': float(corrected_accuracy - original_accuracy),
            'correction_rate': float(correction_rate),
            'corrections_helped': int(helped),
            'corrections_hurt': int(hurt),
            'corrections_neutral': int(neutral),
            'pair_stats': pair_stats,
        }

    def calibrate_pair_alpha(
        self,
        val_loader: torch.utils.data.DataLoader,
        label_i: int,
        label_j: int,
        alpha_range: Tuple[float, float] = (0.0, 2.0),
        n_steps: int = 10,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calibrate alpha for a specific confused pair.

        Args:
            val_loader: Validation DataLoader
            label_i: First label in pair
            label_j: Second label in pair
            alpha_range: Range of alpha values
            n_steps: Number of alpha values to test

        Returns:
            Tuple of (best_alpha, calibration_metrics)
        """
        alphas = np.linspace(alpha_range[0], alpha_range[1], n_steps)
        results = []

        for alpha in alphas:
            # Temporarily set this as the confused pair to test
            original_pairs = self._confused_pairs
            self._confused_pairs = [(label_i, label_j, 1.0)]

            metrics = self.evaluate_contrastive_improvement(
                val_loader,
                alpha=alpha
            )
            metrics['alpha'] = alpha
            results.append(metrics)

            self._confused_pairs = original_pairs

        # Find best alpha by improvement
        improvements = [r['improvement'] for r in results]
        best_idx = np.argmax(improvements)
        best_alpha = float(alphas[best_idx])

        # Store calibrated alpha
        self._bidirectional_alphas[(label_i, label_j)] = {
            'alpha': best_alpha,
            'improvement': improvements[best_idx],
        }

        logger.info(f"Calibrated alpha for ({label_i}, {label_j}): {best_alpha:.3f}")

        return best_alpha, {
            'alphas': list(alphas),
            'improvements': improvements,
            'best_alpha': best_alpha,
            'best_improvement': improvements[best_idx],
        }

    def get_confused_pairs_summary(self) -> Dict[str, Any]:
        """Get summary of confused pairs and their analysis."""
        if self._confused_pairs is None:
            return {'pairs': []}

        pairs_info = []
        for label_i, label_j, conf_score in self._confused_pairs:
            info = {
                'label_i': label_i,
                'label_j': label_j,
                'confusion_score': float(conf_score),
            }

            # Add calibrated alpha if available
            if (label_i, label_j) in self._bidirectional_alphas:
                info['calibrated_alpha'] = self._bidirectional_alphas[(label_i, label_j)]['alpha']

            pairs_info.append(info)

        return {
            'n_pairs': len(self._confused_pairs),
            'pairs': pairs_info,
        }
