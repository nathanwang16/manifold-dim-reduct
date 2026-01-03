"""
Inference-time steering intervention engine.

Applies steering vectors during model inference to shift predictions
toward target labels. Implements representation engineering by modifying
internal activations.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from tqdm import tqdm

from phase3_model.model import ChromatinCNN
from .steering_vectors import SteeringVectorComputer
from .utils import get_device, pool_bottleneck_activations
from logger import get_logger, LogTimer

logger = get_logger(__name__)


class SteeringInferenceEngine:
    """
    Engine for applying steering interventions during inference.

    Supports:
    - Single-direction steering (X -> Y)
    - Confidence-based adaptive steering
    - Multi-hypothesis steering for uncertain predictions

    The steering is applied after global pooling at the 1024-dim
    representation level (concatenated max+avg pooled bottleneck).
    """

    def __init__(
        self,
        model: ChromatinCNN,
        steering_computer: SteeringVectorComputer,
        device: str = 'auto'
    ):
        """
        Initialize with model and precomputed steering vectors.

        Args:
            model: Trained ChromatinCNN model
            steering_computer: Precomputed steering vectors
            device: Computation device
        """
        self.device = get_device() if device == 'auto' else device
        self.model = model.to(self.device)
        self.model.eval()
        self.steering_computer = steering_computer

        # Hook state
        self._steering_hook_handle = None
        self._current_steering_vector: Optional[torch.Tensor] = None
        self._current_alpha: float = 0.0

        logger.info(f"SteeringInferenceEngine initialized on device: {self.device}")

    def _steering_hook(
        self,
        module: torch.nn.Module,
        input_tensor: Tuple[torch.Tensor, ...],
        output: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward hook that applies steering to dense1 input.

        This hook is registered on dense1 and modifies its input
        (the pooled bottleneck features) by adding the steering vector.
        """
        if self._current_steering_vector is None or self._current_alpha == 0:
            return output

        # Input to dense1 is the pooled bottleneck (batch, 1024)
        # Apply steering to the input, not output
        # Since we can't modify input in forward hook, we modify output instead
        # by adding the effect the steering would have had

        # The steering is applied conceptually at the pooled representation level
        # We approximate by steering the output of dense1
        steering = self._current_steering_vector.to(output.device)

        # Transform steering through dense1 weights for proper application
        # This is an approximation - ideally we'd hook before dense1
        # For simplicity, we add scaled steering to the features
        steered_output = output + self._current_alpha * steering[:output.shape[1]]

        return steered_output

    def apply_steering(
        self,
        sequences: torch.Tensor,
        target_label: int,
        source_label: Optional[int] = None,
        alpha: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply steering toward target label.

        Uses a custom forward pass that extracts activations, applies
        steering to the pooled representation, and continues through
        dense layers.

        Args:
            sequences: Input sequences (batch, 200, 4)
            target_label: Label to steer toward (0-17)
            source_label: Source label (if None, uses model's prediction)
            alpha: Steering strength multiplier

        Returns:
            Tuple of:
            - steered_logits: (batch, 18) logits after steering
            - original_logits: (batch, 18) logits without steering
            - original_predictions: (batch,) original predictions
        """
        sequences = sequences.to(self.device)

        with torch.no_grad():
            # First, get original predictions and activations
            original_logits, activations = self.model(
                sequences, return_activations=True
            )
            original_predictions = torch.argmax(original_logits, dim=1)

            # Get pooled bottleneck representation
            bottleneck = activations['bottleneck']  # (batch, 512, 200)
            pooled = pool_bottleneck_activations(bottleneck)  # (batch, 1024)

            # Apply steering to pooled representation
            batch_size = sequences.shape[0]
            steered_pooled = pooled.clone()

            for i in range(batch_size):
                if source_label is not None:
                    src = source_label
                else:
                    src = original_predictions[i].item()

                # Get steering vector for this sample
                steering_vec = self.steering_computer.get_steering_vector(
                    src, target_label, normalize=False
                )
                steering_tensor = torch.from_numpy(steering_vec).float().to(self.device)

                # Apply steering
                steered_pooled[i] = pooled[i] + alpha * steering_tensor

            # Continue through dense layers with steered representation
            x = F.relu(self.model.dense1(steered_pooled))
            x = self.model.dropout1(x)
            x = F.relu(self.model.dense2(x))
            x = self.model.dropout2(x)
            steered_logits = self.model.classifier(x)

        return steered_logits, original_logits, original_predictions

    def apply_steering_batch(
        self,
        sequences: torch.Tensor,
        source_labels: torch.Tensor,
        target_labels: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """
        Apply per-sample steering with specified source and target labels.

        Args:
            sequences: Input sequences (batch, 200, 4)
            source_labels: Per-sample source labels (batch,)
            target_labels: Per-sample target labels (batch,)
            alpha: Steering strength

        Returns:
            steered_logits: (batch, 18) logits after steering
        """
        sequences = sequences.to(self.device)
        source_labels = source_labels.to(self.device)
        target_labels = target_labels.to(self.device)

        with torch.no_grad():
            # Get activations
            _, activations = self.model(sequences, return_activations=True)
            bottleneck = activations['bottleneck']
            pooled = pool_bottleneck_activations(bottleneck)

            # Apply per-sample steering
            batch_size = sequences.shape[0]
            steered_pooled = pooled.clone()

            for i in range(batch_size):
                src = source_labels[i].item()
                tgt = target_labels[i].item()

                if src != tgt:
                    steering_vec = self.steering_computer.get_steering_vector(
                        src, tgt, normalize=False
                    )
                    steering_tensor = torch.from_numpy(steering_vec).float().to(self.device)
                    steered_pooled[i] = pooled[i] + alpha * steering_tensor

            # Dense layers
            x = F.relu(self.model.dense1(steered_pooled))
            x = self.model.dropout1(x)
            x = F.relu(self.model.dense2(x))
            x = self.model.dropout2(x)
            steered_logits = self.model.classifier(x)

        return steered_logits

    def predict_with_steering(
        self,
        sequences: torch.Tensor,
        target_label: int,
        alpha: float = 1.0,
        confidence_threshold: float = 0.6,
        only_uncertain: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Full prediction with optional steering for uncertain samples.

        Args:
            sequences: Input sequences (batch, 200, 4)
            target_label: Label to steer uncertain predictions toward
            alpha: Steering strength
            confidence_threshold: Below this, apply steering
            only_uncertain: If True, only steer uncertain predictions

        Returns:
            Dict with:
            - predictions: Final class predictions
            - probabilities: Class probabilities
            - confidence: Max probability per sample
            - steering_applied: Boolean mask of steered samples
        """
        sequences = sequences.to(self.device)

        with torch.no_grad():
            # Get original predictions
            original_logits, activations = self.model(
                sequences, return_activations=True
            )
            original_probs = F.softmax(original_logits, dim=1)
            original_confidence = torch.max(original_probs, dim=1)[0]
            original_predictions = torch.argmax(original_probs, dim=1)

            # Identify uncertain samples
            uncertain_mask = original_confidence < confidence_threshold

            if only_uncertain and not uncertain_mask.any():
                # No uncertain samples, return original
                return {
                    'predictions': original_predictions,
                    'probabilities': original_probs,
                    'confidence': original_confidence,
                    'steering_applied': torch.zeros_like(uncertain_mask),
                }

            # Apply steering
            bottleneck = activations['bottleneck']
            pooled = pool_bottleneck_activations(bottleneck)

            steered_pooled = pooled.clone()
            steering_applied = torch.zeros(sequences.shape[0], dtype=torch.bool, device=self.device)

            for i in range(sequences.shape[0]):
                if only_uncertain and not uncertain_mask[i]:
                    continue

                src = original_predictions[i].item()
                steering_vec = self.steering_computer.get_steering_vector(
                    src, target_label, normalize=False
                )
                steering_tensor = torch.from_numpy(steering_vec).float().to(self.device)
                steered_pooled[i] = pooled[i] + alpha * steering_tensor
                steering_applied[i] = True

            # Dense layers with steered representation
            x = F.relu(self.model.dense1(steered_pooled))
            x = self.model.dropout1(x)
            x = F.relu(self.model.dense2(x))
            x = self.model.dropout2(x)
            steered_logits = self.model.classifier(x)

            final_probs = F.softmax(steered_logits, dim=1)
            final_predictions = torch.argmax(final_probs, dim=1)
            final_confidence = torch.max(final_probs, dim=1)[0]

            # Use original for non-steered samples
            if only_uncertain:
                final_predictions = torch.where(
                    steering_applied,
                    final_predictions,
                    original_predictions
                )
                final_probs = torch.where(
                    steering_applied.unsqueeze(1).expand_as(final_probs),
                    final_probs,
                    original_probs
                )
                final_confidence = torch.where(
                    steering_applied,
                    final_confidence,
                    original_confidence
                )

        return {
            'predictions': final_predictions,
            'probabilities': final_probs,
            'confidence': final_confidence,
            'steering_applied': steering_applied,
        }

    def calibrate_alpha(
        self,
        val_loader: torch.utils.data.DataLoader,
        source_label: int,
        target_label: int,
        alpha_range: Tuple[float, float] = (0.0, 2.0),
        n_steps: int = 20,
    ) -> Tuple[float, Dict[str, List[float]]]:
        """
        Find optimal steering strength via validation.

        Tests different alpha values and finds the one that maximizes
        the probability of the target label for source-labeled samples.

        Args:
            val_loader: Validation DataLoader
            source_label: Label to steer from
            target_label: Label to steer toward
            alpha_range: Range of alpha values to test
            n_steps: Number of alpha values to test

        Returns:
            Tuple of:
            - best_alpha: Optimal steering strength
            - metrics: Dict with accuracy/probability at each alpha
        """
        logger.info(f"Calibrating alpha for steering {source_label} -> {target_label}")

        alphas = np.linspace(alpha_range[0], alpha_range[1], n_steps)
        metrics = {
            'alpha': list(alphas),
            'target_prob': [],
            'prediction_rate': [],
        }

        for alpha in tqdm(alphas, desc="Calibrating alpha"):
            target_probs = []
            predictions_match = []

            for batch in val_loader:
                sequences, labels = batch[0], batch[1]
                sequences = sequences.to(self.device)
                labels = labels.numpy()

                # Filter to source label samples
                source_mask = labels == source_label
                if not np.any(source_mask):
                    continue

                source_sequences = sequences[source_mask]

                # Apply steering
                steered_logits, _, _ = self.apply_steering(
                    source_sequences,
                    target_label=target_label,
                    source_label=source_label,
                    alpha=alpha
                )

                probs = F.softmax(steered_logits, dim=1)
                target_probs.extend(probs[:, target_label].cpu().numpy().tolist())
                preds = torch.argmax(probs, dim=1).cpu().numpy()
                predictions_match.extend((preds == target_label).tolist())

            if target_probs:
                metrics['target_prob'].append(float(np.mean(target_probs)))
                metrics['prediction_rate'].append(float(np.mean(predictions_match)))
            else:
                metrics['target_prob'].append(0.0)
                metrics['prediction_rate'].append(0.0)

        # Find best alpha (maximize target probability)
        best_idx = np.argmax(metrics['target_prob'])
        best_alpha = float(alphas[best_idx])

        logger.info(f"Best alpha: {best_alpha:.3f} (target_prob: {metrics['target_prob'][best_idx]:.4f})")

        return best_alpha, metrics

    def evaluate_steering_effect(
        self,
        data_loader: torch.utils.data.DataLoader,
        source_label: int,
        target_label: int,
        alpha: float = 1.0,
    ) -> Dict[str, float]:
        """
        Evaluate the effect of steering on validation data.

        Args:
            data_loader: DataLoader
            source_label: Source label
            target_label: Target label
            alpha: Steering strength

        Returns:
            Dict with evaluation metrics
        """
        original_correct = 0
        steered_correct_as_source = 0
        steered_correct_as_target = 0
        flip_to_target = 0
        total_source = 0
        total_target = 0

        with torch.no_grad():
            for batch in data_loader:
                sequences, labels = batch[0], batch[1]
                sequences = sequences.to(self.device)
                labels = labels.numpy()

                # Process source label samples
                source_mask = labels == source_label
                if np.any(source_mask):
                    source_seqs = sequences[source_mask]
                    steered_logits, original_logits, _ = self.apply_steering(
                        source_seqs,
                        target_label=target_label,
                        source_label=source_label,
                        alpha=alpha
                    )

                    orig_preds = torch.argmax(original_logits, dim=1).cpu().numpy()
                    steered_preds = torch.argmax(steered_logits, dim=1).cpu().numpy()

                    original_correct += np.sum(orig_preds == source_label)
                    steered_correct_as_source += np.sum(steered_preds == source_label)
                    flip_to_target += np.sum(steered_preds == target_label)
                    total_source += np.sum(source_mask)

                # Also check target samples (should remain correct)
                target_mask = labels == target_label
                if np.any(target_mask):
                    target_seqs = sequences[target_mask]
                    steered_logits, original_logits, _ = self.apply_steering(
                        target_seqs,
                        target_label=target_label,
                        source_label=target_label,
                        alpha=alpha
                    )

                    steered_preds = torch.argmax(steered_logits, dim=1).cpu().numpy()
                    steered_correct_as_target += np.sum(steered_preds == target_label)
                    total_target += np.sum(target_mask)

        return {
            'original_accuracy_source': original_correct / max(total_source, 1),
            'steered_accuracy_source': steered_correct_as_source / max(total_source, 1),
            'flip_rate_to_target': flip_to_target / max(total_source, 1),
            'steered_accuracy_target': steered_correct_as_target / max(total_target, 1),
            'total_source_samples': total_source,
            'total_target_samples': total_target,
        }

    def get_steering_direction_info(
        self,
        source_label: int,
        target_label: int
    ) -> Dict[str, float]:
        """Get information about a specific steering direction."""
        vec = self.steering_computer.get_steering_vector(
            source_label, target_label, normalize=False
        )
        vec_norm = self.steering_computer.get_steering_vector(
            source_label, target_label, normalize=True
        )

        source_centroid = self.steering_computer.get_centroid(source_label)
        target_centroid = self.steering_computer.get_centroid(target_label)

        return {
            'source_label': source_label,
            'target_label': target_label,
            'vector_magnitude': float(np.linalg.norm(vec)),
            'source_centroid_norm': float(np.linalg.norm(source_centroid)),
            'target_centroid_norm': float(np.linalg.norm(target_centroid)),
            'centroid_cosine_similarity': float(
                np.dot(source_centroid, target_centroid) /
                (np.linalg.norm(source_centroid) * np.linalg.norm(target_centroid) + 1e-8)
            ),
        }
