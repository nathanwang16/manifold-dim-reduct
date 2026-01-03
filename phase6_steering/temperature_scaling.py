"""
Temperature scaling for post-hoc calibration.

Learns an optimal temperature parameter to calibrate model confidence
so that predicted probabilities match actual accuracy rates.
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import minimize_scalar
from typing import Dict, Tuple, Optional
import json
from pathlib import Path

from .utils import compute_softmax
from logger import get_logger

logger = get_logger(__name__)


class TemperatureScaler:
    """
    Learns optimal temperature for calibrating model outputs.

    Temperature scaling divides logits by a learned temperature T > 0
    before applying softmax. Higher T produces softer (less confident)
    distributions, lower T produces sharper distributions.
    """

    def __init__(self, n_classes: int = 18):
        """
        Initialize scaler.

        Args:
            n_classes: Number of output classes
        """
        self.n_classes = n_classes
        self.temperature = 1.0
        self._is_fitted = False

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        method: str = 'nll',
        temp_range: Tuple[float, float] = (0.1, 10.0),
    ) -> float:
        """
        Learn optimal temperature from validation data.

        Args:
            logits: Raw logits of shape (n_samples, n_classes)
            labels: True labels of shape (n_samples,)
            method: Objective to minimize ('nll' or 'ece')
            temp_range: Search range for temperature

        Returns:
            Optimal temperature value
        """
        logger.info(f"Fitting temperature scaler using {method} objective")

        if method == 'nll':
            objective = lambda t: self._nll_objective(logits, labels, t)
        elif method == 'ece':
            objective = lambda t: self._ece_objective(logits, labels, t)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Grid search for robustness
        temps = np.linspace(temp_range[0], temp_range[1], 100)
        losses = [objective(t) for t in temps]
        best_idx = np.argmin(losses)
        initial_temp = temps[best_idx]

        # Fine-tune with scipy
        result = minimize_scalar(
            objective,
            bounds=(max(0.1, initial_temp - 1), min(10.0, initial_temp + 1)),
            method='bounded'
        )

        self.temperature = float(result.x)
        self._is_fitted = True

        logger.info(f"Optimal temperature: {self.temperature:.4f}")
        return self.temperature

    def _nll_objective(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        temperature: float
    ) -> float:
        """Negative log-likelihood objective."""
        scaled_logits = logits / temperature
        probs = compute_softmax(scaled_logits)

        # Get probability of true class
        n_samples = len(labels)
        true_probs = probs[np.arange(n_samples), labels]

        # NLL = -mean(log(p_true))
        nll = -np.mean(np.log(np.clip(true_probs, 1e-10, 1.0)))
        return nll

    def _ece_objective(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        temperature: float,
        n_bins: int = 15
    ) -> float:
        """Expected Calibration Error objective."""
        scaled_logits = logits / temperature
        probs = compute_softmax(scaled_logits)

        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)

        # Compute ECE
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                avg_confidence = np.mean(confidences[in_bin])
                avg_accuracy = np.mean(accuracies[in_bin])
                ece += prop_in_bin * np.abs(avg_accuracy - avg_confidence)

        return ece

    def calibrate(
        self,
        logits: np.ndarray
    ) -> np.ndarray:
        """
        Apply learned temperature to logits.

        Args:
            logits: Raw logits (n_samples, n_classes) or (n_classes,)

        Returns:
            Calibrated probabilities
        """
        scaled_logits = logits / self.temperature
        return compute_softmax(scaled_logits)

    def calibrate_torch(
        self,
        logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply learned temperature to PyTorch logits.

        Args:
            logits: Raw logits tensor

        Returns:
            Calibrated probabilities tensor
        """
        scaled_logits = logits / self.temperature
        return F.softmax(scaled_logits, dim=-1)

    def evaluate_calibration(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 15
    ) -> Dict[str, float]:
        """
        Compare calibration before and after temperature scaling.

        Args:
            logits: Raw logits
            labels: True labels
            n_bins: Number of bins for ECE

        Returns:
            Dict with ECE_before, ECE_after, NLL_before, NLL_after
        """
        # Before calibration
        probs_before = compute_softmax(logits)
        ece_before = self._compute_ece(probs_before, labels, n_bins)
        nll_before = self._nll_objective(logits, labels, 1.0)

        # After calibration
        probs_after = self.calibrate(logits)
        ece_after = self._compute_ece(probs_after, labels, n_bins)
        nll_after = self._nll_objective(logits, labels, self.temperature)

        # Compute accuracy (unchanged by calibration)
        predictions = np.argmax(logits, axis=1)
        accuracy = np.mean(predictions == labels)

        # Compute Brier score
        brier_before = self._compute_brier_score(probs_before, labels)
        brier_after = self._compute_brier_score(probs_after, labels)

        return {
            'accuracy': float(accuracy),
            'ece_before': float(ece_before),
            'ece_after': float(ece_after),
            'ece_improvement': float(ece_before - ece_after),
            'nll_before': float(nll_before),
            'nll_after': float(nll_after),
            'nll_improvement': float(nll_before - nll_after),
            'brier_before': float(brier_before),
            'brier_after': float(brier_after),
            'temperature': float(self.temperature),
        }

    def _compute_ece(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 15
    ) -> float:
        """Compute Expected Calibration Error."""
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                avg_confidence = np.mean(confidences[in_bin])
                avg_accuracy = np.mean(accuracies[in_bin])
                ece += prop_in_bin * np.abs(avg_accuracy - avg_confidence)

        return ece

    def _compute_brier_score(
        self,
        probs: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """Compute Brier score (mean squared error of probabilities)."""
        n_samples = len(labels)
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(n_samples), labels] = 1.0
        return np.mean((probs - one_hot) ** 2)

    def compute_reliability_diagram_data(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 15
    ) -> Dict[str, np.ndarray]:
        """
        Compute data for reliability diagrams (before and after calibration).

        Args:
            logits: Raw logits
            labels: True labels
            n_bins: Number of bins

        Returns:
            Dict with bin_edges, bin_confidences, bin_accuracies, bin_counts
            for both before and after calibration
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)

        results = {}

        for suffix, probs in [
            ('_before', compute_softmax(logits)),
            ('_after', self.calibrate(logits))
        ]:
            confidences = np.max(probs, axis=1)
            predictions = np.argmax(probs, axis=1)
            accuracies = (predictions == labels).astype(float)

            bin_confidences = []
            bin_accuracies = []
            bin_counts = []

            for i in range(n_bins):
                in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
                count = np.sum(in_bin)
                bin_counts.append(count)

                if count > 0:
                    bin_confidences.append(np.mean(confidences[in_bin]))
                    bin_accuracies.append(np.mean(accuracies[in_bin]))
                else:
                    bin_confidences.append((bin_boundaries[i] + bin_boundaries[i+1]) / 2)
                    bin_accuracies.append(0.0)

            results[f'bin_confidences{suffix}'] = np.array(bin_confidences)
            results[f'bin_accuracies{suffix}'] = np.array(bin_accuracies)
            results[f'bin_counts{suffix}'] = np.array(bin_counts)

        results['bin_edges'] = bin_boundaries

        return results

    def save(self, path: str) -> None:
        """Save learned temperature to file."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'temperature': self.temperature,
            'n_classes': self.n_classes,
            'is_fitted': self._is_fitted,
        }

        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved temperature scaler to: {save_path}")

    def load(self, path: str) -> None:
        """Load learned temperature from file."""
        with open(path, 'r') as f:
            data = json.load(f)

        self.temperature = data['temperature']
        self.n_classes = data['n_classes']
        self._is_fitted = data['is_fitted']

        logger.info(f"Loaded temperature scaler from: {path} (T={self.temperature:.4f})")

    @property
    def is_fitted(self) -> bool:
        """Check if scaler has been fitted."""
        return self._is_fitted
