"""
Comprehensive alignment evaluation suite.

Evaluates model alignment across multiple dimensions:
- RC Consistency: Same prediction for S and RC(S)
- Calibration: Reliability diagrams and ECE
- Monotonicity: Strengthening motif increases confidence
- Compositionality: Multiple motif handling
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from phase3_model.model import ChromatinCNN
from .utils import get_device, reverse_complement_tensor, interpolate_sequence, compute_entropy
from .temperature_scaling import TemperatureScaler
from logger import get_logger, LogTimer

logger = get_logger(__name__)


class AlignmentEvaluator:
    """
    Evaluates model alignment across multiple dimensions.

    Implements the alignment evaluation tests from Phase 6.4 of the guide:
    - RC Consistency: Does P(S) == P(RC(S))?
    - Calibration: Do confidences match accuracies?
    - Monotonicity: Does strengthening motifs increase confidence?
    - Compositionality: Does the model handle multiple motifs?
    """

    def __init__(
        self,
        model: ChromatinCNN,
        device: str = 'auto'
    ):
        """
        Initialize with model.

        Args:
            model: Trained ChromatinCNN model
            device: Computation device
        """
        self.device = get_device() if device == 'auto' else device
        self.model = model.to(self.device)
        self.model.eval()

        # Temperature scaler for calibration
        self.temperature_scaler = TemperatureScaler(n_classes=model.n_classes)

        logger.info(f"AlignmentEvaluator initialized on device: {self.device}")

    # ═══════════════════════════════════════════════════════════════
    # RC CONSISTENCY
    # ═══════════════════════════════════════════════════════════════

    def compute_rc_consistency(
        self,
        data_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """
        Measure if model gives same prediction for S and RC(S).

        Args:
            data_loader: DataLoader with sequences

        Returns:
            Dict with:
            - consistency_rate: Fraction with same prediction
            - per_class_consistency: Consistency rate per class
            - inconsistent_indices: Indices of inconsistent samples
            - logit_correlation: Correlation between S and RC(S) logits
        """
        logger.info("Computing RC consistency...")

        all_consistent = []
        all_labels = []
        all_logit_corrs = []
        inconsistent_indices = []
        sample_idx = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="RC consistency"):
                if isinstance(batch, (list, tuple)):
                    sequences = batch[0].to(self.device)
                    labels = batch[1].numpy() if len(batch) > 1 else None
                else:
                    sequences = batch.to(self.device)
                    labels = None

                # Original predictions
                logits_orig = self.model(sequences)
                preds_orig = torch.argmax(logits_orig, dim=1)

                # RC predictions
                sequences_rc = reverse_complement_tensor(sequences)
                logits_rc = self.model(sequences_rc)
                preds_rc = torch.argmax(logits_rc, dim=1)

                # Check consistency
                consistent = (preds_orig == preds_rc).cpu().numpy()
                all_consistent.extend(consistent.tolist())

                # Track inconsistent samples
                for i, is_consistent in enumerate(consistent):
                    if not is_consistent:
                        inconsistent_indices.append(sample_idx + i)
                sample_idx += len(sequences)

                if labels is not None:
                    all_labels.extend(labels.tolist())

                # Compute logit correlation per sample
                for i in range(len(sequences)):
                    corr = np.corrcoef(
                        logits_orig[i].cpu().numpy(),
                        logits_rc[i].cpu().numpy()
                    )[0, 1]
                    all_logit_corrs.append(corr)

        consistency_rate = np.mean(all_consistent)
        logit_correlation = np.mean(all_logit_corrs)

        # Per-class consistency
        per_class_consistency = {}
        if all_labels:
            labels_arr = np.array(all_labels)
            consistent_arr = np.array(all_consistent)
            for label in np.unique(labels_arr):
                mask = labels_arr == label
                if np.any(mask):
                    per_class_consistency[int(label)] = float(np.mean(consistent_arr[mask]))

        logger.info(f"RC consistency: {consistency_rate:.4f}")
        logger.info(f"Logit correlation: {logit_correlation:.4f}")

        return {
            'consistency_rate': float(consistency_rate),
            'per_class_consistency': per_class_consistency,
            'inconsistent_indices': inconsistent_indices,
            'logit_correlation': float(logit_correlation),
            'total_samples': len(all_consistent),
            'inconsistent_count': len(inconsistent_indices),
        }

    def analyze_rc_inconsistencies(
        self,
        sequences: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Deep analysis of RC-inconsistent predictions.

        Args:
            sequences: Input sequences
            labels: True labels

        Returns:
            Analysis of which classes are most affected
        """
        sequences = sequences.to(self.device)

        with torch.no_grad():
            logits_orig = self.model(sequences)
            preds_orig = torch.argmax(logits_orig, dim=1)
            probs_orig = F.softmax(logits_orig, dim=1)

            sequences_rc = reverse_complement_tensor(sequences)
            logits_rc = self.model(sequences_rc)
            preds_rc = torch.argmax(logits_rc, dim=1)
            probs_rc = F.softmax(logits_rc, dim=1)

        preds_orig_np = preds_orig.cpu().numpy()
        preds_rc_np = preds_rc.cpu().numpy()
        labels_np = labels.numpy()

        inconsistent_mask = preds_orig_np != preds_rc_np

        # Analyze flip patterns
        flip_matrix = np.zeros((self.model.n_classes, self.model.n_classes))
        for i in range(len(sequences)):
            if inconsistent_mask[i]:
                flip_matrix[preds_orig_np[i], preds_rc_np[i]] += 1

        # Confidence analysis
        conf_orig = torch.max(probs_orig, dim=1)[0].cpu().numpy()
        conf_rc = torch.max(probs_rc, dim=1)[0].cpu().numpy()

        return {
            'flip_matrix': flip_matrix.tolist(),
            'inconsistent_rate': float(np.mean(inconsistent_mask)),
            'mean_conf_orig_inconsistent': float(np.mean(conf_orig[inconsistent_mask])) if np.any(inconsistent_mask) else 0,
            'mean_conf_rc_inconsistent': float(np.mean(conf_rc[inconsistent_mask])) if np.any(inconsistent_mask) else 0,
            'mean_conf_orig_consistent': float(np.mean(conf_orig[~inconsistent_mask])) if np.any(~inconsistent_mask) else 0,
        }

    # ═══════════════════════════════════════════════════════════════
    # CALIBRATION
    # ═══════════════════════════════════════════════════════════════

    def compute_calibration_metrics(
        self,
        data_loader: torch.utils.data.DataLoader,
        n_bins: int = 15,
        fit_temperature: bool = True
    ) -> Dict[str, Any]:
        """
        Compute calibration error metrics.

        Args:
            data_loader: DataLoader with (sequences, labels)
            n_bins: Number of bins for reliability diagram
            fit_temperature: Whether to fit temperature scaler

        Returns:
            Dict with ECE, MCE, reliability diagram data, etc.
        """
        logger.info("Computing calibration metrics...")

        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Calibration"):
                sequences, labels = batch[0], batch[1]
                sequences = sequences.to(self.device)

                logits = self.model(sequences)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.numpy())

        logits = np.concatenate(all_logits, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        # Fit temperature scaling
        if fit_temperature:
            self.temperature_scaler.fit(logits, labels, method='nll')

        # Get calibration evaluation
        calibration_metrics = self.temperature_scaler.evaluate_calibration(
            logits, labels, n_bins
        )

        # Reliability diagram data
        reliability_data = self.temperature_scaler.compute_reliability_diagram_data(
            logits, labels, n_bins
        )
        calibration_metrics['reliability_diagram'] = reliability_data

        # Compute MCE
        bin_accuracies = reliability_data['bin_accuracies_before']
        bin_confidences = reliability_data['bin_confidences_before']
        bin_counts = reliability_data['bin_counts_before']
        valid_bins = bin_counts > 0
        if np.any(valid_bins):
            calibration_metrics['mce_before'] = float(
                np.max(np.abs(bin_accuracies[valid_bins] - bin_confidences[valid_bins]))
            )
        else:
            calibration_metrics['mce_before'] = 0.0

        bin_accuracies_after = reliability_data['bin_accuracies_after']
        bin_confidences_after = reliability_data['bin_confidences_after']
        bin_counts_after = reliability_data['bin_counts_after']
        valid_bins_after = bin_counts_after > 0
        if np.any(valid_bins_after):
            calibration_metrics['mce_after'] = float(
                np.max(np.abs(bin_accuracies_after[valid_bins_after] - bin_confidences_after[valid_bins_after]))
            )
        else:
            calibration_metrics['mce_after'] = 0.0

        logger.info(f"ECE before: {calibration_metrics['ece_before']:.4f}")
        logger.info(f"ECE after: {calibration_metrics['ece_after']:.4f}")

        return calibration_metrics

    def compute_reliability_diagram(
        self,
        confidences: np.ndarray,
        predictions: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 15
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute reliability diagram data.

        Args:
            confidences: Max softmax probability per sample
            predictions: Predicted labels
            labels: True labels
            n_bins: Number of bins

        Returns:
            Tuple of (bin_confidences, bin_accuracies, bin_counts)
        """
        accuracies = (predictions == labels).astype(float)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)

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

        return np.array(bin_confidences), np.array(bin_accuracies), np.array(bin_counts)

    # ═══════════════════════════════════════════════════════════════
    # MONOTONICITY
    # ═══════════════════════════════════════════════════════════════

    def test_monotonicity(
        self,
        base_sequence: torch.Tensor,
        motif: str,
        position: int,
        expected_label: int,
        strength_levels: int = 10
    ) -> Dict[str, Any]:
        """
        Test if strengthening a motif monotonically increases confidence.

        Args:
            base_sequence: Starting sequence (1, 200, 4)
            motif: Motif string to strengthen (e.g., 'TATAAA')
            position: Position to insert motif
            expected_label: Expected label for this motif
            strength_levels: Number of interpolation steps

        Returns:
            Dict with confidence trajectory and monotonicity score
        """
        base_sequence = base_sequence.to(self.device)

        # Create target sequence with motif
        target_sequence = base_sequence.clone()
        motif_onehot = self._string_to_onehot(motif)

        # Check bounds
        if position + len(motif) > 200:
            position = 200 - len(motif)

        # Insert motif into target
        target_sequence[0, position:position+len(motif), :] = motif_onehot.to(self.device)

        # Interpolate between base and target
        alphas = np.linspace(0, 1, strength_levels)
        confidence_trajectory = []
        prediction_trajectory = []

        with torch.no_grad():
            for alpha in alphas:
                interp_seq = interpolate_sequence(base_sequence, target_sequence, alpha)
                logits = self.model(interp_seq)
                probs = F.softmax(logits, dim=1)

                confidence_trajectory.append(probs[0, expected_label].item())
                prediction_trajectory.append(torch.argmax(probs, dim=1).item())

        # Compute monotonicity score
        monotonicity_score = self._compute_monotonicity_score(confidence_trajectory)

        return {
            'motif': motif,
            'position': position,
            'expected_label': expected_label,
            'strength_levels': alphas.tolist(),
            'confidence_trajectory': confidence_trajectory,
            'prediction_trajectory': prediction_trajectory,
            'monotonicity_score': float(monotonicity_score),
            'final_confidence': confidence_trajectory[-1],
            'confidence_increase': confidence_trajectory[-1] - confidence_trajectory[0],
        }

    def _string_to_onehot(self, seq: str) -> torch.Tensor:
        """Convert DNA string to one-hot tensor."""
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        onehot = torch.zeros(len(seq), 4)
        for i, base in enumerate(seq.upper()):
            if base in mapping:
                onehot[i, mapping[base]] = 1.0
            else:
                onehot[i, :] = 0.25  # Unknown base
        return onehot

    def _compute_monotonicity_score(self, trajectory: List[float]) -> float:
        """
        Compute monotonicity score for a trajectory.

        Score = (# monotonic pairs) / (# total pairs)
        """
        n = len(trajectory)
        if n < 2:
            return 1.0

        monotonic_pairs = 0
        total_pairs = 0

        for i in range(n):
            for j in range(i + 1, n):
                total_pairs += 1
                if trajectory[j] >= trajectory[i] - 1e-6:  # Small tolerance
                    monotonic_pairs += 1

        return monotonic_pairs / total_pairs if total_pairs > 0 else 1.0

    def batch_monotonicity_test(
        self,
        test_cases: List[Dict[str, Any]],
        data_loader: Optional[torch.utils.data.DataLoader] = None
    ) -> Dict[str, Any]:
        """
        Run monotonicity tests for multiple motif-label pairs.

        Args:
            test_cases: List of dicts with 'motif', 'position', 'expected_label'
            data_loader: Optional DataLoader to get base sequences from

        Returns:
            Summary statistics of monotonicity across all tests
        """
        logger.info(f"Running {len(test_cases)} monotonicity tests...")

        results = []

        # Get base sequences
        if data_loader is not None:
            # Use sequences from data loader
            batch = next(iter(data_loader))
            base_sequences = batch[0][:len(test_cases)]
        else:
            # Generate random sequences
            base_sequences = torch.rand(len(test_cases), 200, 4)
            base_sequences = F.gumbel_softmax(base_sequences, tau=0.1, hard=True)

        for i, test_case in enumerate(tqdm(test_cases, desc="Monotonicity tests")):
            base_seq = base_sequences[i:i+1] if i < len(base_sequences) else base_sequences[0:1]
            result = self.test_monotonicity(
                base_seq,
                test_case['motif'],
                test_case.get('position', 100),
                test_case['expected_label']
            )
            results.append(result)

        # Aggregate statistics
        scores = [r['monotonicity_score'] for r in results]
        increases = [r['confidence_increase'] for r in results]

        return {
            'n_tests': len(results),
            'mean_monotonicity_score': float(np.mean(scores)),
            'std_monotonicity_score': float(np.std(scores)),
            'min_monotonicity_score': float(np.min(scores)),
            'mean_confidence_increase': float(np.mean(increases)),
            'perfect_monotonicity_rate': float(np.mean([s == 1.0 for s in scores])),
            'individual_results': results,
        }

    # ═══════════════════════════════════════════════════════════════
    # COMPOSITIONALITY
    # ═══════════════════════════════════════════════════════════════

    def test_compositionality(
        self,
        base_sequence: torch.Tensor,
        motifs: List[Tuple[str, int]],
        expected_labels: List[int]
    ) -> Dict[str, Any]:
        """
        Test if model handles motif combinations sensibly.

        Args:
            base_sequence: Starting sequence (1, 200, 4)
            motifs: List of (motif_string, position) tuples
            expected_labels: Expected labels for each motif

        Returns:
            Analysis of how multiple motifs affect predictions
        """
        base_sequence = base_sequence.to(self.device)

        with torch.no_grad():
            # Baseline prediction
            logits_base = self.model(base_sequence)
            probs_base = F.softmax(logits_base, dim=1)[0]
            pred_base = torch.argmax(probs_base).item()

            # Individual motif predictions
            individual_results = []
            for (motif, position), expected_label in zip(motifs, expected_labels):
                seq = base_sequence.clone()
                motif_onehot = self._string_to_onehot(motif).to(self.device)

                if position + len(motif) <= 200:
                    seq[0, position:position+len(motif), :] = motif_onehot

                logits = self.model(seq)
                probs = F.softmax(logits, dim=1)[0]

                individual_results.append({
                    'motif': motif,
                    'position': position,
                    'expected_label': expected_label,
                    'prediction': torch.argmax(probs).item(),
                    'expected_prob': probs[expected_label].item(),
                    'max_prob': torch.max(probs).item(),
                })

            # Combined motifs prediction
            seq_combined = base_sequence.clone()
            for motif, position in motifs:
                motif_onehot = self._string_to_onehot(motif).to(self.device)
                if position + len(motif) <= 200:
                    seq_combined[0, position:position+len(motif), :] = motif_onehot

            logits_combined = self.model(seq_combined)
            probs_combined = F.softmax(logits_combined, dim=1)[0]

        # Analyze compositionality
        entropy_base = compute_entropy(probs_base.cpu().numpy().reshape(1, -1))[0]
        entropy_combined = compute_entropy(probs_combined.cpu().numpy().reshape(1, -1))[0]

        return {
            'baseline': {
                'prediction': pred_base,
                'entropy': float(entropy_base),
            },
            'individual_motifs': individual_results,
            'combined': {
                'prediction': torch.argmax(probs_combined).item(),
                'probs_for_expected': [probs_combined[l].item() for l in expected_labels],
                'max_prob': torch.max(probs_combined).item(),
                'entropy': float(entropy_combined),
            },
            'entropy_change': float(entropy_combined - entropy_base),
        }

    # ═══════════════════════════════════════════════════════════════
    # REPORT GENERATION
    # ═══════════════════════════════════════════════════════════════

    def generate_alignment_report(
        self,
        val_loader: torch.utils.data.DataLoader,
        output_dir: Path,
        run_monotonicity: bool = False,
        monotonicity_test_cases: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive alignment report with all metrics.

        Args:
            val_loader: Validation DataLoader
            output_dir: Directory for output files
            run_monotonicity: Whether to run monotonicity tests
            monotonicity_test_cases: Test cases for monotonicity

        Returns:
            Complete alignment report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        figures_dir = output_dir / 'figures'
        figures_dir.mkdir(exist_ok=True)

        report = {}

        # 1. RC Consistency
        with LogTimer(logger, "RC Consistency evaluation"):
            report['rc_consistency'] = self.compute_rc_consistency(val_loader)

        # 2. Calibration
        with LogTimer(logger, "Calibration evaluation"):
            report['calibration'] = self.compute_calibration_metrics(val_loader)

        # 3. Plot reliability diagram
        self._plot_reliability_diagram(
            report['calibration']['reliability_diagram'],
            figures_dir / 'reliability_diagram.png'
        )

        # 4. Monotonicity (optional)
        if run_monotonicity and monotonicity_test_cases:
            with LogTimer(logger, "Monotonicity evaluation"):
                report['monotonicity'] = self.batch_monotonicity_test(
                    monotonicity_test_cases, val_loader
                )

        # 5. Save temperature scaler
        self.temperature_scaler.save(output_dir / 'temperature.json')

        # 6. Save report
        # Convert numpy arrays to lists for JSON serialization
        report_json = self._make_json_serializable(report)
        with open(output_dir / 'alignment_report.json', 'w') as f:
            json.dump(report_json, f, indent=2)

        logger.info(f"Alignment report saved to: {output_dir}")

        return report

    def _make_json_serializable(self, obj: Any) -> Any:
        """Recursively convert numpy types to Python types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        return obj

    def _plot_reliability_diagram(
        self,
        reliability_data: Dict[str, np.ndarray],
        save_path: Path
    ) -> None:
        """Plot reliability diagram."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, suffix, title in [
            (axes[0], '_before', 'Before Temperature Scaling'),
            (axes[1], '_after', 'After Temperature Scaling')
        ]:
            bin_confidences = np.array(reliability_data[f'bin_confidences{suffix}'])
            bin_accuracies = np.array(reliability_data[f'bin_accuracies{suffix}'])
            bin_counts = np.array(reliability_data[f'bin_counts{suffix}'])

            # Normalize counts for coloring
            max_count = max(bin_counts.max(), 1)
            colors = plt.cm.Blues(bin_counts / max_count)

            # Bar plot
            n_bins = len(bin_confidences)
            width = 1.0 / n_bins
            positions = np.arange(n_bins) * width + width / 2

            ax.bar(positions, bin_accuracies, width=width * 0.8, color=colors, edgecolor='black')
            ax.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')

            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title(title)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.legend()
            ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved reliability diagram to: {save_path}")
