"""
Main orchestration script for Phase 6: Steering & Alignment.

Runs the complete Phase 6 pipeline:
1. Load trained model and data
2. Extract and cache activations
3. Compute steering vectors
4. Run alignment evaluation
5. Apply contrastive steering for confused pairs
6. Generate reports
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from phase3_model.model import ChromatinCNN
from phase3_model.dataset import ChromatinDataModule
from phase3_model.inference import load_model

from phase6_steering.activation_cache import ActivationCache
from phase6_steering.steering_vectors import SteeringVectorComputer
from phase6_steering.inference_steering import SteeringInferenceEngine
from phase6_steering.contrastive_steering import ContrastiveSteeringEngine
from phase6_steering.alignment_evaluation import AlignmentEvaluator
from phase6_steering.utils import get_device, MetricsTracker

from logger import get_logger, LogTimer, configure_logging, log_metrics

logger = get_logger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def run_activation_extraction(
    model: ChromatinCNN,
    data_module: ChromatinDataModule,
    config: dict,
    force_recompute: bool = False
) -> tuple:
    """Extract and cache activations."""
    cache_dir = config.get('phase6', {}).get('cache_dir', 'phase6_steering/cache')
    cache = ActivationCache(model, cache_dir=cache_dir)

    cache_name = 'activations_train'

    if cache.cache_exists(cache_name) and not force_recompute:
        logger.info("Loading cached activations...")
        return cache.load_cache(cache_name)
    else:
        logger.info("Extracting activations from training data...")
        train_loader = data_module.get_train_dataloader()
        activations, labels, predictions, logits = cache.extract_activations(
            train_loader,
            layer_name='bottleneck',
            apply_pooling=True
        )
        cache.save_cache(cache_name)
        return activations, labels, predictions, logits


def run_steering_computation(
    activations: np.ndarray,
    labels: np.ndarray,
    config: dict,
    force_recompute: bool = False
) -> SteeringVectorComputer:
    """Compute steering vectors."""
    cache_dir = Path(config.get('phase6', {}).get('cache_dir', 'phase6_steering/cache'))
    steering_path = cache_dir / 'steering_vectors.npz'

    if steering_path.exists() and not force_recompute:
        logger.info("Loading cached steering vectors...")
        return SteeringVectorComputer.load(steering_path)
    else:
        logger.info("Computing steering vectors...")
        steering = SteeringVectorComputer(n_classes=18, n_features=activations.shape[1])
        steering.compute_label_centroids(activations, labels)
        steering.compute_steering_vectors()
        steering.save(steering_path)

        # Log statistics
        stats = steering.get_statistics()
        log_metrics(logger, stats, "Steering vector statistics")

        return steering


def run_alignment_evaluation(
    model: ChromatinCNN,
    data_module: ChromatinDataModule,
    config: dict
) -> dict:
    """Run alignment evaluation suite."""
    results_dir = Path(config.get('phase6', {}).get('results_dir', 'phase6_steering/results'))

    evaluator = AlignmentEvaluator(model)
    val_loader = data_module.get_val_dataloader()

    # Example monotonicity test cases (common regulatory motifs)
    monotonicity_cases = [
        {'motif': 'TATAAA', 'position': 25, 'expected_label': 0},  # TATA box
        {'motif': 'CCAAT', 'position': 80, 'expected_label': 1},   # CAAT box
        {'motif': 'GGGCGG', 'position': 100, 'expected_label': 2}, # GC box
        {'motif': 'CACGTG', 'position': 100, 'expected_label': 3}, # E-box
    ]

    report = evaluator.generate_alignment_report(
        val_loader,
        results_dir,
        run_monotonicity=True,
        monotonicity_test_cases=monotonicity_cases
    )

    return report


def run_contrastive_steering(
    model: ChromatinCNN,
    steering: SteeringVectorComputer,
    data_module: ChromatinDataModule,
    config: dict
) -> dict:
    """Run contrastive steering analysis."""
    phase6_config = config.get('phase6', {})
    contrastive_config = phase6_config.get('contrastive', {})

    # Create engines
    steering_engine = SteeringInferenceEngine(model, steering)
    contrastive_engine = ContrastiveSteeringEngine(steering_engine)

    # Get validation predictions
    val_loader = data_module.get_val_dataloader()

    # Extract predictions for confusion analysis
    cache = ActivationCache(model)
    _, labels, predictions, _ = cache.extract_activations(val_loader, apply_pooling=True)

    # Compute confusion matrix and identify confused pairs
    contrastive_engine.compute_confusion_matrix(predictions, labels)
    confused_pairs = contrastive_engine.identify_confused_pairs(
        threshold=contrastive_config.get('confusion_threshold', 0.05),
        top_k=contrastive_config.get('top_k_pairs', 10)
    )

    # Evaluate contrastive steering improvement
    val_loader = data_module.get_val_dataloader()  # Fresh loader
    improvement = contrastive_engine.evaluate_contrastive_improvement(
        val_loader,
        confidence_threshold=phase6_config.get('steering', {}).get('confidence_threshold', 0.6),
        alpha=phase6_config.get('steering', {}).get('default_alpha', 0.5)
    )

    # Save confusion analysis
    results_dir = Path(phase6_config.get('results_dir', 'phase6_steering/results'))
    confusion_report = {
        'confused_pairs': contrastive_engine.get_confused_pairs_summary(),
        'improvement': improvement,
    }

    with open(results_dir / 'confusion_analysis.json', 'w') as f:
        json.dump(confusion_report, f, indent=2)

    return confusion_report


def main():
    """Main orchestration."""
    parser = argparse.ArgumentParser(description='Phase 6: Steering & Alignment')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (overrides config)')
    parser.add_argument('--mode', choices=['full', 'steering', 'alignment', 'contrastive'],
                        default='full', help='Which components to run')
    parser.add_argument('--force-recompute', action='store_true',
                        help='Force recomputation of cached values')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory for log files')
    parser.add_argument('--use-demo-data', action='store_true',
                        help='Use demo data instead of full dataset')
    args = parser.parse_args()

    # Configure logging
    configure_logging(log_dir=args.log_dir)
    logger.info("=" * 60)
    logger.info("Phase 6: Steering & Alignment Techniques")
    logger.info("=" * 60)

    # Load config
    config = load_config(args.config)
    
    # Override demo data setting if specified
    if args.use_demo_data:
        config['use_demo_data'] = True
        logger.info("Using demo data")
    
    phase3_config = config.get('phase3', {})
    phase6_config = config.get('phase6', {})

    # Determine checkpoint path
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = phase3_config.get('checkpoint_dir', 'phase3_model/checkpoints')
        checkpoint_path = str(Path(checkpoint_path) / 'best_model.pt')

    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Load model
    with LogTimer(logger, "Loading model"):
        model = load_model(checkpoint_path, config, device)
        model.eval()

    # Load data
    with LogTimer(logger, "Loading data"):
        data_config = config.get('data', {})
        
        # Use demo data if specified
        if config.get('use_demo_data', False):
            train_seq = 'data/demo_train_sequences.csv'
            train_lbl = 'data/demo_train_labels.csv'
            val_seq = 'data/demo_val_sequences.csv'
            val_lbl = 'data/demo_val_labels.csv'
            test_seq = 'data/demo_test_sequences.csv'
            batch_size = 32  # Smaller batch for demo
            num_workers = 0  # Disable multiprocessing for demo
        else:
            train_seq = data_config.get('train_sequences')
            train_lbl = data_config.get('train_labels')
            val_seq = data_config.get('val_sequences')
            val_lbl = data_config.get('val_labels')
            test_seq = data_config.get('test_sequences')
            batch_size = 256
            num_workers = phase3_config.get('training', {}).get('num_workers', 4)
        
        data_module = ChromatinDataModule(
            train_sequences=train_seq,
            train_labels=train_lbl,
            val_sequences=val_seq,
            val_labels=val_lbl,
            test_sequences=test_seq,
            batch_size=batch_size,
            num_workers=num_workers,
            rc_augment=False,  # No augmentation for analysis
            jitter_prob=0.0,
            noise_prob=0.0,
        )

    # Create output directories
    cache_dir = Path(phase6_config.get('cache_dir', 'phase6_steering/cache'))
    results_dir = Path(phase6_config.get('results_dir', 'phase6_steering/results'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Metrics tracker
    tracker = MetricsTracker(logger)

    # Run components based on mode
    steering = None

    if args.mode in ['full', 'steering', 'contrastive']:
        # 1. Extract activations
        with LogTimer(logger, "Activation extraction"):
            activations, labels, predictions, logits = run_activation_extraction(
                model, data_module, config, args.force_recompute
            )

        # Log basic metrics
        accuracy = np.mean(predictions == labels)
        tracker.log('train_accuracy', accuracy)
        logger.info(f"Training accuracy: {accuracy:.4f}")

        # 2. Compute steering vectors
        with LogTimer(logger, "Steering vector computation"):
            steering = run_steering_computation(
                activations, labels, config, args.force_recompute
            )

    if args.mode in ['full', 'alignment']:
        # 3. Alignment evaluation
        with LogTimer(logger, "Alignment evaluation"):
            alignment_report = run_alignment_evaluation(model, data_module, config)

        tracker.log('rc_consistency', alignment_report['rc_consistency']['consistency_rate'])
        tracker.log('ece_before', alignment_report['calibration']['ece_before'])
        tracker.log('ece_after', alignment_report['calibration']['ece_after'])

    if args.mode in ['full', 'contrastive']:
        # 4. Contrastive steering
        if steering is None:
            # Need to load steering vectors
            steering_path = cache_dir / 'steering_vectors.npz'
            steering = SteeringVectorComputer.load(steering_path)

        with LogTimer(logger, "Contrastive steering analysis"):
            confusion_report = run_contrastive_steering(
                model, steering, data_module, config
            )

        tracker.log('contrastive_improvement',
                    confusion_report['improvement']['improvement'])

    # Summary
    logger.info("=" * 60)
    logger.info("Phase 6 Complete - Summary")
    logger.info("=" * 60)
    summary = tracker.summarize()
    for metric, stats in summary.items():
        logger.info(f"  {metric}: {stats['last']:.4f}")

    # Save summary
    summary_path = results_dir / 'phase6_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
