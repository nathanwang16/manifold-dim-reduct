"""
Phase 3: Run pipeline for ChromatinCNN training and inference.

This script orchestrates the complete Phase 3 pipeline:
1. Load configuration
2. Initialize model and data module
3. Train the model with interpretability focus
4. Generate predictions for test set
5. Save results

Uses the centralized logger for comprehensive logging and tracking.
"""

import sys
import json
import argparse
from pathlib import Path
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from logger import get_logger, configure_logging, LogTimer, log_metrics

from phase3_model.model import ChromatinCNN, ChromatinCNNConfig
from phase3_model.dataset import ChromatinDataModule
from phase3_model.train import Trainer
from phase3_model.inference import load_model, predict


def main():
    """Main Phase 3 pipeline."""
    parser = argparse.ArgumentParser(description='Run Phase 3: ChromatinCNN Training and Inference')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['train', 'inference', 'both'], default='both',
                        help='Pipeline mode: train, inference, or both')
    parser.add_argument('--checkpoint', type=str, default='phase3_model/checkpoints/best_model.pt',
                        help='Checkpoint path for inference')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Output predictions file')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--use_demo_data', action='store_true',
                        help='Use demo data instead of full dataset')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint')

    args = parser.parse_args()

    # Configure logging
    configure_logging(log_dir='logs', console_level='INFO')
    logger = get_logger(__name__)

    logger.info("=" * 70)
    logger.info("Phase 3: ChromatinCNN Training and Inference Pipeline")
    logger.info("=" * 70)

    # Load config
    with LogTimer(logger, "Loading configuration"):
        with open(args.config, 'r') as f:
            config = json.load(f)

    # Override demo data setting if specified
    if args.use_demo_data:
        config['use_demo_data'] = True
        logger.info("Using demo data")

    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    logger.info(f"Device: {device}")

    # Log MPS-specific optimizations if applicable
    if device == 'mps':
        logger.info("MPS device detected - enabling Metal Performance Shaders acceleration")

    # Extract Phase 3 config
    phase3_config = config.get('phase3', {})

    # Log key configuration
    logger.info("Configuration:")
    logger.info(f"  Mode: {args.mode}")
    logger.info(f"  Number of epochs: {args.num_epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {phase3_config.get('learning_rate', 1e-3)}")
    logger.info(f"  Warmup epochs: {phase3_config.get('warmup_epochs', 5)}")
    logger.info(f"  Conv1 filters: {phase3_config.get('conv1_filters', 128)}")
    logger.info(f"  Conv2 filters: {phase3_config.get('conv2_filters', 256)}")
    logger.info(f"  Bottleneck filters: {phase3_config.get('bottleneck_filters', 512)}")
    logger.info(f"  L1 regularization: {phase3_config.get('use_l1_regularization', True)}")
    logger.info(f"  Label smoothing: {phase3_config.get('label_smoothing', 0.05)}")

    # Training mode
    if args.mode in ['train', 'both']:
        logger.info("-" * 70)
        logger.info("Starting Training Phase")
        logger.info("-" * 70)

        # Create model config
        model_config = ChromatinCNNConfig(
            n_classes=phase3_config.get('n_classes', 18),
            conv1_filters=phase3_config.get('conv1_filters', 128),
            conv2_filters=phase3_config.get('conv2_filters', 256),
            bottleneck_filters=phase3_config.get('bottleneck_filters', 512),
            kernel1=phase3_config.get('kernel1', 19),
            kernel2=phase3_config.get('kernel2', 11),
            dropout_rate=phase3_config.get('dropout_rate', 0.3),
            use_l1_regularization=phase3_config.get('use_l1_regularization', True),
            l1_weight=phase3_config.get('l1_weight', 1e-5),
            label_smoothing=phase3_config.get('label_smoothing', 0.05),
            learning_rate=phase3_config.get('learning_rate', 1e-3),
            warmup_epochs=phase3_config.get('warmup_epochs', 5),
        )

        # Create model
        model = ChromatinCNN(
            n_classes=model_config.n_classes,
            conv1_filters=model_config.conv1_filters,
            conv2_filters=model_config.conv2_filters,
            bottleneck_filters=model_config.bottleneck_filters,
            kernel1=model_config.kernel1,
            kernel2=model_config.kernel2,
            dropout_rate=model_config.dropout_rate,
            use_l1_regularization=model_config.use_l1_regularization,
        )

        logger.info(f"Model created: {model.__class__.__name__}")
        logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Create data module
        data_config = config['data']
        phase3_data_config = phase3_config.get('training', {})

        # Determine pin_memory based on device
        use_pin_memory = (device == 'cuda' or device == 'mps')

        data_module = ChromatinDataModule(
            train_sequences=data_config['train_sequences'],
            train_labels=data_config['train_labels'],
            val_sequences=data_config['val_sequences'],
            val_labels=data_config['val_labels'],
            test_sequences=data_config['test_sequences'],
            batch_size=args.batch_size,
            num_workers=phase3_data_config.get('num_workers', 4),
            rc_augment=phase3_data_config.get('rc_augment', True),
            jitter_prob=phase3_data_config.get('jitter_prob', 0.3),
            noise_prob=phase3_data_config.get('noise_prob', 0.01),
            sequence_length=data_config['sequence_length'],
            cache_data=phase3_data_config.get('cache_data', True),
            pin_memory=use_pin_memory,
        )

        # Log dataset sizes
        dataset_sizes = data_module.get_dataset_sizes()
        logger.info("Dataset sizes:")
        for split, size in dataset_sizes.items():
            logger.info(f"  {split}: {size:,}")

        # Create trainer
        trainer = Trainer(
            model=model,
            config=model_config,
            data_module=data_module,
            device=device,
            checkpoint_dir=phase3_config.get('checkpoint_dir', 'phase3_model/checkpoints'),
            save_best_only=True,
        )

        # Resume from checkpoint if provided
        if args.resume is not None:
            trainer.load_checkpoint(args.resume)
        # Resume from checkpoint file if not provided
        elif args.mode in ['train', 'both'] and Path('phase3_model/checkpoints/best_model.pt').exists():
            logger.info("Found existing checkpoint - use --resume to load it or it will be overwritten")
            logger.info("To resume training: python -m phase3_model.run_phase3 --resume phase3_model/checkpoints/best_model.pt")

        # Train
        logger.info("Starting training...")
        with LogTimer(logger, "Training"):
            trainer.train(
                num_epochs=args.num_epochs,
                early_stopping_patience=10,
                save_checkpoint=True,
                resume=args.resume,
            )

        # Log final metrics
        final_metrics = {
            'best_val_accuracy': trainer.best_val_accuracy,
            'best_val_loss': trainer.best_val_loss,
            'final_epoch': trainer.current_epoch,
        }
        log_metrics(logger, final_metrics, "Training Complete")

        logger.info(f"Best validation accuracy: {trainer.best_val_accuracy:.4f}")

        # Update checkpoint path for inference
        checkpoint_path = Path(phase3_config.get('checkpoint_dir', 'phase3_model/checkpoints')) / 'best_model.pt'
        args.checkpoint = str(checkpoint_path)

    # Inference mode
    if args.mode in ['inference', 'both']:
        logger.info("-" * 70)
        logger.info("Starting Inference Phase")
        logger.info("-" * 70)

        # Check if checkpoint exists
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            logger.info("Please run training first or specify a valid checkpoint path")
            return

        # Load model
        with LogTimer(logger, "Loading model"):
            model = load_model(args.checkpoint, config, device)

        # Create data module for inference
        data_config = config['data']

        data_module = ChromatinDataModule(
            train_sequences=data_config['train_sequences'],
            train_labels=data_config['train_labels'],
            val_sequences=data_config['val_sequences'],
            val_labels=data_config['val_labels'],
            test_sequences=data_config['test_sequences'],
            batch_size=args.batch_size,
            num_workers=4,
            rc_augment=False,
            jitter_prob=0.0,
            noise_prob=0.0,
            sequence_length=data_config['sequence_length'],
            cache_data=True,
        )

        # Get test dataloader
        test_loader = data_module.get_test_dataloader()
        logger.info(f"Test dataset size: {len(data_module.test_dataset):,}")

        # Generate predictions
        logger.info("Generating predictions with RC averaging...")
        with LogTimer(logger, "Inference"):
            predictions = predict(
                model,
                test_loader,
                device,
                use_rc_averaging=True,
                return_probabilities=False,
            )

        # Save predictions (convert from 0-17 back to 1-18 labels)
        import pandas as pd
        predictions = predictions + 1
        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv(args.output, index=False, header=False)
        logger.info(f"Predictions saved to {args.output}")

        # Log prediction distribution
        import numpy as np
        unique, counts = np.unique(predictions, return_counts=True)
        logger.info("Prediction distribution:")
        for label, count in zip(unique, counts):
            logger.info(f"  Label {label}: {count} ({100*count/len(predictions):.2f}%)")

        # Log final metrics
        inference_metrics = {
            'num_predictions': len(predictions),
            'output_file': args.output,
            'num_classes': len(unique),
        }
        log_metrics(logger, inference_metrics, "Inference Complete")

    logger.info("-" * 70)
    logger.info("Phase 3 Pipeline Complete")
    logger.info("-" * 70)


if __name__ == '__main__':
    main()

