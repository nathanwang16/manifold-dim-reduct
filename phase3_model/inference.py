"""
Inference script for ChromatinCNN.

Generates predictions for test sequences with:
- RC averaging for improved equivariance
- Batch processing
- Predictions export to CSV
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
from tqdm import tqdm

from phase3_model.model import ChromatinCNN
from phase3_model.dataset import ChromatinDataModule
from logger import get_logger, LogTimer

logger = get_logger(__name__)


def load_model(
    checkpoint_path: str,
    config_dict: dict,
    device: str = 'auto',
) -> ChromatinCNN:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        config_dict: Configuration dictionary
        device: Device to load model on

    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {checkpoint_path}")

    # Load checkpoint with weights_only=True for security
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract model config
    model_config_dict = checkpoint.get('config', {})

    # Create model
    model = ChromatinCNN(
        n_classes=model_config_dict.get('n_classes', 18),
        conv1_filters=model_config_dict.get('conv1_filters', 128),
        conv2_filters=model_config_dict.get('conv2_filters', 256),
        bottleneck_filters=model_config_dict.get('bottleneck_filters', 512),
        kernel1=model_config_dict.get('kernel1', 19),
        kernel2=model_config_dict.get('kernel2', 11),
        dropout_rate=model_config_dict.get('dropout_rate', 0.3),
        use_l1_regularization=model_config_dict.get('use_l1_regularization', True),
    )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded. Epoch: {checkpoint['epoch']}, Val Acc: {checkpoint['best_val_accuracy']:.4f}")

    return model


def predict(
    model: ChromatinCNN,
    data_loader: torch.utils.data.DataLoader,
    device: str,
    use_rc_averaging: bool = True,
    return_probabilities: bool = False,
) -> np.ndarray:
    """
    Generate predictions for a dataloader.

    Args:
        model: Trained ChromatinCNN model
        data_loader: Dataloader with sequences
        device: Device to run inference on
        use_rc_averaging: Use reverse complement averaging
        return_probabilities: Return probabilities instead of class predictions

    Returns:
        Predictions array of shape (n_samples,) or (n_samples, n_classes) for probabilities
    """
    model.eval()

    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Generating predictions"):
            # Handle both (sequences, labels) and (sequences,) formats
            if isinstance(batch, (list, tuple)):
                sequences = batch[0]
            else:
                sequences = batch

            sequences = sequences.to(device)

            if use_rc_averaging:
                # Use RC averaging for better equivariance
                probs = model.predict_with_rc_consistency(sequences, return_probs=True)
            else:
                # Simple forward pass
                logits = model(sequences)
                probs = torch.softmax(logits, dim=1)

            all_probabilities.append(probs.cpu().numpy())

    # Concatenate all batches
    all_probabilities = np.vstack(all_probabilities)

    if return_probabilities:
        return all_probabilities
    else:
        predictions = np.argmax(all_probabilities, axis=1)
        return predictions


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Run inference with ChromatinCNN')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output predictions file')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for inference')
    parser.add_argument('--use_rc_averaging', action='store_true', default=True, help='Use reverse complement averaging')
    parser.add_argument('--return_probabilities', action='store_true', help='Return class probabilities instead of predictions')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config_dict = json.load(f)

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

    logger.info(f"Using device: {device}")

    # Log MPS-specific optimizations if applicable
    if device == 'mps':
        logger.info("MPS device detected - enabling Metal Performance Shaders acceleration")

    # Load model
    model = load_model(args.checkpoint, config_dict, device)

    # Create data module for test data
    data_config = config_dict['data']
    phase3_config = config_dict.get('phase3', {})

    # Determine pin_memory based on device
    use_pin_memory = (device == 'cuda' or device == 'mps')

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
        pin_memory=use_pin_memory,
    )

    # Get test dataloader
    test_loader = data_module.get_test_dataloader()
    logger.info(f"Test dataset size: {len(data_module.test_dataset)}")

    # Generate predictions
    with LogTimer(logger, "Inference"):
        if args.return_probabilities:
            probabilities = predict(
                model,
                test_loader,
                device,
                use_rc_averaging=args.use_rc_averaging,
                return_probabilities=True,
            )

            # Save probabilities
            prob_df = pd.DataFrame(probabilities)
            prob_df.to_csv(args.output, index=False, header=False)
            logger.info(f"Probabilities saved to {args.output}")
        else:
            predictions = predict(
                model,
                test_loader,
                device,
                use_rc_averaging=args.use_rc_averaging,
                return_probabilities=False,
            )

            # Save predictions (convert from 0-17 back to 1-18 labels)
            predictions = predictions + 1
            predictions_df = pd.DataFrame(predictions)
            predictions_df.to_csv(args.output, index=False, header=False)
            logger.info(f"Predictions saved to {args.output}")
            logger.info(f"Predictions shape: {predictions.shape}")

            # Log prediction distribution
            unique, counts = np.unique(predictions, return_counts=True)
            logger.info("Prediction distribution:")
            for label, count in zip(unique, counts):
                logger.info(f"  Label {label}: {count} ({100*count/len(predictions):.2f}%)")


if __name__ == '__main__':
    main()

