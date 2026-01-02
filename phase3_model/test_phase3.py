"""
Quick test script to verify Phase 3 implementation.

Runs a minimal training loop on demo data to verify:
- Model architecture works correctly
- Data loaders function properly
- Training and validation pass without errors
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from logger import get_logger, configure_logging, LogTimer
from model import ChromatinCNN, ChromatinCNNConfig
from dataset import ChromatinDataModule

# Configure logging
configure_logging(log_dir='logs', console_level='INFO')
logger = get_logger(__name__)


def test_model_architecture():
    """Test model architecture and forward pass."""
    logger.info("Testing model architecture...")

    # Create model
    config = ChromatinCNNConfig(n_classes=18)
    model = ChromatinCNN(
        n_classes=config.n_classes,
        conv1_filters=config.conv1_filters,
        conv2_filters=config.conv2_filters,
        bottleneck_filters=config.bottleneck_filters,
        kernel1=config.kernel1,
        kernel2=config.kernel2,
        dropout_rate=config.dropout_rate,
        use_l1_regularization=config.use_l1_regularization,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 200, 4)
    logger.info(f"Input shape: {x.shape}")

    # Regular forward pass
    logits = model(x)
    logger.info(f"Output shape: {logits.shape}")

    # Test with activations
    logits, activations = model(x, return_activations=True)
    logger.info("Activations returned:")
    for key, value in activations.items():
        logger.info(f"  {key}: {value.shape}")

    # Test with positions
    logits, activations, positions = model(x, return_activations=True, return_positions=True)
    logger.info(f"Max activation positions: {positions.shape}")

    # Test RC averaging
    predictions = model.predict_with_rc_consistency(x)
    logger.info(f"Predictions shape: {predictions.shape}")
    logger.info(f"Predictions: {predictions}")

    # Test L1 penalty
    l1_penalty = model.get_l1_penalty()
    logger.info(f"L1 penalty: {l1_penalty.item():.6f}")

    # Test first conv filters extraction
    filters = model.get_first_conv_filters()
    logger.info(f"First conv filters shape: {filters.shape}")

    logger.info("✓ Model architecture test passed")


def test_data_loading():
    """Test data loading and augmentation."""
    logger.info("Testing data loading...")

    # Create data module with demo data
    data_module = ChromatinDataModule(
        train_sequences='data/demo_train_sequences.csv',
        train_labels='data/demo_train_labels.csv',
        val_sequences='data/demo_val_sequences.csv',
        val_labels='data/demo_val_labels.csv',
        test_sequences='data/demo_test_sequences.csv',
        batch_size=8,
        num_workers=0,  # Use 0 for testing
        rc_augment=True,
        jitter_prob=0.5,
        noise_prob=0.01,
        sequence_length=200,
        cache_data=True,
    )

    # Check dataset sizes
    sizes = data_module.get_dataset_sizes()
    logger.info(f"Dataset sizes: {sizes}")

    # Test train dataloader
    train_loader = data_module.get_train_dataloader()
    sequences, labels = next(iter(train_loader))
    logger.info(f"Train batch - sequences shape: {sequences.shape}, labels shape: {labels.shape}")
    logger.info(f"Train labels (first 5): {labels[:5]}")

    # Test val dataloader
    val_loader = data_module.get_val_dataloader()
    sequences, labels = next(iter(val_loader))
    logger.info(f"Val batch - sequences shape: {sequences.shape}, labels shape: {labels.shape}")

    # Test test dataloader
    test_loader = data_module.get_test_dataloader()
    sequences, labels = next(iter(test_loader))
    logger.info(f"Test batch - sequences shape: {sequences.shape}")

    logger.info("✓ Data loading test passed")


def test_mini_training():
    """Test a mini training loop."""
    logger.info("Testing mini training loop...")

    # Create model
    config = ChromatinCNNConfig(n_classes=18)
    model = ChromatinCNN(
        n_classes=config.n_classes,
        conv1_filters=config.conv1_filters,
        conv2_filters=config.conv2_filters,
        bottleneck_filters=config.bottleneck_filters,
        kernel1=config.kernel1,
        kernel2=config.kernel2,
        dropout_rate=config.dropout_rate,
        use_l1_regularization=config.use_l1_regularization,
    )

    # Create data module
    data_module = ChromatinDataModule(
        train_sequences='data/demo_train_sequences.csv',
        train_labels='data/demo_train_labels.csv',
        val_sequences='data/demo_val_sequences.csv',
        val_labels='data/demo_val_labels.csv',
        test_sequences='data/demo_test_sequences.csv',
        batch_size=16,
        num_workers=0,
        rc_augment=True,
        jitter_prob=0.3,
        noise_prob=0.01,
        sequence_length=200,
        cache_data=True,
    )

    # Setup optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

    # Training loop (2 epochs)
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    model = model.to(device)

    for epoch in range(2):
        logger.info(f"Epoch {epoch + 1}")

        # Train
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        train_loader = data_module.get_train_dataloader()
        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(sequences)
            loss = criterion(logits, labels)

            # Add L1 penalty
            if config.use_l1_regularization:
                l1_penalty = model.get_l1_penalty()
                loss = loss + config.l1_weight * l1_penalty

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(logits, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = train_loss / len(train_loader)
        train_accuracy = correct / total

        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")

        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        val_loader = data_module.get_val_dataloader()
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)

                logits = model(sequences)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                _, predicted = torch.max(logits, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total

        logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    logger.info("✓ Mini training test passed")


def main():
    """Run all tests."""
    logger.info("=" * 70)
    logger.info("Phase 3 Implementation Tests")
    logger.info("=" * 70)

    try:
        with LogTimer(logger, "Test Model Architecture"):
            test_model_architecture()

        print()

        with LogTimer(logger, "Test Data Loading"):
            test_data_loading()

        print()

        with LogTimer(logger, "Test Mini Training"):
            test_mini_training()

        logger.info("=" * 70)
        logger.info("All tests passed! ✓")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == '__main__':
    main()

