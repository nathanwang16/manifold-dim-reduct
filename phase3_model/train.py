"""
Training script for ChromatinCNN with interpretability focus.

Implements training protocol from Phase 3.4:
- AdamW optimizer with weight decay
- Warmup + cosine annealing learning rate schedule
- Early stopping with patience
- Label smoothing
- L1 regularization on first conv layer
- Gradient clipping
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import numpy as np
from pathlib import Path
import json
from typing import Dict, Optional, Tuple
import argparse

from phase3_model.model import ChromatinCNN, ChromatinCNNConfig
from phase3_model.dataset import ChromatinDataModule
from logger import get_logger, LogTimer, log_metrics, ProgressLogger

logger = get_logger(__name__)


class Trainer:
    """
    Trainer class for ChromatinCNN with comprehensive logging and monitoring.
    """

    def __init__(
        self,
        model: ChromatinCNN,
        config: ChromatinCNNConfig,
        data_module: ChromatinDataModule,
        device: str = 'auto',
        checkpoint_dir: str = 'phase3_model/checkpoints',
        log_metrics_file: str = 'phase3_model/training_metrics.jsonl',
        save_best_only: bool = True,
    ):
        """
        Initialize trainer.

        Args:
            model: ChromatinCNN model
            config: Model and training configuration
            data_module: Data module with train/val/test dataloaders
            device: Device to train on ('cuda' or 'cpu')
            checkpoint_dir: Directory to save checkpoints
            log_metrics_file: Path to save training metrics
            save_best_only: Only save best checkpoint (by validation accuracy)
        """
        self.model = model.to(device)
        self.config = config
        self.data_module = data_module
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_metrics_file = Path(log_metrics_file)
        self.save_best_only = save_best_only

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_metrics_file.parent.mkdir(parents=True, exist_ok=True)

        # Setup optimizer and scheduler
        self._setup_optimizer_and_scheduler()

        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

        # Training state
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Training history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rates': [],
        }

        logger.info(f"Trainer initialized on device: {device}")

    def _setup_optimizer_and_scheduler(self):
        """Setup AdamW optimizer with warmup and cosine annealing schedule."""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-4,
        )

        # Warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1e-2,
            end_factor=1.0,
            total_iters=self.config.warmup_epochs,  # Configurable warmup epochs
        )

        # Main cosine annealing scheduler
        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=45,  # Total epochs - warmup
            eta_min=1e-6,
        )

        # Combine schedulers
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[self.config.warmup_epochs],  # Switch after warmup
        )

        logger.info(f"Optimizer and scheduler configured with LR={self.config.learning_rate}, warmup={self.config.warmup_epochs} epochs")

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (train_loss, train_accuracy)
        """
        self.model.train()

        train_loader = self.data_module.get_train_dataloader()
        progress = ProgressLogger(
            logger,
            total=len(train_loader),
            desc=f"Epoch {epoch} [Train]",
            log_every=50,
        )

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (sequences, labels) in enumerate(train_loader):
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            logits = self.model(sequences)
            loss = self.criterion(logits, labels)

            # Add L1 regularization penalty
            if self.config.use_l1_regularization:
                l1_penalty = self.model.get_l1_penalty()
                loss = loss + self.config.l1_weight * l1_penalty

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            progress.update()

        progress.finish()

        # Compute epoch metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def validate(self) -> Tuple[float, float]:
        """
        Validate the model.

        Returns:
            Tuple of (val_loss, val_accuracy)
        """
        self.model.eval()

        val_loader = self.data_module.get_val_dataloader()
        progress = ProgressLogger(
            logger,
            total=len(val_loader),
            desc="Validation",
            log_every=50,
        )

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                logits = self.model(sequences)
                loss = self.criterion(logits, labels)

                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(logits, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                progress.update()

        progress.finish()

        # Compute metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def train(
        self,
        num_epochs: int = 50,
        early_stopping_patience: int = 10,
        save_checkpoint: bool = True,
        resume: Optional[str] = None,
    ):
        """
        Train the model.

        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            save_checkpoint: Whether to save checkpoints
            resume: Path to checkpoint file to resume from
        """
        # Resume from checkpoint if provided
        if resume is not None:
            self.load_checkpoint(resume)
            logger.info(f"Resuming training from {resume}")
            start_epoch = self.current_epoch + 1
        else:
            start_epoch = 1

        logger.info(f"Starting training for {num_epochs} epochs")

        try:
            for epoch in range(start_epoch, num_epochs + 1):
                self.current_epoch = epoch

                # Train
                with LogTimer(logger, f"Epoch {epoch} training"):
                    train_loss, train_accuracy = self.train_epoch(epoch)

                # Validate
                with LogTimer(logger, "Validation"):
                    val_loss, val_accuracy = self.validate()

                # Update learning rate
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']

                # Log metrics
                metrics = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_accuracy': train_accuracy,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'learning_rate': current_lr,
                }

                log_metrics(logger, metrics, f"Epoch {epoch} Summary")

                # Update history
                self.history['train_loss'].append(train_loss)
                self.history['train_accuracy'].append(train_accuracy)
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_accuracy)
                self.history['learning_rates'].append(current_lr)

                # Save checkpoint if validation accuracy improved
                if val_accuracy > self.best_val_accuracy:
                    logger.info(
                        f"Validation accuracy improved from {self.best_val_accuracy:.4f} to {val_accuracy:.4f}"
                    )
                    self.best_val_accuracy = val_accuracy
                    self.best_val_loss = val_loss
                    self.patience_counter = 0

                    if save_checkpoint:
                        self._save_checkpoint(is_best=True)
                else:
                    self.patience_counter += 1

                # Early stopping
                if self.patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                    break

                # Periodic checkpoint
                if save_checkpoint and not self.save_best_only and epoch % 10 == 0:
                    self._save_checkpoint(is_best=False, epoch=epoch)

            logger.info(f"Training completed. Best validation accuracy: {self.best_val_accuracy:.4f}")

            # Save training history
            self._save_history()

        finally:
            # Ensure dataloaders are cleaned up even if training fails
            self.data_module.cleanup()

    def _save_checkpoint(self, is_best: bool, epoch: Optional[int] = None):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.to_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'best_val_loss': self.best_val_loss,
        }

        if is_best:
            path = self.checkpoint_dir / 'best_model.pt'
            logger.info(f"Saving best checkpoint to {path}")
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            logger.info(f"Saving checkpoint to {path}")

        torch.save(checkpoint, path)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load model from checkpoint and return epoch number.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Epoch number from checkpoint
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_accuracy = checkpoint['best_val_accuracy']
        self.best_val_loss = checkpoint['best_val_loss']

        logger.info(
            f"Resumed from epoch {self.current_epoch} "
            f"(val_acc: {self.best_val_accuracy:.4f})"
        )

        return self.current_epoch

    def _save_history(self):
        """Save training history to JSON."""
        history_path = self.log_metrics_file
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train ChromatinCNN')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--checkpoint_dir', type=str, default='phase3_model/checkpoints', help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')

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

    # Create model config
    phase3_config = config_dict.get('phase3', {})
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

    logger.info(f"Model: {model.__class__.__name__}")

    # Create data module
    data_config = config_dict['data']
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
    logger.info(f"Dataset sizes: {dataset_sizes}")

    # Create trainer
    trainer = Trainer(
        model=model,
        config=model_config,
        data_module=data_module,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        save_best_only=True,
    )

    # Resume from checkpoint if provided
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)

    # Resume from checkpoint if specified
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(
        num_epochs=args.num_epochs,
        early_stopping_patience=10,
        save_checkpoint=True,
    )


if __name__ == '__main__':
    main()

