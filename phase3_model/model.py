"""
Interpretability-Friendly CNN Architecture for Chromatin State Prediction

Based on Phase 3.2 of the guide, implementing:
- Motif Detection Block with interpretable convolutional layers
- Sparse Bottleneck for SAE attachment
- Spatial Aggregation with global pooling
- Decision Block for classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ChromatinCNN(nn.Module):
    """
    Interpretability-friendly 1D CNN for chromatin state prediction.

    Architecture:
    1. Motif Detection Block - interpretable conv layers
    2. Sparse Bottleneck - SAE attachment point
    3. Spatial Aggregation - global pooling
    4. Decision Block - classification

    Designed for mechanistic interpretability with minimal polysemanticity.
    """

    def __init__(
        self,
        n_classes: int = 18,
        conv1_filters: int = 128,
        conv2_filters: int = 256,
        bottleneck_filters: int = 512,
        kernel1: int = 19,
        kernel2: int = 11,
        dropout_rate: float = 0.3,
        use_l1_regularization: bool = True,
    ):
        """
        Initialize the ChromatinCNN.

        Args:
            n_classes: Number of chromatin state classes (default: 18)
            conv1_filters: Number of filters in first conv layer
            conv2_filters: Number of filters in second conv layer
            bottleneck_filters: Number of filters in bottleneck layer
            kernel1: Kernel size for first conv layer (motif length)
            kernel2: Kernel size for second conv layer
            dropout_rate: Dropout probability in dense layers
            use_l1_regularization: Apply L1 regularization to first conv layer
        """
        super(ChromatinCNN, self).__init__()

        self.n_classes = n_classes
        self.use_l1 = use_l1_regularization

        # ════════════════════════════════════════════════════
        # MOTIF DETECTION BLOCK (Interpretable)
        # ════════════════════════════════════════════════════
        self.conv1 = nn.Conv1d(
            in_channels=4,  # A, C, G, T
            out_channels=conv1_filters,
            kernel_size=kernel1,
            padding='same',
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(conv1_filters)

        self.conv2 = nn.Conv1d(
            in_channels=conv1_filters,
            out_channels=conv2_filters,
            kernel_size=kernel2,
            padding='same',
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(conv2_filters)

        # ════════════════════════════════════════════════════
        # SPARSE BOTTLENECK (SAE Attachment Point)
        # ════════════════════════════════════════════════════
        self.bottleneck = nn.Conv1d(
            in_channels=conv2_filters,
            out_channels=bottleneck_filters,
            kernel_size=1,  # 1x1 convolution for position-wise feature mixing
            bias=False,
        )
        self.bn_bottleneck = nn.BatchNorm1d(bottleneck_filters)

        # ════════════════════════════════════════════════════
        # SPATIAL AGGREGATION
        # ════════════════════════════════════════════════════
        # Global Max Pooling - captures strongest activation per filter
        # Global Average Pooling - captures overall presence
        # Both provide complementary information
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # ════════════════════════════════════════════════════
        # DECISION BLOCK
        # ════════════════════════════════════════════════════
        # Concatenated features: bottleneck_filters * 2 (max + avg)
        dense_input_size = bottleneck_filters * 2

        self.dense1 = nn.Linear(dense_input_size, 512)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.dense2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.classifier = nn.Linear(256, n_classes)

        # Store for interpretability
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization for ReLU layers."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def get_first_conv_filters(self) -> torch.Tensor:
        """
        Extract first conv layer filters for motif analysis.

        Returns:
            Filter weights of shape (n_filters, 4, kernel_size)
        """
        return self.conv1.weight.detach()

    def get_l1_penalty(self) -> torch.Tensor:
        """
        Compute L1 regularization penalty for first conv layer.
        Encourages sparse, interpretable motifs.

        Returns:
            L1 penalty value
        """
        if not self.use_l1:
            return torch.tensor(0.0, device=self.conv1.weight.device)

        return torch.mean(torch.abs(self.conv1.weight))

    def forward(
        self,
        x: torch.Tensor,
        return_activations: bool = False,
        return_positions: bool = False,
    ) -> torch.Tensor or Tuple[torch.Tensor, dict]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, 200, 4) or (batch, 4, 200)
            return_activations: If True, return intermediate activations for interpretability
            return_positions: If True, return position information for max pooling

        Returns:
            Class logits of shape (batch, n_classes)
            Optionally returns dict of intermediate activations and position info
        """
        # Ensure channel-first format (batch, 4, 200)
        if x.shape[1] != 4:
            x = x.transpose(1, 2)  # (batch, 200, 4) -> (batch, 4, 200)

        activations = {}

        # ════════════════════════════════════════════════════
        # MOTIF DETECTION BLOCK
        # ════════════════════════════════════════════════════
        x1 = F.relu(self.bn1(self.conv1(x)))  # (batch, 128, 200)
        activations['conv1'] = x1

        x2 = F.relu(self.bn2(self.conv2(x1)))  # (batch, 256, 200)
        activations['conv2'] = x2

        # ════════════════════════════════════════════════════
        # SPARSE BOTTLENECK (SAE Attachment Point)
        # ════════════════════════════════════════════════════
        x_bottleneck = F.relu(self.bn_bottleneck(self.bottleneck(x2)))  # (batch, 512, 200)
        activations['bottleneck'] = x_bottleneck

        # ════════════════════════════════════════════════════
        # SPATIAL AGGREGATION
        # ════════════════════════════════════════════════════
        # Global Max Pooling
        x_max = self.global_max_pool(x_bottleneck)  # (batch, 512, 1)
        x_max = x_max.squeeze(-1)  # (batch, 512)

        # Store position information for interpretability
        positions = None
        if return_positions:
            # Find positions of max activations for each filter
            _, max_positions = torch.max(x_bottleneck, dim=2)  # (batch, 512)
            positions = max_positions

        # Global Average Pooling (parallel branch)
        x_avg = self.global_avg_pool(x_bottleneck)  # (batch, 512, 1)
        x_avg = x_avg.squeeze(-1)  # (batch, 512)

        # Concatenate
        x = torch.cat([x_max, x_avg], dim=1)  # (batch, 1024)

        # ════════════════════════════════════════════════════
        # DECISION BLOCK
        # ════════════════════════════════════════════════════
        x = F.relu(self.dense1(x))
        x = self.dropout1(x)
        activations['dense1'] = x

        x = F.relu(self.dense2(x))
        x = self.dropout2(x)
        activations['dense2'] = x

        logits = self.classifier(x)  # (batch, n_classes)

        if return_activations and return_positions:
            return logits, activations, positions
        elif return_activations:
            return logits, activations
        elif return_positions:
            return logits, None, positions
        else:
            return logits

    def predict_with_rc_consistency(
        self,
        x: torch.Tensor,
        return_probs: bool = False
    ) -> torch.Tensor:
        """
        Make predictions with reverse complement averaging.

        This improves RC equivariance by averaging predictions from both strands.

        Args:
            x: Input tensor of shape (batch, 200, 4)
            return_probs: If True, return probabilities instead of class predictions

        Returns:
            Predictions or probabilities of shape (batch, n_classes)
        """
        # Forward pass
        logits_orig = self(x)

        # Compute reverse complement
        x_rc = x.flip(dims=[1])[:, :, [3, 2, 1, 0]]  # Reverse and complement A<->T, C<->G
        logits_rc = self(x_rc)

        # Average predictions
        logits_avg = (logits_orig + logits_rc) / 2

        if return_probs:
            return F.softmax(logits_avg, dim=1)
        else:
            return torch.argmax(logits_avg, dim=1)


class ChromatinCNNConfig:
    """Configuration class for ChromatinCNN hyperparameters."""

    def __init__(
        self,
        n_classes: int = 18,
        conv1_filters: int = 128,
        conv2_filters: int = 256,
        bottleneck_filters: int = 512,
        kernel1: int = 19,
        kernel2: int = 11,
        dropout_rate: float = 0.3,
        use_l1_regularization: bool = True,
        l1_weight: float = 1e-5,
        label_smoothing: float = 0.05,
        learning_rate: float = 1e-3,
        warmup_epochs: int = 5,
    ):
        self.n_classes = n_classes
        self.conv1_filters = conv1_filters
        self.conv2_filters = conv2_filters
        self.bottleneck_filters = bottleneck_filters
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.dropout_rate = dropout_rate
        self.use_l1_regularization = use_l1_regularization
        self.l1_weight = l1_weight
        self.label_smoothing = label_smoothing
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return self.__dict__

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ChromatinCNNConfig":
        """Create config from dictionary."""
        return cls(**config_dict)




