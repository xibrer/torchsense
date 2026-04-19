"""Pure PyTorch network for SCG-to-blood-pressure prediction.

Uses a ResNet1D backbone with a dual-output regression head to predict
systolic (SBP) and diastolic (DBP) blood pressure from SCG signals.
"""

from __future__ import annotations

import torch
from torch import nn

from torchsense.models.components.backbones.resnet1d import ResNet1d


class SCG2BPModel(nn.Module):
    """SCG-to-BP prediction network.

    Args:
        in_channels: Number of SCG input channels.
        base_filters: Base filter count for ResNet1D.
        num_blocks: Residual blocks per stage.
        num_stages: Number of downsampling stages.
        kernel_size: Convolution kernel size.
        dropout: Dropout rate.
        use_se: Use Squeeze-and-Excitation attention.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_filters: int = 64,
        num_blocks: int = 2,
        num_stages: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.2,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = ResNet1d(
            in_channels=in_channels,
            base_filters=base_filters,
            num_blocks=num_blocks,
            num_stages=num_stages,
            kernel_size=kernel_size,
            dropout=dropout,
            use_se=use_se,
            output_dim=None,  # feature maps only
        )

        feature_dim = base_filters * (2 ** (num_stages - 1))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 2),  # [SBP, DBP]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: SCG input of shape (B, C, T).

        Returns:
            BP predictions of shape (B, 2) -> [SBP, DBP] in mmHg.
        """
        features = self.backbone(x)
        pooled = self.pool(features)
        return self.head(pooled)
