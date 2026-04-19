"""1D ResNet backbone for classification and regression tasks on time-series signals."""

from __future__ import annotations

import torch
from torch import nn

from torchsense.models.components.layers.attention import SqueezeExcitation1d
from torchsense.models.components.layers.conv_block import ConvBlock1d


class _ResidualBlock1d(nn.Module):
    """Two-layer residual block with optional SE attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        dropout: float = 0.0,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = ConvBlock1d(in_channels, out_channels, kernel_size, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size, padding="same", bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.se = SqueezeExcitation1d(out_channels) if use_se else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.skip = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        out = self.dropout(out)
        return self.act(out + identity)


class ResNet1d(nn.Module):
    """1D ResNet for time-series classification or feature extraction.

    Args:
        in_channels: Number of input channels.
        base_filters: Filters in the first stage.
        num_blocks: Number of residual blocks per stage.
        num_stages: Number of stages (each doubles the filter count and halves length).
        kernel_size: Convolution kernel size.
        dropout: Dropout rate.
        use_se: Whether to use Squeeze-and-Excitation attention.
        output_dim: If set, adds a global pooling + linear head. If None, returns feature maps.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_filters: int = 64,
        num_blocks: int = 2,
        num_stages: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.1,
        use_se: bool = True,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()

        self.stem = ConvBlock1d(in_channels, base_filters, kernel_size=15, stride=2)

        stages = []
        ch = base_filters
        for s in range(num_stages):
            out_ch = base_filters * (2 ** s)
            for b in range(num_blocks):
                stride = 2 if (b == 0 and s > 0) else 1
                in_ch = ch if b == 0 else out_ch
                stages.append(_ResidualBlock1d(in_ch, out_ch, kernel_size, stride, dropout, use_se))
            ch = out_ch
        self.stages = nn.Sequential(*stages)
        self.feature_dim = ch

        self.head: nn.Module
        if output_dim is not None:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(ch, output_dim),
            )
        else:
            self.head = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        return self.head(x)
