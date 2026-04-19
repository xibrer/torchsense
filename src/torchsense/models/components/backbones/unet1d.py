"""1D U-Net backbone for signal-to-signal tasks (e.g. SCG -> ECG reconstruction).

Lightweight architecture (~364K params at default settings) following the
Vib2ECG benchmark design.
"""

from __future__ import annotations

import torch
from torch import nn

from torchsense.models.components.layers.conv_block import ConvBlock1d


class _EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = ConvBlock1d(in_ch, out_ch, kernel_size=kernel_size, dropout=dropout)
        self.conv2 = ConvBlock1d(out_ch, out_ch, kernel_size=kernel_size, dropout=dropout)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.conv2(x)
        return self.pool(x), x  # (downsampled, skip)


class _DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, dropout: float = 0.0) -> None:
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv1 = ConvBlock1d(out_ch * 2, out_ch, kernel_size=kernel_size, dropout=dropout)
        self.conv2 = ConvBlock1d(out_ch, out_ch, kernel_size=kernel_size, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # handle length mismatch from odd-length inputs
        if x.shape[-1] != skip.shape[-1]:
            x = nn.functional.pad(x, (0, skip.shape[-1] - x.shape[-1]))
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        return self.conv2(x)


class _Bottleneck(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = ConvBlock1d(in_ch, out_ch, kernel_size=kernel_size, dropout=dropout)
        self.conv2 = ConvBlock1d(out_ch, out_ch, kernel_size=kernel_size, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class UNet1d(nn.Module):
    """1D U-Net for signal-to-signal mapping.

    Args:
        in_channels: Number of input channels (e.g. 3 for tri-axis accelerometer).
        out_channels: Number of output channels (e.g. 1 for single-lead ECG).
        base_filters: Number of filters in the first encoder stage.
        depth: Number of encoder/decoder stages.
        kernel_size: Convolution kernel size.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_filters: int = 32,
        depth: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.depth = depth

        # Encoder
        self.encoders = nn.ModuleList()
        ch = in_channels
        for i in range(depth):
            out_ch = base_filters * (2 ** i)
            self.encoders.append(_EncoderBlock(ch, out_ch, kernel_size, dropout))
            ch = out_ch

        # Bottleneck
        bottleneck_ch = base_filters * (2 ** depth)
        self.bottleneck = _Bottleneck(ch, bottleneck_ch, kernel_size, dropout)

        # Decoder
        self.decoders = nn.ModuleList()
        ch = bottleneck_ch
        for i in range(depth - 1, -1, -1):
            out_ch = base_filters * (2 ** i)
            self.decoders.append(_DecoderBlock(ch, out_ch, kernel_size, dropout))
            ch = out_ch

        self.head = nn.Conv1d(ch, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return self.head(x)
