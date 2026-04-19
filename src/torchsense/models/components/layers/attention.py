"""1D attention layers for signal processing models."""

from __future__ import annotations

import torch
from torch import nn


class SqueezeExcitation1d(nn.Module):
    """Squeeze-and-Excitation block for 1D signals.

    Adaptively recalibrates channel-wise feature responses by
    modelling channel interdependencies.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.se(x).unsqueeze(-1)
        return x * scale
