"""Pure PyTorch network for SCG-to-ECG reconstruction.

Wraps the UNet1d backbone with task-specific configuration.
"""

from __future__ import annotations

import torch
from torch import nn

from torchsense.models.components.backbones.unet1d import UNet1d


class SCG2ECGModel(nn.Module):
    """SCG-to-ECG reconstruction network.

    Maps multi-axis accelerometer (SCG) signals to ECG waveforms
    using a 1D U-Net architecture.

    Args:
        in_channels: Number of SCG input channels (axes).
        out_channels: Number of ECG output leads.
        base_filters: Base filter count for U-Net.
        depth: Number of U-Net encoder/decoder stages.
        kernel_size: Convolution kernel size.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_filters: int = 32,
        depth: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.unet = UNet1d(
            in_channels=in_channels,
            out_channels=out_channels,
            base_filters=base_filters,
            depth=depth,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: SCG input tensor of shape (B, C_in, T).

        Returns:
            Reconstructed ECG of shape (B, C_out, T).
        """
        return self.unet(x)
