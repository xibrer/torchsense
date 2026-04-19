"""1D convolutional building blocks."""

from __future__ import annotations

from torch import nn


class ConvBlock1d(nn.Module):
    """Conv1d -> BatchNorm1d -> Activation, with optional residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | str | None = None,
        dilation: int = 1,
        groups: int = 1,
        activation: str = "relu",
        dropout: float = 0.0,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.residual = residual

        if padding is None:
            padding = "same" if stride == 1 else (kernel_size // 2)

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = _get_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.skip = nn.Identity()
        if residual and (in_channels != out_channels or stride != 1):
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.dropout(self.act(self.bn(self.conv(x))))
        if self.residual:
            out = out + self.skip(identity)
        return out


def _get_activation(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU(inplace=True),
        "gelu": nn.GELU(),
        "leaky_relu": nn.LeakyReLU(0.1, inplace=True),
        "elu": nn.ELU(inplace=True),
        "silu": nn.SiLU(inplace=True),
    }
    if name not in activations:
        msg = f"Unknown activation: {name}. Choose from {list(activations.keys())}"
        raise ValueError(msg)
    return activations[name]
