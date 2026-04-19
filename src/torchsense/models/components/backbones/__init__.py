"""Backbone architectures for TorchSense."""

from torchsense.models.components.backbones.resnet1d import ResNet1d
from torchsense.models.components.backbones.unet1d import UNet1d

__all__ = ["UNet1d", "ResNet1d"]
