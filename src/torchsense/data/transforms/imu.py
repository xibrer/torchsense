"""Transforms for IMU / SCG signal data augmentation."""

from __future__ import annotations

import torch


class AddGaussianNoise:
    """Add Gaussian noise to a signal tensor."""

    def __init__(self, std: float = 0.01) -> None:
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std


class RandomScale:
    """Randomly scale the signal amplitude."""

    def __init__(self, low: float = 0.8, high: float = 1.2) -> None:
        self.low = low
        self.high = high

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.empty(1).uniform_(self.low, self.high).item()
        return x * scale


class RandomTimeShift:
    """Randomly shift the signal along the time axis (circular)."""

    def __init__(self, max_shift: int = 50) -> None:
        self.max_shift = max_shift

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
        return torch.roll(x, shifts=int(shift), dims=-1)


class Compose:
    """Compose multiple transforms sequentially."""

    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x
