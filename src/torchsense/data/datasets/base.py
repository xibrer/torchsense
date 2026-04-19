"""Base dataset class for TorchSense, analogous to anomalib's AnomalibDataset.

``TorchSenseDataset`` separates the ``Dataset`` layer from the ``DataModule``
layer.  Subclasses define how individual samples are fetched, while the
``DataModule`` handles lifecycle (loading, splitting, DataLoader creation).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class TorchSenseDataset(Dataset):
    """Base class for all TorchSense datasets.

    Subclasses must implement :meth:`__getitem__` and :meth:`__len__`.
    They should also expose :attr:`subject_ids` if subject-based
    splitting is needed, and implement :meth:`subset` for the
    ``DataModule`` split logic.

    Args:
        transform: Optional callable applied to the ``"input"`` tensor
            in :meth:`__getitem__`.
    """

    def __init__(self, transform: Any | None = None) -> None:
        super().__init__()
        self.transform = transform

    @property
    def subject_ids(self) -> np.ndarray | None:
        """Per-sample subject identifiers used for subject-based splitting.

        Returns:
            Array of shape ``(N,)`` with an ID per sample, or ``None``
            if subject information is unavailable.
        """
        return None

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a single sample as a dict with at least ``"input"`` and ``"target"``."""

    @abstractmethod
    def __len__(self) -> int:
        """Total number of samples."""

    def subset(self, indices: np.ndarray) -> TorchSenseDataset:
        """Create a new dataset containing only the given sample indices.

        Must be implemented by subclasses that support DataModule-level
        splitting.

        Args:
            indices: 1-D integer array of sample indices to keep.

        Returns:
            A new ``TorchSenseDataset`` instance with the selected samples.
        """
        msg = f"{self.__class__.__name__} does not implement subset(). Implement it to enable DataModule-level splitting."
        raise NotImplementedError(msg)
