"""Vib2ECG dataset for SCG-to-ECG reconstruction.

The Vib2ECG dataset contains paired multi-channel SCG (from IMU z-axis) and
12-lead ECG signals recorded at 1kHz from 17 subjects at 6 chest-lead positions.

Reference: https://arxiv.org/abs/2603.15539
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from torchsense.data.datasets.base import TorchSenseDataset
from torchsense.pre_processing.imu import numpy_to_tensor


class Vib2ECGDataset(TorchSenseDataset):
    """PyTorch Dataset for Vib2ECG SCG-ECG pairs.

    Each sample is a dict with:
        - ``"input"``: SCG tensor of shape ``(C_in, window_size)``
        - ``"target"``: ECG tensor of shape ``(C_out, window_size)``
        - ``"subject_id"``: int
        - ``"lead_idx"``: int

    Args:
        scg_segments: SCG array of shape ``(N, C_in, T)``.
        ecg_segments: ECG array of shape ``(N, C_out, T)``.
        subject_ids_arr: Subject ID per segment of shape ``(N,)``.
        lead_indices: Lead index per segment of shape ``(N,)``.
        transform: Optional callable applied to the input tensor.
    """

    def __init__(
        self,
        scg_segments: np.ndarray,
        ecg_segments: np.ndarray,
        subject_ids_arr: np.ndarray,
        lead_indices: np.ndarray,
        transform: Any | None = None,
    ) -> None:
        super().__init__(transform=transform)
        self._scg = scg_segments
        self._ecg = ecg_segments
        self._subject_ids = subject_ids_arr
        self._lead_indices = lead_indices

    @property
    def subject_ids(self) -> np.ndarray:
        return self._subject_ids

    def __len__(self) -> int:
        return len(self._scg)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        scg = numpy_to_tensor(self._scg[idx])
        ecg = numpy_to_tensor(self._ecg[idx])

        if self.transform is not None:
            scg = self.transform(scg)

        return {
            "input": scg,
            "target": ecg,
            "subject_id": int(self._subject_ids[idx]),
            "lead_idx": int(self._lead_indices[idx]),
        }

    def subset(self, indices: np.ndarray) -> Vib2ECGDataset:
        return Vib2ECGDataset(
            scg_segments=self._scg[indices],
            ecg_segments=self._ecg[indices],
            subject_ids_arr=self._subject_ids[indices],
            lead_indices=self._lead_indices[indices],
            transform=self.transform,
        )
