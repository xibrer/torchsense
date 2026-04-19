"""Vib2ECG data module for SCG-to-ECG reconstruction.

Reference: https://arxiv.org/abs/2603.15539
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from torchsense.data.datamodules.base import TorchSenseDataModule
from torchsense.data.datasets.imu.vib2ecg import Vib2ECGDataset
from torchsense.pre_processing.imu import (
    bandpass_filter,
    normalize_signal,
    segment_signal,
)


class Vib2ECGDataModule(TorchSenseDataModule):
    """DataModule for the Vib2ECG dataset.

    Loads paired SCG-ECG data from an HDF5 file, applies preprocessing
    (bandpass filter, normalization, segmentation), and delegates
    train/val/test splitting to the base class via :class:`TestSplitMode`
    and :class:`ValSplitMode`.

    Args:
        data_root: Path to the directory containing the HDF5 file.
        filename: Name of the HDF5 file.
        scg_axes: Which IMU axes to use as SCG channels (0=x, 1=y, 2=z).
            Default ``[2]`` uses z-axis (perpendicular to chest).
        ecg_leads: Which ECG leads to reconstruct. Default ``[0]`` for V1.
        window_size: Segment length in samples.
        stride: Sliding window stride.
        fs: Sampling frequency in Hz.
        low_freq: Bandpass filter lower cutoff.
        high_freq: Bandpass filter upper cutoff.
        normalize: Normalization method (``'zscore'`` or ``'minmax'``).
        train_transform: Optional transform applied to SCG during training.
    """

    def __init__(
        self,
        data_root: str | Path,
        filename: str = "vib2ecg.h5",
        scg_axes: list[int] | None = None,
        ecg_leads: list[int] | None = None,
        window_size: int = 2048,
        stride: int = 1024,
        fs: float = 1000.0,
        low_freq: float = 0.5,
        high_freq: float = 40.0,
        normalize: str = "zscore",
        train_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("test_split_mode", "by_subject")
        kwargs.setdefault("test_subjects", [15, 16])
        kwargs.setdefault("val_split_mode", "from_train")
        kwargs.setdefault("val_split_ratio", 0.15)
        super().__init__(data_root=data_root, **kwargs)

        self.filename = filename
        self.scg_axes = scg_axes or [2]
        self.ecg_leads = ecg_leads or [0]
        self.window_size = window_size
        self.stride = stride
        self.fs = fs
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.norm_method = normalize
        self.train_transform = train_transform

    def _setup(self) -> None:
        """Load HDF5, preprocess, segment, and create the full dataset."""
        h5_path = self.data_root / self.filename
        if not h5_path.exists():
            msg = (
                f"Vib2ECG HDF5 file not found at {h5_path}. "
                "Download from https://ieee-dataport.org/documents/"
                "vib2ecg-paired-chest-lead-scg-ecg-dataset-and-benchmark-ecg-reconstruction"
            )
            raise FileNotFoundError(msg)

        scg_segs, ecg_segs, subjects, leads = self._load_and_preprocess(h5_path)
        self._full_dataset = Vib2ECGDataset(
            scg_segments=scg_segs,
            ecg_segments=ecg_segs,
            subject_ids_arr=subjects,
            lead_indices=leads,
            transform=self.train_transform,
        )

    def _load_and_preprocess(
        self, h5_path: Path,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load HDF5, apply preprocessing, and segment signals."""
        all_scg_segs: list[np.ndarray] = []
        all_ecg_segs: list[np.ndarray] = []
        all_subjects: list[np.ndarray] = []
        all_leads: list[np.ndarray] = []

        with h5py.File(h5_path, "r") as f:
            imu_data = np.array(f["imu_data_1kHz"])
            ecg_data = np.array(f["PCG_data_1kHz"])

        n_subjects, n_leads = imu_data.shape[0], imu_data.shape[1]

        for subj in range(n_subjects):
            for lead in range(n_leads):
                scg = imu_data[subj, lead, self.scg_axes, :]
                ecg = ecg_data[subj, lead, self.ecg_leads, :]

                scg = bandpass_filter(scg, self.low_freq, self.high_freq, self.fs)
                scg = normalize_signal(scg, self.norm_method)
                ecg = normalize_signal(ecg, self.norm_method)

                scg_seg = segment_signal(scg, self.window_size, self.stride)
                ecg_seg = segment_signal(ecg, self.window_size, self.stride)

                n_segs = len(scg_seg)
                all_scg_segs.append(scg_seg)
                all_ecg_segs.append(ecg_seg)
                all_subjects.append(np.full(n_segs, subj))
                all_leads.append(np.full(n_segs, lead))

        return (
            np.concatenate(all_scg_segs, axis=0),
            np.concatenate(all_ecg_segs, axis=0),
            np.concatenate(all_subjects, axis=0),
            np.concatenate(all_leads, axis=0),
        )
