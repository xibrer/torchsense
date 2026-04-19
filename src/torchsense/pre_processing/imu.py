"""Pre-processing utilities for IMU / SCG signals."""

from __future__ import annotations

import numpy as np
import torch
from scipy import signal as sp_signal


def bandpass_filter(
    data: np.ndarray,
    low_freq: float = 0.5,
    high_freq: float = 40.0,
    fs: float = 1000.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter.

    Args:
        data: Input signal array of shape (..., T).
        low_freq: Lower cutoff frequency in Hz.
        high_freq: Upper cutoff frequency in Hz.
        fs: Sampling frequency in Hz.
        order: Filter order.

    Returns:
        Filtered signal with the same shape.
    """
    nyq = fs / 2.0
    b, a = sp_signal.butter(order, [low_freq / nyq, high_freq / nyq], btype="band")
    return sp_signal.filtfilt(b, a, data, axis=-1).astype(data.dtype)


def normalize_signal(data: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Normalize a signal along the time axis.

    Args:
        data: Signal of shape (..., T).
        method: 'zscore' for zero-mean unit-variance, 'minmax' for [0, 1] range.
    """
    if method == "zscore":
        mean = data.mean(axis=-1, keepdims=True)
        std = data.std(axis=-1, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        return (data - mean) / std
    elif method == "minmax":
        vmin = data.min(axis=-1, keepdims=True)
        vmax = data.max(axis=-1, keepdims=True)
        denom = np.where(vmax == vmin, 1.0, vmax - vmin)
        return (data - vmin) / denom
    else:
        msg = f"Unknown normalization method: {method}"
        raise ValueError(msg)


def segment_signal(
    data: np.ndarray,
    window_size: int,
    stride: int | None = None,
) -> np.ndarray:
    """Segment a signal into fixed-length windows using a sliding window.

    Args:
        data: Signal of shape (C, T).
        window_size: Length of each segment.
        stride: Step between consecutive windows. Defaults to window_size (no overlap).

    Returns:
        Array of shape (N, C, window_size) where N is the number of segments.
    """
    if stride is None:
        stride = window_size

    _, length = data.shape
    starts = range(0, length - window_size + 1, stride)
    segments = np.stack([data[:, s : s + window_size] for s in starts], axis=0)
    return segments


def preprocess_scg(
    scg: np.ndarray,
    fs: float = 1000.0,
    low_freq: float = 0.5,
    high_freq: float = 40.0,
    normalize: str = "zscore",
) -> np.ndarray:
    """Full SCG preprocessing pipeline: filter -> normalize.

    Args:
        scg: Raw SCG signal of shape (C, T).
        fs: Sampling rate in Hz.
        low_freq: Bandpass lower cutoff.
        high_freq: Bandpass upper cutoff.
        normalize: Normalization method.
    """
    scg = bandpass_filter(scg, low_freq, high_freq, fs)
    scg = normalize_signal(scg, normalize)
    return scg


def numpy_to_tensor(data: np.ndarray) -> torch.Tensor:
    """Convert numpy array to float32 torch tensor."""
    return torch.from_numpy(data.copy()).float()
