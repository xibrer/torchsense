"""Post-processing utilities for reconstructed ECG signals."""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal


def smooth_ecg(ecg: np.ndarray, window_length: int = 11, polyorder: int = 3) -> np.ndarray:
    """Apply Savitzky-Golay smoothing to a reconstructed ECG signal.

    Args:
        ecg: ECG signal of shape (..., T).
        window_length: Filter window length (must be odd).
        polyorder: Polynomial order for the filter.
    """
    return sp_signal.savgol_filter(ecg, window_length, polyorder, axis=-1)


def detect_r_peaks(
    ecg: np.ndarray,
    fs: float = 1000.0,
    min_distance_ms: float = 300.0,
    height_factor: float = 0.5,
) -> np.ndarray:
    """Detect R-peaks in a single-lead ECG signal.

    Args:
        ecg: 1D ECG signal of shape (T,).
        fs: Sampling frequency.
        min_distance_ms: Minimum distance between peaks in ms.
        height_factor: Peak height threshold as mean + factor * std.

    Returns:
        Array of R-peak sample indices.
    """
    min_distance = int(min_distance_ms * fs / 1000.0)
    height = ecg.mean() + height_factor * ecg.std()
    peaks, _ = sp_signal.find_peaks(ecg, height=height, distance=min_distance)
    return peaks


def compute_heart_rate(r_peaks: np.ndarray, fs: float = 1000.0) -> float:
    """Compute heart rate in BPM from R-peak locations.

    Args:
        r_peaks: Array of R-peak sample indices.
        fs: Sampling frequency.

    Returns:
        Heart rate in beats per minute.
    """
    if len(r_peaks) < 2:
        return 0.0
    rr_intervals = np.diff(r_peaks) / fs
    mean_rr = rr_intervals.mean()
    return 60.0 / mean_rr if mean_rr > 0 else 0.0
