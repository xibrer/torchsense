"""Cardiac-specific metrics for ECG signal evaluation."""

from __future__ import annotations

import torch
from torchmetrics import Metric


class QRSDetectionAccuracy(Metric):
    """R-peak detection accuracy comparing predicted vs. ground truth ECG.

    Uses a simple threshold-based peak detection to find R-peaks in both
    the predicted and target ECG, then computes precision and recall
    with a tolerance window.

    Args:
        tolerance_ms: Matching tolerance in milliseconds.
        fs: Sampling frequency in Hz.
        min_distance_ms: Minimum distance between peaks in milliseconds.
    """

    full_state_update: bool = False

    def __init__(
        self,
        tolerance_ms: float = 50.0,
        fs: float = 1000.0,
        min_distance_ms: float = 300.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tolerance_samples = int(tolerance_ms * fs / 1000.0)
        self.min_distance = int(min_distance_ms * fs / 1000.0)

        self.add_state("true_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update with a batch of predicted and target ECG signals.

        Args:
            preds: Predicted ECG of shape (B, 1, T) or (B, T).
            target: Ground truth ECG of shape (B, 1, T) or (B, T).
        """
        if preds.dim() == 3:
            preds = preds.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)

        for i in range(preds.shape[0]):
            pred_peaks = self._find_peaks(preds[i])
            true_peaks = self._find_peaks(target[i])
            tp, fp, fn = self._match_peaks(pred_peaks, true_peaks)
            self.true_positives += tp
            self.false_positives += fp
            self.false_negatives += fn

    def compute(self) -> dict[str, torch.Tensor]:
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-10)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        return {"precision": precision, "recall": recall, "f1": f1}

    def _find_peaks(self, signal: torch.Tensor) -> torch.Tensor:
        """Simple peak detection: local maxima above mean + 0.5 * std."""
        sig = signal.detach().cpu()
        threshold = sig.mean() + 0.5 * sig.std()

        # Local maxima: signal[i] > signal[i-1] and signal[i] > signal[i+1]
        left = sig[:-2]
        center = sig[1:-1]
        right = sig[2:]
        is_peak = (center > left) & (center > right) & (center > threshold)

        peak_indices = torch.where(is_peak)[0] + 1  # offset by 1

        # Enforce minimum distance
        if len(peak_indices) > 1:
            keep = [peak_indices[0]]
            for idx in peak_indices[1:]:
                if idx - keep[-1] >= self.min_distance:
                    keep.append(idx)
            peak_indices = torch.stack(keep)

        return peak_indices

    def _match_peaks(
        self, pred_peaks: torch.Tensor, true_peaks: torch.Tensor
    ) -> tuple[int, int, int]:
        """Match predicted peaks to ground truth within tolerance."""
        if len(pred_peaks) == 0 and len(true_peaks) == 0:
            return 0, 0, 0
        if len(pred_peaks) == 0:
            return 0, 0, len(true_peaks)
        if len(true_peaks) == 0:
            return 0, len(pred_peaks), 0

        matched = set()
        tp = 0
        for pp in pred_peaks:
            dists = torch.abs(true_peaks - pp)
            min_dist, min_idx = dists.min(dim=0)
            if min_dist <= self.tolerance_samples and min_idx.item() not in matched:
                tp += 1
                matched.add(min_idx.item())

        fp = len(pred_peaks) - tp
        fn = len(true_peaks) - tp
        return tp, fp, fn
