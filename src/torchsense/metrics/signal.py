"""Signal reconstruction quality metrics."""

from __future__ import annotations

import torch
from torchmetrics import Metric


class SignalToNoiseRatio(Metric):
    """Signal-to-Noise Ratio (SNR) in dB for signal reconstruction.

    SNR = 10 * log10(||target||^2 / ||target - prediction||^2)

    Higher is better. Reports the mean SNR across the batch.
    """

    full_state_update: bool = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("snr_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        noise = target - preds
        signal_power = (target**2).sum(dim=-1)
        noise_power = (noise**2).sum(dim=-1)
        noise_power = torch.clamp(noise_power, min=1e-10)

        snr = 10.0 * torch.log10(signal_power / noise_power)
        self.snr_sum += snr.sum()
        self.count += snr.numel()

    def compute(self) -> torch.Tensor:
        return self.snr_sum / self.count


class PearsonCorrelation(Metric):
    """Pearson correlation coefficient averaged over the batch.

    Measures the linear correlation between predicted and target signals.
    Values range from -1 to 1, where 1 means perfect positive correlation.
    """

    full_state_update: bool = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("corr_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        # Flatten channel dim if present: (B, C, T) -> (B*C, T)
        if preds.dim() == 3:
            preds = preds.reshape(-1, preds.shape[-1])
            target = target.reshape(-1, target.shape[-1])

        p_mean = preds.mean(dim=-1, keepdim=True)
        t_mean = target.mean(dim=-1, keepdim=True)
        p_centered = preds - p_mean
        t_centered = target - t_mean

        num = (p_centered * t_centered).sum(dim=-1)
        denom = torch.sqrt((p_centered**2).sum(dim=-1) * (t_centered**2).sum(dim=-1))
        denom = torch.clamp(denom, min=1e-10)

        corr = num / denom
        self.corr_sum += corr.sum()
        self.count += corr.numel()

    def compute(self) -> torch.Tensor:
        return self.corr_sum / self.count
