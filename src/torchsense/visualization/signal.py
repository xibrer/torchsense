"""Signal visualization utilities for time-series data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from torchsense.visualization.base import Visualizer


# ======================================================================
# Standalone plotting functions (kept for direct use)
# ======================================================================

def plot_signal_comparison(
    input_signal: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    save_path: str | Path | None = None,
    fs: float = 1000.0,
    title: str = "Signal Reconstruction",
    figsize: tuple[int, int] = (14, 8),
) -> None:
    """Plot input SCG, predicted ECG, and ground truth ECG for visual comparison.

    Args:
        input_signal: Input SCG tensor of shape (N, C_in, T) or (C_in, T).
        prediction: Predicted ECG of shape (N, C_out, T) or (C_out, T).
        target: Ground truth ECG of shape (N, C_out, T) or (C_out, T).
        save_path: If set, save the figure to this path instead of showing.
        fs: Sampling frequency for time axis labeling.
        title: Figure title.
        figsize: Figure size (width, height).
    """
    import matplotlib.pyplot as plt

    if input_signal.dim() == 3:
        input_signal = input_signal[0]
    if prediction.dim() == 3:
        prediction = prediction[0]
    if target.dim() == 3:
        target = target[0]

    inp = input_signal.detach().cpu().numpy()
    pred = prediction.detach().cpu().numpy()
    tgt = target.detach().cpu().numpy()

    n_in = inp.shape[0]
    n_out = pred.shape[0]
    n_rows = n_in + n_out
    t = np.arange(inp.shape[-1]) / fs

    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)
    if n_rows == 1:
        axes = [axes]

    for i in range(n_in):
        axes[i].plot(t, inp[i], color="#2196F3", linewidth=0.5, alpha=0.8)
        axes[i].set_ylabel(f"SCG ch{i}")
        axes[i].set_title(f"{title} - Input Channel {i}" if i == 0 else "")

    for i in range(n_out):
        ax = axes[n_in + i]
        ax.plot(t, tgt[i], color="#4CAF50", linewidth=0.6, alpha=0.8, label="Ground Truth")
        ax.plot(t, pred[i], color="#FF5722", linewidth=0.6, alpha=0.8, label="Prediction")
        ax.set_ylabel(f"ECG lead{i}")
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_bp_prediction(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 5),
) -> None:
    """Scatter plot of predicted vs actual blood pressure values.

    Args:
        predictions: Predicted BP of shape (N, 2) -> [SBP, DBP].
        targets: Ground truth BP of shape (N, 2).
        save_path: If set, save figure to this path.
        figsize: Figure size.
    """
    import matplotlib.pyplot as plt

    pred = predictions.detach().cpu().numpy()
    tgt = targets.detach().cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for ax, idx, name in [(ax1, 0, "Systolic (SBP)"), (ax2, 1, "Diastolic (DBP)")]:
        ax.scatter(tgt[:, idx], pred[:, idx], alpha=0.5, s=10, color="#2196F3")
        lims = [min(tgt[:, idx].min(), pred[:, idx].min()) - 5, max(tgt[:, idx].max(), pred[:, idx].max()) + 5]
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5)
        ax.set_xlabel("Actual (mmHg)")
        ax.set_ylabel("Predicted (mmHg)")
        ax.set_title(name)
        ax.set_aspect("equal")
        mae = np.abs(pred[:, idx] - tgt[:, idx]).mean()
        ax.text(0.05, 0.95, f"MAE: {mae:.1f} mmHg", transform=ax.transAxes, va="top", fontsize=10)

    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ======================================================================
# Visualizer components (nn.Module + Callback)
# ======================================================================

class SignalVisualizer(Visualizer):
    """Visualizer for signal reconstruction tasks (e.g. SCG→ECG).

    Generates waveform comparison plots of input, prediction, and target.

    Args:
        output_dir: Save directory.
        num_samples: Max samples to visualize.
        fs: Sampling frequency for time axis.
    """

    def __init__(
        self,
        output_dir: str | Path = "./results/visualizations",
        num_samples: int = 4,
        fs: float = 1000.0,
    ) -> None:
        super().__init__(output_dir=output_dir, num_samples=num_samples)
        self.fs = fs

    def visualize(self, step_output: dict[str, Any], save_path: Path) -> None:
        if "prediction" not in step_output or "target" not in step_output:
            return
        plot_signal_comparison(
            input_signal=step_output.get("input", step_output["prediction"]),
            prediction=step_output["prediction"],
            target=step_output["target"],
            save_path=save_path,
            fs=self.fs,
        )


class BPVisualizer(Visualizer):
    """Visualizer for blood pressure prediction tasks (e.g. SCG→BP).

    Generates scatter plots of predicted vs actual blood pressure.
    """

    def visualize(self, step_output: dict[str, Any], save_path: Path) -> None:
        if "prediction" not in step_output or "target" not in step_output:
            return
        plot_bp_prediction(
            predictions=step_output["prediction"],
            targets=step_output["target"],
            save_path=save_path,
        )
