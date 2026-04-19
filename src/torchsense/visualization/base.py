"""Base Visualizer for TorchSense, analogous to anomalib's Visualizer.

Implemented as ``nn.Module + Callback`` so it can be registered as a
component of ``TorchSenseModule`` and automatically injected into the
Lightning Trainer via ``configure_callbacks()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from torch import nn


class Visualizer(nn.Module, Callback):
    """Base visualizer component.

    Collects a small number of samples from the first validation batch
    and generates visualizations at the end of every validation epoch.
    Subclass and override :meth:`visualize` to produce task-specific plots.

    Args:
        output_dir: Directory where figures are saved.
        num_samples: Max samples to collect per epoch.
    """

    def __init__(
        self,
        output_dir: str | Path = "./results/visualizations",
        num_samples: int = 4,
    ) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self._collected: list[dict[str, torch.Tensor]] = []

    # ------------------------------------------------------------------
    # Override in subclasses
    # ------------------------------------------------------------------
    def visualize(
        self,
        step_output: dict[str, Any],
        save_path: Path,
    ) -> None:
        """Generate and save a visualization.

        Args:
            step_output: Merged dict of model outputs and batch data.
            save_path: File path to save the figure.
        """

    # ------------------------------------------------------------------
    # Lightning Callback hooks
    # ------------------------------------------------------------------
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, pl_module, batch, dataloader_idx
        if batch_idx == 0 and outputs is not None:
            subset = {
                k: v[: self.num_samples].detach().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in outputs.items()
            }
            self._collected.append(subset)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if not self._collected:
            return
        try:
            save_dir = self.output_dir / f"epoch_{trainer.current_epoch}"
            save_dir.mkdir(parents=True, exist_ok=True)
            self.visualize(self._collected[0], save_dir / "visualization.png")
        finally:
            self._collected.clear()

    # ------------------------------------------------------------------
    # nn.Module forward (pass-through)
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
