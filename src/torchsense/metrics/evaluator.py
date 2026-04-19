"""Evaluator module for TorchSense, analogous to anomalib's Evaluator.

The Evaluator is both an ``nn.Module`` (so it travels with the model's
state-dict) and a Lightning ``Callback`` (so it hooks into the training
loop automatically).  Each ``TorchSenseModule`` holds an optional Evaluator
that is surfaced via ``configure_callbacks()``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import lightning.pytorch as pl
from lightning.pytorch import Callback, LightningModule, Trainer
from torch import nn
from torch.nn import ModuleList

from torchsense.metrics.base import TorchSenseMetric


class Evaluator(nn.Module, Callback):
    """Manages metric computation across validation and test stages.

    Args:
        val_metrics: Metrics computed during validation.
        test_metrics: Metrics computed during testing.  If ``None`` the
            *val_metrics* are reused for testing.
    """

    def __init__(
        self,
        val_metrics: Sequence[TorchSenseMetric] | TorchSenseMetric | None = None,
        test_metrics: Sequence[TorchSenseMetric] | TorchSenseMetric | None = None,
    ) -> None:
        super().__init__()
        self.val_metrics = ModuleList(self._to_list(val_metrics))
        self.test_metrics = ModuleList(self._to_list(test_metrics) or list(self.val_metrics))

    # ------------------------------------------------------------------
    # Validation hooks
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
        del trainer, pl_module, batch, batch_idx, dataloader_idx
        if outputs is None:
            return
        for metric in self.val_metrics:
            metric.update(outputs)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del trainer
        for metric in self.val_metrics:
            if metric.update_called:
                pl_module.log(f"val/{metric.name}", metric.compute(), prog_bar=True, on_epoch=True)
                metric.reset()

    # ------------------------------------------------------------------
    # Test hooks
    # ------------------------------------------------------------------
    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, pl_module, batch, batch_idx, dataloader_idx
        if outputs is None:
            return
        for metric in self.test_metrics:
            metric.update(outputs)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del trainer
        for metric in self.test_metrics:
            if metric.update_called:
                pl_module.log(f"test/{metric.name}", metric.compute(), on_epoch=True)
                metric.reset()

    # ------------------------------------------------------------------
    # nn.Module forward (no-op, required by the interface)
    # ------------------------------------------------------------------
    def forward(self) -> None:  # pragma: no cover
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_list(
        metrics: Sequence[TorchSenseMetric] | TorchSenseMetric | None,
    ) -> list[TorchSenseMetric]:
        if metrics is None:
            return []
        if isinstance(metrics, TorchSenseMetric):
            return [metrics]
        return list(metrics)
