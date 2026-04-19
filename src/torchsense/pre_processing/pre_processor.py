"""Pre-processing module for TorchSense pipelines.

The pre-processor is implemented as both :class:`torch.nn.Module` and
:class:`lightning.pytorch.Callback` to support both standalone inference
and Lightning training workflows, following the anomalib design pattern.

During Lightning training, transforms are applied via ``on_*_batch_start``
hooks.  For standalone / export use, transforms are applied via ``forward()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable


class PreProcessor(nn.Module, Callback):
    """TorchSense pre-processor.

    Applies a callable transform to the ``"input"`` tensor of every batch
    before it reaches the model.  When no transform is provided the batch
    passes through unchanged.

    Args:
        transform: A callable ``(torch.Tensor) -> torch.Tensor`` applied to
            ``batch["input"]``.  Typically one of the transforms from
            :mod:`torchsense.data.transforms.imu` or a ``Compose`` pipeline.
    """

    def __init__(self, transform: Callable[[torch.Tensor], torch.Tensor] | None = None) -> None:
        super().__init__()
        self.transform = transform

    # ------------------------------------------------------------------
    # Lightning Callback hooks
    # ------------------------------------------------------------------
    def _apply_transform(self, batch: dict[str, Any]) -> None:
        if self.transform is not None:
            batch["input"] = self.transform(batch["input"])

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: dict[str, Any], batch_idx: int,
    ) -> None:
        del trainer, pl_module, batch_idx
        self._apply_transform(batch)

    def on_validation_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: dict[str, Any], batch_idx: int,
    ) -> None:
        del trainer, pl_module, batch_idx
        self._apply_transform(batch)

    def on_test_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, pl_module, batch_idx, dataloader_idx
        self._apply_transform(batch)

    def on_predict_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, pl_module, batch_idx, dataloader_idx
        self._apply_transform(batch)

    # ------------------------------------------------------------------
    # nn.Module forward – used for standalone / export inference
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the transform to a raw input tensor.

        Args:
            x: Input signal tensor.

        Returns:
            Transformed tensor.
        """
        if self.transform is not None:
            return self.transform(x)
        return x
