"""Post-processing module for TorchSense pipelines.

The post-processor is implemented as both :class:`torch.nn.Module` and
:class:`lightning.pytorch.Callback`, following the anomalib design pattern.

During Lightning training, post-processing is applied via ``on_*_batch_end``
hooks.  For standalone / export use, it is applied via ``forward()``.
"""

from __future__ import annotations

from typing import Any

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from torch import nn


class PostProcessor(nn.Module, Callback):
    """Base TorchSense post-processor.

    Subclass this and override :meth:`post_process_batch` (Lightning hooks)
    and :meth:`forward` (standalone inference) to implement task-specific
    post-processing logic.
    """

    def post_process_batch(self, outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> None:
        """Apply post-processing to a batch of model outputs.

        The default implementation is a no-op; override in subclasses.

        Args:
            outputs: Model output dict (at minimum contains ``"prediction"``).
                     Modified **in-place**.
            batch: Original batch dict for reference (e.g. metadata).
        """

    # ------------------------------------------------------------------
    # Lightning Callback hooks
    # ------------------------------------------------------------------
    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, pl_module, batch_idx, dataloader_idx
        self.post_process_batch(outputs, batch)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, pl_module, batch_idx, dataloader_idx
        self.post_process_batch(outputs, batch)

    # ------------------------------------------------------------------
    # nn.Module forward – standalone / export inference
    # ------------------------------------------------------------------
    def forward(self, prediction: torch.Tensor) -> torch.Tensor:
        """Post-process a single prediction tensor.

        The default implementation is a pass-through.

        Args:
            prediction: Model prediction tensor.

        Returns:
            Post-processed tensor.
        """
        return prediction


class SignalPostProcessor(PostProcessor):
    """Post-processor for signal reconstruction tasks (e.g. SCG→ECG).

    Applies optional output clamping and value range normalization to
    reconstructed waveforms.

    Args:
        clamp_min: Lower clamp bound for the prediction. ``None`` disables.
        clamp_max: Upper clamp bound for the prediction. ``None`` disables.
    """

    def __init__(
        self,
        clamp_min: float | None = None,
        clamp_max: float | None = None,
    ) -> None:
        super().__init__()
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def post_process_batch(self, outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> None:
        del batch
        pred = outputs["prediction"]
        if self.clamp_min is not None or self.clamp_max is not None:
            outputs["prediction"] = pred.clamp(min=self.clamp_min, max=self.clamp_max)

    def forward(self, prediction: torch.Tensor) -> torch.Tensor:
        if self.clamp_min is not None or self.clamp_max is not None:
            return prediction.clamp(min=self.clamp_min, max=self.clamp_max)
        return prediction


class BPPostProcessor(PostProcessor):
    """Post-processor for blood pressure prediction tasks (e.g. SCG→BP).

    Clamps SBP/DBP predictions to physiologically plausible ranges and
    optionally applies a linear calibration offset.

    Args:
        sbp_range: ``(min, max)`` plausible SBP range in mmHg.
        dbp_range: ``(min, max)`` plausible DBP range in mmHg.
        sbp_offset: Additive calibration offset for SBP.
        dbp_offset: Additive calibration offset for DBP.
    """

    def __init__(
        self,
        sbp_range: tuple[float, float] = (60.0, 250.0),
        dbp_range: tuple[float, float] = (30.0, 150.0),
        sbp_offset: float = 0.0,
        dbp_offset: float = 0.0,
    ) -> None:
        super().__init__()
        self.sbp_range = sbp_range
        self.dbp_range = dbp_range
        self.sbp_offset = sbp_offset
        self.dbp_offset = dbp_offset

    def _clamp_bp(self, prediction: torch.Tensor) -> torch.Tensor:
        """Clamp and calibrate a ``(B, 2)`` tensor of [SBP, DBP]."""
        sbp = prediction[:, 0] + self.sbp_offset
        dbp = prediction[:, 1] + self.dbp_offset
        sbp = sbp.clamp(min=self.sbp_range[0], max=self.sbp_range[1])
        dbp = dbp.clamp(min=self.dbp_range[0], max=self.dbp_range[1])
        return torch.stack([sbp, dbp], dim=1)

    def post_process_batch(self, outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> None:
        del batch
        outputs["prediction"] = self._clamp_bp(outputs["prediction"])

    def forward(self, prediction: torch.Tensor) -> torch.Tensor:
        return self._clamp_bp(prediction)
