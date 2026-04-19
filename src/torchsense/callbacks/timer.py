"""Timer callback for tracking training duration."""

from __future__ import annotations

import time

import lightning.pytorch as pl

from torchsense.utils.logging import get_logger

logger = get_logger(__name__)


class TimerCallback(pl.Callback):
    """Logs elapsed time at the end of each training epoch and full training."""

    def __init__(self) -> None:
        super().__init__()
        self._train_start: float = 0.0
        self._epoch_start: float = 0.0

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._train_start = time.perf_counter()

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._epoch_start = time.perf_counter()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        elapsed = time.perf_counter() - self._epoch_start
        logger.info(f"Epoch {trainer.current_epoch} completed in {elapsed:.1f}s")

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        total = time.perf_counter() - self._train_start
        minutes, seconds = divmod(total, 60)
        logger.info(f"Training completed in {int(minutes)}m {seconds:.1f}s")
