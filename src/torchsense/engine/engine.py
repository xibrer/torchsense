"""Training engine that orchestrates the TorchSense workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar

from torchsense.callbacks.timer import TimerCallback
from torchsense.data.datamodules.base import TorchSenseDataModule
from torchsense.models.components.base.torchsense_module import TorchSenseModule
from torchsense.utils.logging import get_logger

logger = get_logger(__name__)


class Engine:
    """High-level training engine wrapping Lightning Trainer.

    Provides a simplified API for training, validation, testing, prediction,
    and model export with sensible defaults.
    """

    def __init__(
        self,
        max_epochs: int = 100,
        accelerator: str = "auto",
        devices: int | str = "auto",
        default_root_dir: str | Path = "./results",
        early_stopping_patience: int | None = 10,
        **trainer_kwargs: Any,
    ) -> None:
        self.max_epochs = max_epochs
        self.accelerator = accelerator
        self.devices = devices
        self.default_root_dir = Path(default_root_dir)
        self.early_stopping_patience = early_stopping_patience
        self._trainer_kwargs = trainer_kwargs
        self._trainer: pl.Trainer | None = None

    def _build_callbacks(self) -> list[pl.Callback]:
        callbacks: list[pl.Callback] = [
            ModelCheckpoint(
                dirpath=self.default_root_dir / "checkpoints",
                filename="best-{epoch}-{val/loss:.4f}",
                monitor="val/loss",
                mode="min",
                save_top_k=3,
                save_last=True,
            ),
            RichProgressBar(),
            TimerCallback(),
        ]
        if self.early_stopping_patience is not None:
            callbacks.append(
                EarlyStopping(monitor="val/loss", patience=self.early_stopping_patience, mode="min")
            )
        return callbacks

    def _get_trainer(self) -> pl.Trainer:
        if self._trainer is None:
            self._trainer = pl.Trainer(
                max_epochs=self.max_epochs,
                accelerator=self.accelerator,
                devices=self.devices,
                default_root_dir=str(self.default_root_dir),
                callbacks=self._build_callbacks(),
                **self._trainer_kwargs,
            )
        return self._trainer

    # ------------------------------------------------------------------
    # Core workflows
    # ------------------------------------------------------------------
    def fit(self, model: TorchSenseModule, datamodule: TorchSenseDataModule, **kwargs: Any) -> None:
        """Train the model."""
        logger.info("Starting training: model=%s, task=%s", model.name, model.task_type.value)
        trainer = self._get_trainer()
        trainer.fit(model, datamodule=datamodule, **kwargs)

    def validate(self, model: TorchSenseModule, datamodule: TorchSenseDataModule, **kwargs: Any) -> list[dict]:
        """Validate the model."""
        trainer = self._get_trainer()
        return trainer.validate(model, datamodule=datamodule, **kwargs)

    def test(self, model: TorchSenseModule, datamodule: TorchSenseDataModule, **kwargs: Any) -> list[dict]:
        """Test the model."""
        trainer = self._get_trainer()
        return trainer.test(model, datamodule=datamodule, **kwargs)

    def predict(self, model: TorchSenseModule, datamodule: TorchSenseDataModule, **kwargs: Any) -> list:
        """Run prediction."""
        trainer = self._get_trainer()
        return trainer.predict(model, datamodule=datamodule, **kwargs)

    def train(self, model: TorchSenseModule, datamodule: TorchSenseDataModule, **kwargs: Any) -> list[dict]:
        """Convenience method: ``fit`` followed by ``test``.

        Returns:
            Test results from ``trainer.test()``.
        """
        self.fit(model, datamodule=datamodule, **kwargs)
        return self.test(model, datamodule=datamodule)

    def export(
        self,
        model: TorchSenseModule,
        output_path: str | Path | None = None,
        input_shape: tuple[int, ...] = (1, 1, 2048),
        **kwargs: Any,
    ) -> Path:
        """Export the model to ONNX format.

        Args:
            model: Trained TorchSenseModule.
            output_path: Destination path.  Defaults to
                ``<default_root_dir>/export/<model_name>.onnx``.
            input_shape: Dummy input shape for tracing.
            **kwargs: Forwarded to :meth:`TorchSenseModule.export_onnx`.

        Returns:
            Path to the exported file.
        """
        if output_path is None:
            output_path = self.default_root_dir / "export" / f"{model.name}.onnx"
        return model.export_onnx(output_path, input_shape=input_shape, **kwargs)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config_path: str | Path, **overrides: Any) -> tuple[Engine, TorchSenseModule]:
        """Create an Engine and model from a YAML configuration file.

        Args:
            config_path: Path to a LightningCLI-compatible YAML config that
                specifies both ``model`` and optionally ``trainer`` sections.
            **overrides: Key-value overrides on top of the config.

        Returns:
            ``(engine, model)`` tuple ready for ``engine.fit(model, ...)``.
        """
        model = TorchSenseModule.from_config(config_path, **overrides)
        engine = cls()
        return engine, model
