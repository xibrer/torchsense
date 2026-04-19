"""Base module for all TorchSense models, analogous to anomalib's AnomalibModule.

``TorchSenseModule`` provides:
- Four pluggable sub-components: **pre_processor**, **post_processor**,
  **evaluator**, **visualizer** (each ``nn.Module + Callback``).
- A unified ``forward(x)`` export pipeline:
  ``pre_processor → model → post_processor``.
- ``configure_callbacks()`` that automatically registers all components.
- ``_resolve_component`` supporting ``bool | Instance | None`` three-state config.
- ``from_config()`` class method for YAML-based instantiation.
- ONNX export via :class:`ExportMixin`.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch import Callback
from torch import nn

from torchsense import ModalityType, TaskType
from torchsense.metrics.evaluator import Evaluator
from torchsense.models.components.base.export_mixin import ExportMixin
from torchsense.post_processing import PostProcessor
from torchsense.pre_processing import PreProcessor
from torchsense.visualization.base import Visualizer

logger = logging.getLogger(__name__)


class TorchSenseModule(ExportMixin, pl.LightningModule):
    """Base class for all TorchSense models.

    Args:
        learning_rate: Optimizer learning rate.
        weight_decay: Optimizer weight decay.
        optimizer: Optimizer name (``'adam'``, ``'adamw'``, ``'sgd'``).
        scheduler: LR scheduler name or ``None``.
        scheduler_kwargs: Extra scheduler arguments.
        pre_processor: Pre-processor instance, ``True`` for default,
            ``False`` / ``None`` to disable.
        post_processor: Post-processor instance, ``True`` for default,
            ``False`` / ``None`` to disable.
        evaluator: Evaluator instance, ``True`` for default,
            ``False`` / ``None`` to disable.
        visualizer: Visualizer instance, ``True`` for default,
            ``False`` / ``None`` to disable.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer: str = "adam",
        scheduler: str | None = None,
        scheduler_kwargs: dict[str, Any] | None = None,
        pre_processor: PreProcessor | bool | None = True,
        post_processor: PostProcessor | bool | None = True,
        evaluator: Evaluator | bool | None = False,
        visualizer: Visualizer | bool | None = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["pre_processor", "post_processor", "evaluator", "visualizer"],
        )

        self.model: nn.Module
        self.loss_fn: nn.Module

        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._optimizer_name = optimizer
        self._scheduler_name = scheduler
        self._scheduler_kwargs = scheduler_kwargs or {}

        self.pre_processor = self._resolve_component(pre_processor, nn.Module, self.configure_pre_processor)
        self.post_processor = self._resolve_component(post_processor, nn.Module, self.configure_post_processor)
        self.evaluator = self._resolve_component(evaluator, Evaluator, self.configure_evaluator)
        self.visualizer = self._resolve_component(visualizer, Visualizer, self.configure_visualizer)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        """Model class name."""
        return self.__class__.__name__

    @property
    @abstractmethod
    def task_type(self) -> TaskType:
        """The task this model solves (reconstruction, regression, etc.)."""

    @property
    @abstractmethod
    def modality(self) -> ModalityType:
        """The input modality of this model."""

    # ------------------------------------------------------------------
    # Component configuration – override in subclasses
    # ------------------------------------------------------------------
    def configure_pre_processor(self) -> PreProcessor:
        """Return the default pre-processor."""
        return PreProcessor()

    def configure_post_processor(self) -> PostProcessor:
        """Return the default post-processor."""
        return PostProcessor()

    def configure_evaluator(self) -> Evaluator | None:
        """Return the default evaluator.  ``None`` means no evaluator."""
        return None

    def configure_visualizer(self) -> Visualizer | None:
        """Return the default visualizer.  ``None`` means no visualizer."""
        return None

    def configure_callbacks(self) -> list[Callback]:
        """Collect all sub-components that are Lightning Callbacks."""
        return [
            component
            for component in (self.pre_processor, self.post_processor, self.evaluator, self.visualizer)
            if isinstance(component, Callback)
        ]

    # ------------------------------------------------------------------
    # Forward: export / standalone inference pipeline
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Export-ready inference pipeline.

        ``pre_processor(x) → model(x) → post_processor(x)``

        During Lightning training the pre/post processors run as
        **callbacks** instead, so ``forward`` is only used for
        standalone inference and ONNX export.
        """
        if self.pre_processor is not None:
            x = self.pre_processor(x)
        x = self.model(x)
        if self.post_processor is not None:
            x = self.post_processor(x)
        return x

    # ------------------------------------------------------------------
    # Batch-level model inference (override for custom output keys)
    # ------------------------------------------------------------------
    def model_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the model on a batch and return an output dict.

        The default implementation calls ``self.model(batch["input"])``
        and wraps the result under the ``"prediction"`` key.  Override
        in subclasses that need additional output keys.

        Args:
            batch: Dict with at least ``"input"`` and ``"target"``.

        Returns:
            Dict with at least ``"prediction"``.
        """
        prediction = self.model(batch["input"])
        return {"prediction": prediction}

    # ------------------------------------------------------------------
    # Training / evaluation loop
    # ------------------------------------------------------------------
    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        output = self.model_step(batch)
        loss = self._compute_loss(output, batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, Any]:
        del batch_idx
        output = self.model_step(batch)
        loss = self._compute_loss(output, batch)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        return {**output, **batch}

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, Any]:
        del batch_idx
        output = self.model_step(batch)
        loss = self._compute_loss(output, batch)
        self.log("test/loss", loss, on_epoch=True)
        return {**output, **batch}

    def predict_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0,
    ) -> dict[str, Any]:
        del batch_idx, dataloader_idx
        output = self.model_step(batch)
        return {**output, **batch}

    @abstractmethod
    def _compute_loss(self, output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the training loss from model output and ground truth batch."""

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------
    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = self._build_optimizer()
        config: dict[str, Any] = {"optimizer": optimizer}
        if self._scheduler_name is not None:
            scheduler = self._build_scheduler(optimizer)
            config["lr_scheduler"] = {"scheduler": scheduler, "monitor": "val/loss"}
        return config

    def _build_optimizer(self) -> torch.optim.Optimizer:
        optimizers = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
        }
        if self._optimizer_name not in optimizers:
            msg = f"Unsupported optimizer: {self._optimizer_name}. Choose from {list(optimizers.keys())}"
            raise ValueError(msg)
        return optimizers[self._optimizer_name](
            self.parameters(),
            lr=self._learning_rate,
            weight_decay=self._weight_decay,
        )

    def _build_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        schedulers = {
            "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
            "step": torch.optim.lr_scheduler.StepLR,
            "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        }
        if self._scheduler_name not in schedulers:
            msg = f"Unsupported scheduler: {self._scheduler_name}. Choose from {list(schedulers.keys())}"
            raise ValueError(msg)
        return schedulers[self._scheduler_name](optimizer, **self._scheduler_kwargs)

    # ------------------------------------------------------------------
    # from_config class method
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config_path: str | Path, **overrides: Any) -> TorchSenseModule:
        """Instantiate a model from a YAML configuration file.

        Args:
            config_path: Path to a YAML config (compatible with LightningCLI).
            **overrides: Key-value overrides applied on top of the config.

        Returns:
            Instantiated ``TorchSenseModule`` subclass.
        """
        from jsonargparse import ActionConfigFile, ArgumentParser

        config_path = Path(config_path)
        if not config_path.exists():
            msg = f"Configuration file not found: {config_path}"
            raise FileNotFoundError(msg)

        parser = ArgumentParser()
        parser.add_argument("-c", "--config", action=ActionConfigFile)
        parser.add_subclass_arguments(TorchSenseModule, "model", required=False, fail_untyped=False)
        args = ["--config", str(config_path)]
        for key, value in overrides.items():
            args.extend([f"--{key}", str(value)])
        config = parser.parse_args(args=args)
        instantiated = parser.instantiate_classes(config)
        model = instantiated.get("model")
        if isinstance(model, TorchSenseModule):
            return model
        msg = f"Instantiated object is not a TorchSenseModule: {type(model)}"
        raise TypeError(msg)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_component(
        component: nn.Module | bool | None,
        component_type: type,
        default_factory: Callable,
    ) -> nn.Module | None:
        """Resolve a sub-component from ``bool | Instance | None``.

        - ``True`` → call *default_factory* to create the default.
        - ``False`` / ``None`` → disabled (returns ``None``).
        - An instance of *component_type* → used as-is.
        """
        if isinstance(component, bool):
            return default_factory() if component else None
        if component is None:
            return None
        if isinstance(component, component_type):
            return component
        msg = f"Expected {component_type.__name__}, bool, or None — got {type(component)}"
        raise TypeError(msg)
