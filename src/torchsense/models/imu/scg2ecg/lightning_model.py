"""Lightning module for SCG-to-ECG reconstruction training."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchmetrics import MeanSquaredError, PearsonCorrCoef

from torchsense import ModalityType, TaskType
from torchsense.metrics.base import TorchSenseMetric
from torchsense.metrics.evaluator import Evaluator
from torchsense.models import register_model
from torchsense.models.components.base.torchsense_module import TorchSenseModule
from torchsense.models.imu.scg2ecg.torch_model import SCG2ECGModel
from torchsense.post_processing import PostProcessor, SignalPostProcessor
from torchsense.pre_processing import PreProcessor
from torchsense.visualization.base import Visualizer
from torchsense.visualization.signal import SignalVisualizer


@register_model("scg2ecg")
class SCG2ECG(TorchSenseModule):
    """SCG-to-ECG reconstruction Lightning module.

    Trains a 1D U-Net to reconstruct ECG waveforms from chest-mounted
    accelerometer (seismocardiography) signals.

    Args:
        in_channels: Number of SCG axes used as input.
        out_channels: Number of ECG leads to reconstruct.
        base_filters: Base U-Net filter count.
        depth: U-Net depth.
        kernel_size: Convolution kernel size.
        dropout: Dropout rate.
        loss: Loss function name ('mse', 'l1', 'huber').
        learning_rate: Optimizer learning rate.
        weight_decay: Optimizer weight decay.
        optimizer: Optimizer name.
        scheduler: LR scheduler name or None.
        scheduler_kwargs: Additional scheduler keyword arguments.
        pre_processor: Pre-processor (bool / instance / None).
        post_processor: Post-processor (bool / instance / None).
        evaluator: Evaluator (bool / instance / None).
        visualizer: Visualizer (bool / instance / None).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_filters: int = 32,
        depth: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.1,
        loss: str = "mse",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer: str = "adam",
        scheduler: str | None = "cosine",
        scheduler_kwargs: dict[str, Any] | None = None,
        pre_processor: PreProcessor | bool | None = True,
        post_processor: PostProcessor | bool | None = True,
        evaluator: Evaluator | bool | None = True,
        visualizer: Visualizer | bool | None = False,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs or {"T_max": 100},
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

        self.model = SCG2ECGModel(
            in_channels=in_channels,
            out_channels=out_channels,
            base_filters=base_filters,
            depth=depth,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.loss_fn = _get_loss(loss)

    @property
    def task_type(self) -> TaskType:
        return TaskType.RECONSTRUCTION

    @property
    def modality(self) -> ModalityType:
        return ModalityType.IMU

    # ------------------------------------------------------------------
    # Default components
    # ------------------------------------------------------------------
    def configure_post_processor(self) -> SignalPostProcessor:
        return SignalPostProcessor()

    def configure_evaluator(self) -> Evaluator:
        mse = TorchSenseMetric(MeanSquaredError(), fields=["prediction", "target"], flatten=True)
        pcc = TorchSenseMetric(PearsonCorrCoef(), fields=["prediction", "target"], flatten=True)
        return Evaluator(val_metrics=[mse, pcc])

    def configure_visualizer(self) -> SignalVisualizer:
        return SignalVisualizer()

    # ------------------------------------------------------------------
    # Model step
    # ------------------------------------------------------------------
    def model_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ecg_pred = self.model(batch["input"])
        return {"prediction": ecg_pred}

    def _compute_loss(self, output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.loss_fn(output["prediction"], batch["target"])


def _get_loss(name: str) -> nn.Module:
    losses = {
        "mse": nn.MSELoss(),
        "l1": nn.L1Loss(),
        "huber": nn.HuberLoss(),
    }
    if name not in losses:
        msg = f"Unknown loss: {name}. Choose from {list(losses.keys())}"
        raise ValueError(msg)
    return losses[name]
