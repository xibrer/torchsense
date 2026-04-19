"""Lightning module for SCG-to-blood-pressure prediction."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchmetrics import MeanAbsoluteError

from torchsense import ModalityType, TaskType
from torchsense.metrics.base import TorchSenseMetric
from torchsense.metrics.evaluator import Evaluator
from torchsense.models import register_model
from torchsense.models.components.base.torchsense_module import TorchSenseModule
from torchsense.models.imu.scg2bp.torch_model import SCG2BPModel
from torchsense.post_processing import BPPostProcessor, PostProcessor
from torchsense.pre_processing import PreProcessor
from torchsense.visualization.base import Visualizer
from torchsense.visualization.signal import BPVisualizer


class _BPColumnMetric(TorchSenseMetric):
    """Metric that extracts a single BP column (SBP or DBP) from prediction/target.

    Args:
        metric: A ``torchmetrics.Metric`` instance.
        column: Column index to extract (0 = SBP, 1 = DBP).
        prefix: Logging prefix.
    """

    def __init__(self, metric: torch.nn.Module, column: int, prefix: str = "") -> None:
        super().__init__(metric=metric, fields=["prediction", "target"], prefix=prefix)
        self.column = column

    def extract(self, step_output: dict[str, Any]) -> tuple[torch.Tensor, ...]:
        for field in self.fields:
            if field not in step_output:
                if self.strict:
                    msg = (
                        f"Metric {self.name}: required field '{field}' not found "
                        f"in step output. Available keys: {list(step_output.keys())}"
                    )
                    raise ValueError(msg)
                return ()
        pred = step_output["prediction"][:, self.column]
        target = step_output["target"][:, self.column]
        return (pred, target)


@register_model("scg2bp")
class SCG2BP(TorchSenseModule):
    """SCG-to-Blood-Pressure prediction Lightning module.

    Trains a ResNet1D to predict systolic and diastolic blood pressure
    from chest-mounted accelerometer signals.

    Args:
        in_channels: Number of SCG axes.
        base_filters: Base ResNet filter count.
        num_blocks: Residual blocks per stage.
        num_stages: Number of stages.
        kernel_size: Convolution kernel size.
        dropout: Dropout rate.
        use_se: Use SE attention.
        loss: Loss function ('mse', 'l1', 'huber').
        learning_rate: Optimizer LR.
        weight_decay: Weight decay.
        optimizer: Optimizer name.
        scheduler: LR scheduler name.
        scheduler_kwargs: Scheduler kwargs.
        pre_processor: Pre-processor (bool / instance / None).
        post_processor: Post-processor (bool / instance / None).
        evaluator: Evaluator (bool / instance / None).
        visualizer: Visualizer (bool / instance / None).
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_filters: int = 64,
        num_blocks: int = 2,
        num_stages: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.2,
        use_se: bool = True,
        loss: str = "huber",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer: str = "adamw",
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

        self.model = SCG2BPModel(
            in_channels=in_channels,
            base_filters=base_filters,
            num_blocks=num_blocks,
            num_stages=num_stages,
            kernel_size=kernel_size,
            dropout=dropout,
            use_se=use_se,
        )
        self.loss_fn = _get_loss(loss)

    @property
    def task_type(self) -> TaskType:
        return TaskType.REGRESSION

    @property
    def modality(self) -> ModalityType:
        return ModalityType.IMU

    # ------------------------------------------------------------------
    # Default components
    # ------------------------------------------------------------------
    def configure_post_processor(self) -> BPPostProcessor:
        return BPPostProcessor()

    def configure_evaluator(self) -> Evaluator:
        mae_sbp = _BPColumnMetric(MeanAbsoluteError(), column=0, prefix="sbp_")
        mae_dbp = _BPColumnMetric(MeanAbsoluteError(), column=1, prefix="dbp_")
        return Evaluator(val_metrics=[mae_sbp, mae_dbp])

    def configure_visualizer(self) -> BPVisualizer:
        return BPVisualizer()

    # ------------------------------------------------------------------
    # Model step — clean, no metric-specific splitting
    # ------------------------------------------------------------------
    def model_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        bp_pred = self.model(batch["input"])  # (B, 2)
        return {"prediction": bp_pred}

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
