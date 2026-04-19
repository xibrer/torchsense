"""Export mixin providing ONNX model export for TorchSense modules."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


class ExportMixin:
    """Mixin that adds ONNX export capability to a :class:`TorchSenseModule`.

    The export uses the model's ``forward(x)`` pipeline which chains
    ``pre_processor → model → post_processor``, producing a self-contained
    artifact suitable for deployment.
    """

    def export_onnx(
        self: nn.Module,
        output_path: str | Path,
        input_shape: tuple[int, ...] = (1, 1, 2048),
        opset_version: int = 17,
        dynamic_axes: dict[str, dict[int, str]] | None = None,
        **kwargs: Any,
    ) -> Path:
        """Export the model to ONNX format.

        Args:
            output_path: Destination ``.onnx`` file path.
            input_shape: Shape of the dummy input tensor used for tracing.
            opset_version: ONNX opset version.
            dynamic_axes: Dynamic axis specification.  Defaults to dynamic
                batch dimension on both input and output.
            **kwargs: Extra arguments forwarded to :func:`torch.onnx.export`.

        Returns:
            Resolved path to the exported file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if dynamic_axes is None:
            dynamic_axes = {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            }

        dummy_input = torch.randn(*input_shape, device=next(self.parameters()).device)

        was_training = self.training
        self.eval()
        try:
            torch.onnx.export(
                self,
                dummy_input,
                str(output_path),
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                **kwargs,
            )
        finally:
            if was_training:
                self.train()

        logger.info("Exported ONNX model to %s", output_path)
        return output_path
