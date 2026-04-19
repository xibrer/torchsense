"""Base metric wrapper for TorchSense, analogous to anomalib's AnomalibMetric.

Provides a unified interface for extracting data from step-output dicts
and delegating to a ``torchmetrics.Metric``.  Subclass and override
:meth:`extract` for complex extraction logic (e.g. column slicing).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torchmetrics import Metric


class TorchSenseMetric(nn.Module):
    """Metric wrapper that auto-extracts values from step-output dicts.

    For simple cases, pass ``fields`` to extract dict keys directly.
    For complex extraction (e.g. column slicing), subclass and override
    :meth:`extract`.

    Args:
        metric: A ``torchmetrics.Metric`` instance.
        fields: Ordered list of dict keys to extract and pass to
            ``metric.update(*values)``.
        prefix: String prepended to the metric name when logging.
        flatten: If ``True``, flatten each extracted tensor before updating.
        strict: If ``True``, raise ``ValueError`` when a required field is
            missing from the step output.  If ``False``, silently skip the
            update.

    Raises:
        ValueError: When ``strict=True`` and a required field is missing.

    Example:
        >>> from torchmetrics import MeanSquaredError
        >>> m = TorchSenseMetric(MeanSquaredError(), fields=["prediction", "target"], flatten=True)
        >>> m.update({"prediction": torch.randn(2, 1, 64), "target": torch.randn(2, 1, 64)})
        >>> m.compute()
        tensor(...)
    """

    def __init__(
        self,
        metric: Metric,
        fields: Sequence[str] | None = None,
        prefix: str = "",
        flatten: bool = False,
        strict: bool = True,
    ) -> None:
        super().__init__()
        self.metric = metric
        self.fields = list(fields) if fields is not None else []
        self.prefix = prefix
        self.flatten = flatten
        self.strict = strict

    @property
    def name(self) -> str:
        return f"{self.prefix}{self.metric.__class__.__name__}"

    def extract(self, step_output: dict[str, Any]) -> tuple[torch.Tensor, ...]:
        """Extract metric inputs from *step_output*.

        The default implementation pulls values by ``self.fields`` and
        optionally flattens them.  Override in subclasses for custom
        extraction logic (e.g. column slicing, derived quantities).

        Args:
            step_output: Merged dict of model outputs and batch data.

        Returns:
            Tuple of tensors to pass to ``metric.update(*values)``.
            Return an empty tuple to skip the update.

        Raises:
            ValueError: When ``strict=True`` and a field is missing.
        """
        values: list[torch.Tensor] = []
        for field in self.fields:
            if field not in step_output:
                if self.strict:
                    msg = (
                        f"Metric {self.name}: required field '{field}' not found "
                        f"in step output. Available keys: {list(step_output.keys())}"
                    )
                    raise ValueError(msg)
                return ()
            val = step_output[field]
            if val is None:
                if self.strict:
                    msg = f"Metric {self.name}: field '{field}' is None."
                    raise ValueError(msg)
                return ()
            values.append(val)
        if self.flatten:
            values = [v.flatten() for v in values]
        return tuple(values)

    def update(self, step_output: dict[str, Any]) -> None:
        """Extract values via :meth:`extract` and forward to the inner metric."""
        values = self.extract(step_output)
        if values:
            self.metric.update(*values)

    def compute(self) -> torch.Tensor:
        return self.metric.compute()

    def reset(self) -> None:
        self.metric.reset()

    @property
    def update_called(self) -> bool:
        return self.metric._update_called  # noqa: SLF001
