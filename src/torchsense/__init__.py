"""TorchSense: A modular deep learning framework for intelligent sensing."""

from __future__ import annotations

from enum import Enum

__version__ = "0.1.0dev"


class TaskType(str, Enum):
    """Supported task types across all modalities."""

    RECONSTRUCTION = "reconstruction"
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    ENHANCEMENT = "enhancement"


class ModalityType(str, Enum):
    """Supported sensing modalities."""

    IMU = "imu"
    RADAR = "radar"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"
