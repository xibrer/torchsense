"""TorchSense models registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchsense.models.components.base.torchsense_module import TorchSenseModule

_MODEL_REGISTRY: dict[str, type[TorchSenseModule]] = {}


def register_model(name: str):
    """Decorator to register a model class."""

    def wrapper(cls):
        _MODEL_REGISTRY[name] = cls
        return cls

    return wrapper


def get_model(name: str, **kwargs) -> TorchSenseModule:
    """Instantiate a model by name."""
    if name not in _MODEL_REGISTRY:
        msg = f"Model '{name}' not found. Available: {list(_MODEL_REGISTRY.keys())}"
        raise KeyError(msg)
    return _MODEL_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    """List all registered model names."""
    return list(_MODEL_REGISTRY.keys())
