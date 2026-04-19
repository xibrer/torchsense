"""LightningCLI entry point for TorchSense."""

from __future__ import annotations

from lightning.pytorch.cli import LightningCLI

from torchsense.data.datamodules.base import TorchSenseDataModule
from torchsense.models.components.base.torchsense_module import TorchSenseModule


class TorchSenseCLI(LightningCLI):
    """CLI for TorchSense with auto-configured subcommands (fit, validate, test, predict)."""

    def add_arguments_to_parser(self, parser) -> None:
        parser.link_arguments("data.init_args.batch_size", "model.init_args.batch_size", apply_on="parse")


def main() -> None:
    """Main entry point for the TorchSense CLI."""
    TorchSenseCLI(
        model_class=TorchSenseModule,
        datamodule_class=TorchSenseDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
    )


if __name__ == "__main__":
    main()
