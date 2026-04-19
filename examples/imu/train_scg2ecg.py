"""Example: Train SCG-to-ECG reconstruction model using TorchSense API.

Usage:
    # With API:
    python examples/imu/train_scg2ecg.py --data_root ./data/vib2ecg

    # With CLI (equivalent):
    torchsense fit --config configs/model/scg2ecg_unet.yaml --config configs/data/vib2ecg.yaml
"""

from __future__ import annotations

import argparse

from torchsense.data.datamodules.imu.vib2ecg import Vib2ECGDataModule
from torchsense.data.transforms.imu import AddGaussianNoise, Compose, RandomScale
from torchsense.engine import Engine
from torchsense.models.imu.scg2ecg import SCG2ECG


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SCG-to-ECG model")
    parser.add_argument("--data_root", type=str, default="./data/vib2ecg")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--base_filters", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--window_size", type=int, default=2048)
    args = parser.parse_args()

    train_transform = Compose([
        AddGaussianNoise(std=0.01),
        RandomScale(low=0.9, high=1.1),
    ])

    datamodule = Vib2ECGDataModule(
        data_root=args.data_root,
        window_size=args.window_size,
        batch_size=args.batch_size,
        train_transform=train_transform,
    )

    model = SCG2ECG(
        in_channels=1,
        out_channels=1,
        base_filters=args.base_filters,
        depth=args.depth,
        learning_rate=args.learning_rate,
    )

    engine = Engine(max_epochs=args.max_epochs)
    engine.fit(model, datamodule)
    engine.test(model, datamodule)


if __name__ == "__main__":
    main()
