from torchsense.models.lit_model import (LitRegressModel,
                                         LitClassModel,
                                         LitMultimodalModel)
import lightning as L
import os
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import torch

L.seed_everything(42)
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/base")
print(CHECKPOINT_PATH)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
torch.set_float32_matmul_precision("medium")


class Trainer:
    def __init__(self, model, task="r",
                 precision="32", max_epochs=5,
                 accelerator="auto",
                 lr=0.001):
        if task == "r":
            self.model = LitRegressModel(model, lr=lr)
        elif task == "c":
            self.model = LitClassModel(model, lr=lr)
        elif task == "m":
            self.model = LitMultimodalModel(model, lr=lr)
        else:
            raise ValueError("task must be either 'r' or 'c'")
        self.max_epochs = max_epochs
        self.accelerator = accelerator
        self.precision = precision

    def fit(self, train_loader, val_loader):
        trainer = L.Trainer(
            default_root_dir=CHECKPOINT_PATH,
            precision=self.precision,
            accelerator=self.accelerator,
            max_epochs=self.max_epochs,
            callbacks=[
                ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss",
                                save_last=True, save_top_k=3, filename='{epoch}-{val_loss:.2f}'),
                # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer

            ],
        )
        trainer.fit(self.model, train_loader, val_loader)
