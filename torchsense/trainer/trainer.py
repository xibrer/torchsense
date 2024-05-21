from torchsense.models.lit_model import *
import lightning as L
import os
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
from lightning.pytorch.tuner import Tuner

L.seed_everything(42)
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/base")
print(CHECKPOINT_PATH)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
torch.set_float32_matmul_precision("medium")


class Trainer:
    def __init__(self, model,task="r", precision="32", max_epochs=5,
                 lr=0.001, *args):
        if task == "r":
            self.model = LitRegressModel(model, lr=lr)
        elif task == "c":
            self.model = LitClassModel(model, lr=lr)
        elif task == "m":
            self.model = LitMultimodalModel(model, lr=lr)
        elif task == "two" or task == "t":
            self.model = LitTwoStageModel(model, lr=lr)
        else:
            raise ValueError("task must be either 'r' or 'c'")
        self.max_epochs = max_epochs
        self.precision = precision
        self.trainer = L.Trainer(
            default_root_dir=CHECKPOINT_PATH,
            precision=self.precision,
            accelerator="auto",
            max_epochs=self.max_epochs,
            callbacks=[
                ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss",
                                save_last=True, save_top_k=3, filename='{epoch}-{val_loss:.2f}'),
                # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer

            ], *args,
        )

    def fit(self, train_loader, val_loader):

        self.trainer.fit(self.model, train_loader, val_loader)

    def lr_find(self, train_loader, val_loader):

        tuner = Tuner(self.trainer)

        # finds learning rate automatically
        # sets hparams.lr or hparams.learning_rate to that learning rate
        lr_finder = tuner.lr_find(self.model, train_loader, val_loader)
        print(lr_finder.suggestion())
