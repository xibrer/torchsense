from torchsense.models.lit_model import *
import lightning as L
import os
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import torch
from lightning.pytorch.tuner import Tuner
import time
from pathlib import Path
L.seed_everything(42)

torch.set_float32_matmul_precision("medium")


class Trainer:
    def __init__(self, model, save_name = "base",loss=None, task="r", precision="32", max_epochs=5,
                 lr=0.001, logger = None,*args):
        if task == "r":
            self.model = LitRegressModel(model, lr=lr, loss_fn=loss)
        elif task == "c":
            self.model = LitClassModel(model, lr=lr)
        elif task == "m":
            self.model = LitMultimodalModel(model, lr=lr)
        elif task == "two" or task == "t":
            self.model = LitTwoStageModel(model, lr=lr)
        else:
            raise ValueError("task must be either 'r' or 'c'")
        CHECKPOINT_PATH = Path("outputs-lightnings", save_name,time.strftime("%Y%m%d/%H%M%S"))

        print("model saved at :",CHECKPOINT_PATH/"ckpt")
        os.makedirs(CHECKPOINT_PATH/"ckpt", exist_ok=True)
        if logger is None:
            logger = CSVLogger(CHECKPOINT_PATH,name=None)
        self.max_epochs = max_epochs
        self.precision = precision
        self.trainer = L.Trainer(
            precision=self.precision,
            accelerator="auto",
            max_epochs=self.max_epochs,
            logger=logger,
            callbacks=[
                ModelCheckpoint(dirpath=CHECKPOINT_PATH/"ckpt", filename='{epoch}-{val_loss_epoch:.2f}',
                                save_weights_only=True, mode="min", monitor="val_loss_epoch",
                                enable_version_counter=False,
                                save_last=True, save_top_k=3),
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
