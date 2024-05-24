import lightning as L
import torch
from torchmetrics.functional.classification.accuracy import accuracy
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchmetrics.regression import MeanSquaredError
from typing import Callable, Optional
from torchmetrics.audio import (ScaleInvariantSignalNoiseRatio,
                                PerceptualEvaluationSpeechQuality,
                                ShortTimeObjectiveIntelligibility,
                                ScaleInvariantSignalDistortionRatio,
                                SignalDistortionRatio, )


class LitTwoStageModel(L.LightningModule):
    def __init__(self, model,
                 lr: float = 0.0001, gamma=0.7) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.model.apply(self.weights_init)
        self.loss_fn = MeanSquaredError()
        self.si_snr = ScaleInvariantSignalNoiseRatio()
        self.pesq = PerceptualEvaluationSpeechQuality(16000, 'nb')
        self.stoi = ShortTimeObjectiveIntelligibility(16000, False)
        self.sdr = SignalDistortionRatio()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def _calculate_loss(self, batch, mode="train"):
        x = batch[0]
        targets = batch[1]

        preds = self.model(x)
        loss = self.loss_fn(preds.flatten(), targets.flatten())
        sdr = self.sdr(preds, targets)
        stoi = self.stoi(preds, targets)
        pesq = self.pesq(preds, targets)
        si_snr = self.si_snr(preds, targets)
        self.log("%s_loss" % mode, loss, prog_bar=True, on_step=(mode == "train"), on_epoch=(mode == "val"))
        if mode == "val":
            self.log(f'{mode}_sdr', sdr, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f'{mode}_si_snr', si_snr, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f'{mode}_stoi', stoi, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f'{mode}_pesq', pesq, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1)
        # optimizer = optim.Adadelta(self.parameters(), lr=self.hparams.lr)
        # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        # 对于任何其他类型的模块，如果它有子模块，则递归地应用 weights_init 函数
        elif isinstance(m, nn.Module):
            for name, child in m.named_children():
                self.weights_init(child)
