import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional.classification.accuracy import accuracy


class LitClassModel(L.LightningModule):
    def __init__(self, model, lr=1, gamma=0.7) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.model.apply(self.weights_init)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def _calculate_loss(self, batch, mode="train"):
        x, y = batch
        preds = self.model(x)
        # print(preds.max())
        loss = self.loss_fn(preds, y)
        acc = (preds.argmax(dim=-1) == y).float().mean()

        self.log("%s_loss" % mode, loss, prog_bar=True)
        self.log("%s_acc" % mode, acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        optimizer = optim.Adadelta(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=self.hparams.gamma
        )
        return [optimizer], [lr_scheduler]

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
