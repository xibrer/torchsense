import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchsense.metrics import mape_loss
from .utils import get_loss_fn


class LitRegressModel(L.LightningModule):
    def __init__(self, model, loss=None, lr=0.0001, gamma=0.7) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model", "loss_fn"])
        self.model = model
        # self.model.apply(self.weights_init)
        self.total_train_loss = []
        self.validation_step_outputs = []
        self.loss_name = loss
        self.loss_fn = get_loss_fn(loss)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def _calculate_loss(self, batch, mode="train"):
        """
        Calculates the loss for a given batch of data.

        Args:
            batch (tuple): A tuple containing the input data (x) and the target labels (y).
            mode (str, optional): The mode of operation. Defaults to "train".

        Returns:
            torch.Tensor: The calculated loss value.

        """
        x, z = batch[0]
        # print(x)
        y = batch[1]
        if not isinstance(x, torch.Tensor):
            x = tuple(x)

        preds = self.model(x)
        if self.loss_name == "clt":
            # Calculate the mean squared error loss between the predictions and the ground truth for the first prediction
            p_loss = F.mse_loss(preds[0], y.squeeze(1))

            # Calculate the factor as the difference between the max and min absolute values of the first prediction
            p_factor = torch.max(torch.abs(y), -1)[0] - torch.min(torch.abs(y), -1)[0]

            # Calculate the mean absolute percentage error loss between the second prediction and the difference of x and y
            n_loss = F.mse_loss(preds[0], (x - preds[1]).squeeze(1))
            # Calculate the factor as the difference between the max and min absolute values of the second prediction
            n_factor = (
                    torch.max(torch.abs(x - y), -1)[0] - torch.min(torch.abs(x - y), -1)[0]
            )
            # Print the calculated losses

            # Calculate the mean squared error loss between the sum of predictions and x
            rec_loss = F.mse_loss(preds[0] + preds[1], x.squeeze(1))
            n1 = torch.mean(p_factor) / (torch.mean(p_factor) + torch.mean(n_factor))
            p1 = torch.mean(n_factor) / (torch.mean(p_factor) + torch.mean(n_factor))
            print(p1, n1)
            print(f"p_loss: {p_loss*p1}, n_loss: {n_loss*n1}, rec_loss: {rec_loss}")
            # Combine the losses
            loss = p_loss * 0.5 + n_loss * 0.5  # + 0.03*rec_loss

        elif self.loss_name == "triplet":
            loss = self.loss_fn(preds, y, x)
        elif self.loss_name == "piece":
            loss = self.loss_fn(preds, y, z)
        else:
            loss = self.loss_fn(preds, y)
        if mode == "train":
            self.total_train_loss.append(loss)
            self.log("%s_loss" % mode, loss, prog_bar=True, on_step=True, on_epoch=True)
        else:
            self.validation_step_outputs.append(loss)
            self.log(
                "%s_loss" % mode, loss, prog_bar=True, on_step=False, on_epoch=True
            )

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
        # optimizer = optim.Adadelta(self.parameters(), lr=self.hparams.lr)
        # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def on_train_epoch_end(self):
        # log epoch metric
        stacked_tensors = torch.stack(self.total_train_loss)
        # 计算堆叠张量的均值
        mean_tensor = torch.mean(stacked_tensors, dim=0)
        self.log("train_loss_epoch", mean_tensor)
        self.total_train_loss = []

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def on_validation_epoch_end(self):
        stacked_tensors = torch.stack(self.validation_step_outputs)
        # 计算堆叠张量的均值
        mean_tensor = torch.mean(stacked_tensors, dim=0)
        self.log("val_loss_epoch", mean_tensor)
        self.validation_step_outputs.clear()

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
