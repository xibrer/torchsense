import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, List

from .utils import get_loss_fn


class LitKDModel(L.LightningModule):
    def __init__(
            self,
            model,
            teacher_model=None,
            loss=None,
            register_layer=Optional[List[str]],
            lr=0.0001,
            gamma=0.7,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model", "loss_fn", "teacher_model"])
        self.teacher = teacher_model
        self.student = model
        self.student.apply(self.weights_init)
        self.total_train_loss = []
        self.validation_step_outputs = []
        self.loss_name = loss
        self.loss_fn = get_loss_fn(loss)
        self.activations = {}
        self.register_layers = register_layer
        self.register_hooks()
        print("Teacher model layers:")

    def forward(self, x):
        return self.student(x)

    def _calculate_loss(self, batch, mode="train"):
        x = batch[0]
        y = batch[1]
        if not isinstance(x, torch.Tensor):
            x = tuple(x)

        preds = self.student(x)[0]
        # 前向传播
        with torch.no_grad():
            teacher_preds = self.teacher(x)

        loss = self.loss_fn(
            preds, y, teacher_preds, self.activations, self.register_layers
        )

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
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.25)
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

    def register_hooks(self):
        for layer_name in self.register_layers:
            # 检查并注册teacher模型的钩子
            teacher_layer = getattr(self.teacher, layer_name, None)
            if teacher_layer is not None:
                teacher_hook = self.get_activation(f"teacher_{layer_name}")
                teacher_layer.register_forward_hook(teacher_hook)
                print(f"Registered teacher hook on layer: {layer_name}")
            else:
                print(f"Teacher model does not have a layer named: {layer_name}")

            # 检查并注册student模型的钩子
            student_layer = getattr(self.student, layer_name, None)
            if student_layer is not None:
                student_hook = self.get_activation(f"student_{layer_name}")
                student_layer.register_forward_hook(student_hook)
                print(f"Registered student hook on layer: {layer_name}")
            else:
                print(f"Student model does not have a layer named: {layer_name}")

    def get_activation(self, name):
        def hook(model, input, output):
            # print(f"Hook called for {name}")
            self.activations[name] = output.detach()
            return output  # 确保返回有效的输出

        return hook
