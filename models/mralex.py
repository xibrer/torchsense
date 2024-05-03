import torch
import torch.nn as nn
from torch import optim
# from models.MobileNet.MobileNetV1 import DepthSeparableConv1d, DepthSeparableConvTranspose1d
import torch.nn.functional as F
import lightning.pytorch as pl
import os
# classes
import yaml


class DepthSeperabelConv1d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv1d(
                input_channels,
                input_channels,
                kernel_size,
                groups=input_channels,
                **kwargs),
            nn.BatchNorm1d(input_channels),
            nn.ReLU(inplace=True)
        )

        self.pointwise = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, 1),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            DepthSeperabelConv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            DepthSeperabelConv1d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1,
                                 bias=False),
            nn.BatchNorm1d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class resalex(pl.LightningModule):
    def __init__(self, configs):
        super().__init__()
        self.save_hyperparameters()
        self.configs = configs
        self.training_step_outputs = []
        self.val_step_outputs = []
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))
        self.conv2_x = self._make_layer(BasicBlock, 64, 1, 2)
        self.conv3_x = self._make_layer(BasicBlock, 128, 1, 1)
        self.conv4_x = self._make_layer(BasicBlock, 256, 1, 2)
        # self.conv5_x = self._make_layer(BasicBlock, 256, 1, 2)
        self.final_layer = nn.Sequential(nn.Conv1d(256, 1, 1),
                                         nn.Linear(880, 110),
                                         nn.Dropout(0.2),
                                         nn.Linear(110, configs["model_params"]["num_classes"]),
                                         )

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        """The forward function takes in an image and returns the reconstructed image."""

        b_size = x.shape[0]
        output = self.conv1(x)
        output = self.conv2_x(output)
        # print(output.size())
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        # output = self.conv5_x(output)
        # print(output.size())
        x_hat = self.final_layer(output)
        # x_hat = x_hat.reshape(b_size, 1, -1)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        x, labels, _ = batch
        x_hat = self.forward(x)
        _loss = nn.SmoothL1Loss()
        loss = _loss(x_hat.flatten(), labels.flatten())
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.configs["exp_params"]["lr"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        loss = self._get_reconstruction_loss(train_batch)
        self.training_step_outputs.append(loss)
        self.log("train_loss", loss)
        # print(loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self._get_reconstruction_loss(val_batch)
        self.val_step_outputs.append(loss)
        self.log("val_loss", loss, "batch_size", batch_size=100)

# if __name__ == '__main__':
#     # gradient check

#     config_filename = os.path.join("../", 'configs/resalex.yml')
#     with open(config_filename, 'r') as file:
#         try:
#             configs = yaml.safe_load(file)
#         except yaml.YAMLError as exc:
#             print(exc)
#     batch_size = 1
#     model = resalex(configs).cuda()
#     loss_fn = torch.nn.MSELoss()
#     input = Variable(torch.randn(batch_size, 1, 3520)).cuda()
#     target = Variable(torch.randn(batch_size, 1, 2)).double().cuda()
#     flops, params = profile(model, inputs=(input, ))
#     flops, params = clever_format([flops, params], "%.3f")
#     output = model(input)
#     output = output[0].double()
#     res = torch.autograd.gradcheck(loss_fn, (output.flatten(), target.flatten()), eps=1e-6, raise_exception=True)
#     print(flops, params)
#     print(output.size())
