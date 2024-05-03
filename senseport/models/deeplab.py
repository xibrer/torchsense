# from models import BaseVAE
import torch
from torch import nn
import mobilenet


class DepthSeparableConv1d(nn.Module):

    def __init__(self, input_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.Depthwise = nn.Sequential(
            nn.Conv1d(
                input_channels, input_channels,
                kernel_size, groups=input_channels, **kwargs),
            nn.BatchNorm1d(input_channels),
            nn.LeakyReLU(inplace=True)
        )

        self.Pointwise = nn.Sequential(
            nn.Conv1d(input_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.Depthwise(x)
        x = self.Pointwise(x)
        return x


class DepthSeparableConvTranspose1d(nn.Module):

    def __init__(self, input_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.Depthwise = nn.Sequential(
            nn.ConvTranspose1d(input_channels, input_channels,
                               kernel_size, groups=input_channels, **kwargs),
            nn.BatchNorm1d(input_channels),
            nn.LeakyReLU(inplace=True)
        )

        self.Pointwise = nn.Sequential(
            nn.Conv1d(input_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.Depthwise(x)
        # print("dw:", x.size())
        x = self.Pointwise(x)
        # print("pw:", x.size())
        # print("--------------")
        return x


class BasicConv1d(nn.Module):

    def __init__(self, input_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(
            input_channels, out_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class LinearBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, t=6):
        super(LinearBottleNeck, self).__init__()

        self.residual = nn.Sequential(
            # pw
            nn.Conv1d(in_channels, in_channels * t, 1, bias=False),
            nn.BatchNorm1d(in_channels * t),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv1d(in_channels * t, in_channels * t,
                      3, stride=stride, padding=1, groups=in_channels * t, bias=False),
            nn.BatchNorm1d(in_channels * t),
            nn.ReLU6(inplace=True),
            # pw
            nn.Conv1d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels)
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        residual = self.residual(x)
        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual


class MobileNetV1(nn.Module):

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        modules = [
            BasicConv1d(in_channels, out_channels=32,
                        kernel_size=3, stride=2, padding=1),
            DepthSeparableConv1d(32, out_channels=64,
                                 kernel_size=3, stride=2, padding=1, bias=False),
            DepthSeparableConv1d(64, out_channels=128,
                                 kernel_size=3, stride=1, padding=1, bias=False),
            DepthSeparableConv1d(128, out_channels=256,
                                 kernel_size=3, stride=2, padding=1, bias=False),
            DepthSeparableConv1d(256, out_channels=512,
                                 kernel_size=3, stride=2, padding=1, bias=False),
            DepthSeparableConv1d(512, out_channels=512,
                                 kernel_size=3, stride=2, padding=1, bias=False)
        ]

        # Build Encoder 通过Sequential将网络层和激活函数结合起来，输出激活后的五层网络节点
        self.encoder = nn.Sequential(*modules)  # 将输入i加载到网络上
        self.low_level_features = self.encoder[0:1]
        self.middle_level_feature = self.encoder[1]
        self.high_level_features = self.encoder[2:]

    def forward(self, x):
        low_level_feat = self.low_level_features(x)
        # print("low after backbone", x.shape)
        middle_level_feat = self.middle_level_feature(low_level_feat)
        x = self.high_level_features(middle_level_feat)
        # print("x after backbone:", x.size())
        return low_level_feat, middle_level_feat, x


def build_backbone(backbone, in_channels):

    if backbone == 'mobilenet':
        return mobilenet.MobileNetV1(in_channels)
    else:
        raise