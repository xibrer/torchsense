from typing import List

import torch
import torch.nn.functional as F
# from models import BaseVAE
from torch import nn, Tensor
# from torch import tensor as Tensor
from torch.nn import SmoothL1Loss

map_size = 10  # length after final conventional  20*32


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


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class MobileBottleneck(nn.Module):
    def __init__(self, in_channels, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and in_channels == oup

        conv_layer = nn.Conv1d
        norm_layer = nn.BatchNorm1d
        if nl == 'RE':
            nlin_layer = nn.ReLU  # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            conv_layer(in_channels, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super().__init__()

        self.latent_dim = latent_dim

        modules_i = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512, 1024]

        # Build Encoder 通过Sequential将网络层和激活函数结合起来，输出激活后的五层网络节点
        modules_i.append(
            nn.Sequential(
                BasicConv1d(in_channels, out_channels=32,
                            kernel_size=3, stride=2, padding=1),
                MobileBottleneck(32, 64, 3, 2, exp=88, se=False, nl='RE'),
                MobileBottleneck(64, 128, 3, 2, exp=96, se=True, nl='RE'),
                MobileBottleneck(128, 256, 3, 2, exp=288, se=True, nl='HS'),
                MobileBottleneck(256, 512, 3, 2, exp=512, se=True, nl='HS'),
                MobileBottleneck(512, 1024, 3, 2, exp=1024, se=True, nl='HS')

            ))
        self.encoder_i = nn.Sequential(*modules_i)  # 将输入i加载到网络上

        self.fc_mu_i = nn.Sequential(
            nn.Linear(hidden_dims[-1] * map_size, latent_dim),  # 将1024*20转化为64维,
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU())

        # Build Decoder
        modules = []
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[-1] * map_size),  # 将64维,
            nn.BatchNorm1d(hidden_dims[-1] * map_size),
            nn.LeakyReLU())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    DepthSeparableConvTranspose1d(hidden_dims[i], hidden_dims[i + 1],
                                                  kernel_size=3, stride=2, padding=1, output_padding=1)))
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            DepthSeparableConvTranspose1d(hidden_dims[-1], hidden_dims[-1],
                                          kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Conv1d(hidden_dims[-1], out_channels=1,
                      kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def encode_i(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder_i(input)
        # print(result.shape)
        # flatten it to 1 dim to better performance
        result = torch.flatten(result, start_dim=1)
        # print(result.shape)
        # with the latent Gaussian distribution
        mu_i = self.fc_mu_i(result)
        # print(mu_i.shape)
        return mu_i

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        # print(result.shape)
        result = result.view(-1, 1024, map_size)
        # print(result.shape)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def loss_function(self, *args, gt) -> dict:
        recons = args[0]
        recons_loss_ = SmoothL1Loss()
        recons_loss = recons_loss_(recons.flatten(), gt.flatten())
        loss = recons_loss
        return loss

    def forward(self, input_i: Tensor, **kwargs) -> List[Tensor]:

        mu_i = self.encode_i(input_i)

        return [self.decode(mu_i), input_i, mu_i]

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the models
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)
        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
