import torch
# from models import BaseVAE
from torch import nn, Tensor
from typing import List, Callable, Union, Any, TypeVar, Tuple
# from torch import tensor as Tensor
from torch.nn import MSELoss, SmoothL1Loss

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


class MobileNetV1(nn.Module):

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
                DepthSeparableConv1d(32, out_channels=64,
                                     kernel_size=3, stride=2, padding=1, bias=False),
                DepthSeparableConv1d(64, out_channels=128,
                                     kernel_size=3, stride=2, padding=1, bias=False),
                DepthSeparableConv1d(128, out_channels=128,
                                     kernel_size=3, stride=1, padding=1, bias=False),
                DepthSeparableConv1d(128, out_channels=256,
                                     kernel_size=3, stride=2, padding=1, bias=False),
                DepthSeparableConv1d(256, out_channels=256,
                                     kernel_size=3, stride=1, padding=1, bias=False),
                DepthSeparableConv1d(256, out_channels=512,
                                     kernel_size=3, stride=2, padding=1, bias=False),
                DepthSeparableConv1d(512, out_channels=1024,
                                     kernel_size=3, stride=2, padding=1, bias=False)
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
