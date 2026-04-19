import torch
import torch.nn as nn
# from models.MobileNet.MobileNetV1 import DepthSeparableConv1d, DepthSeparableConvTranspose1d
import torch.nn.functional as F
import os
# classes
import yaml
from torch.autograd import Variable
from thop import profile
from thop import clever_format


class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_dims: list,
                 latent_dim: int,
                 map_size: int,
                 length: int):
        super(Encoder, self).__init__()
        # modules = [nn.Dropout(0.2)]
        modules = []
        for h_dim in hidden_dims:
            length = length / 2
            modules.append(nn.Sequential(nn.Conv1d(in_channels, out_channels=h_dim,
                                                   kernel_size=3, stride=2, padding=1),
                                         nn.LayerNorm(int(length)),
                                         nn.LeakyReLU()))
            in_channels = h_dim
        self.net = nn.Sequential(*modules)
        self.fc = nn.Sequential(nn.Flatten(),
                                nn.Linear(hidden_dims[-1] * map_size, latent_dim),
                                nn.LeakyReLU())

    def forward(self, x):
        x = self.net(x)
        # x = self.fc(x)
        return x


class Autoencoder(nn.Module):
    def __init__(
            self,
            num_input_channels: int,
            latent_dim: int,
            hidden_dims: list = None,
            length: int = 3520,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64,128,256]
        feature_map_size = int(length / pow(2, len(hidden_dims)))

        # Creating encoder and decoder
        self.encoder = Encoder(num_input_channels, hidden_dims, latent_dim, feature_map_size, length)
        self.final_layer2 = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.Sigmoid(),
            nn.Linear(256, 2)
        )
        self.final_layer = nn.Sequential(nn.Conv1d(256,128,
                                               kernel_size=3,
                                               stride=2,
                                               padding=1),
                            nn.BatchNorm1d(128),
                            nn.LeakyReLU(),
                            nn.Conv1d(128, out_channels= 1,
                                      kernel_size= 3, padding= 1),
                            nn.Dropout(0.2),
                            nn.Linear(110,1),
                           )
    def forward(self, x):
        """The forward function takes in an image and returns the reconstructed image."""

        b_size = x.shape[0]
        z = self.encoder(x)
        x_hat = self.final_layer(z)
        x_hat = x_hat.reshape(b_size, 1, -1)
        return x_hat

if __name__ == '__main__':
    # gradient check
    batch_size = 1
    model = Autoencoder(1,64).cuda()
    loss_fn = torch.nn.MSELoss()
    input = Variable(torch.randn(batch_size, 1, 3520)).cuda()
    target = Variable(torch.randn(batch_size, 1, 2)).double().cuda()
    flops, params = profile(model, inputs=(input, ))
    flops, params = clever_format([flops, params], "%.3f")
    output = model(input)
    output = output[0].double()
    res = torch.autograd.gradcheck(loss_fn, (output.flatten(), target.flatten()), eps=1e-6, raise_exception=True)
    print(flops, params)
    print(output.size())