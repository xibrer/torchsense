import torch
import torch.nn as nn

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock1D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(in_channels),
            nn.PReLU(),
            nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(in_channels)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator1D(nn.Module):
    def __init__(self, in_channels=1, num_residual_blocks=16):
        super(Generator1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=9, stride=1, padding=4)
        self.prelu = nn.PReLU()

        res_blocks = []
        for _ in range(num_residual_blocks):
            res_blocks.append(ResidualBlock1D(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        upsample_blocks = []
        for _ in range(2):  # Upsample by 4 (2x2)
            upsample_blocks.append(nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1))
            upsample_blocks.append(nn.Upsample(scale_factor=2, mode='nearest'))
            upsample_blocks.append(nn.PReLU())
        self.upsample_blocks = nn.Sequential(*upsample_blocks)

        self.conv3 = nn.Conv1d(64, in_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        out1 = self.prelu(self.conv1(x))
        out = self.res_blocks(out1)
        out = self.bn2(self.conv2(out))
        out = out1 + out
        out = self.upsample_blocks(out)
        out = self.conv3(out)
        # print(out.shape)
        return out

    
class Discriminator1D(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator1D, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(512, 1024, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(1024, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))
