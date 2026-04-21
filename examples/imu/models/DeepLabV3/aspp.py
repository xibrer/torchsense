import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)


class ASPP(nn.Module):
    def __init__(self, backbone, output_stride):
        super(ASPP, self).__init__()
        if backbone == 'mobilenet':
            in_channels = 512
        else:
            in_channels = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(in_channels, 256, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(in_channels, 256, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(in_channels, 256, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(in_channels, 256, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                             nn.Conv1d(in_channels, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm1d(256),
                                             nn.LeakyReLU())
        self.final_layer = nn.Sequential(
            nn.Conv1d(256*5, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            # nn.Dropout(0.5)
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        # print(x4.shape[2:])
        x5 = F.interpolate(x5, size=x1.shape[2:], mode='linear')
        # print(x.shape)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        # print(x.shape)
        x = self.final_layer(x)

        return x


def build_aspp(backbone, output_stride):
    return ASPP(backbone, output_stride)
