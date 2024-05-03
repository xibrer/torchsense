import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss

from models.DeepLabV3.backbone import build_backbone
from models.DeepLabV3.backbone.mobilenet import DepthSeparableConv1d


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
            nn.Conv1d(256 * 5, 256, 1, bias=False),
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


class Decoder(nn.Module):
    def __init__(self, num_classes, backbone):
        super(Decoder, self).__init__()
        if backbone == 'mobilenet':
            # mobile low lever output channel
            low_level_out_channels = 32
            middle_level_out_channels = 64
        else:
            raise NotImplementedError
        # 对低维数据进行处理
        self.low_conv = nn.Sequential(
            nn.Conv1d(low_level_out_channels, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU())

        # 对middle维数据进行处理
        self.middle_conv = nn.Sequential(
            nn.Conv1d(middle_level_out_channels, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU())

        # 处理合并后的数据
        self.last_conv = nn.Sequential(
            DepthSeparableConv1d(448, 256, 3, stride=1, padding=1, bias=False),
            nn.Dropout(0.5),
            DepthSeparableConv1d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.Dropout(0.2),
            nn.ConvTranspose1d(256, 256, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Conv1d(256, num_classes, kernel_size=1, stride=1),
            nn.Sigmoid())

    def forward(self, low_level_feat, middle_level_feat, x):
        print("final1", x.shape)
        low_level_feat = self.low_conv(low_level_feat)
        print("low", low_level_feat.shape)
        print("middle", middle_level_feat.shape)
        middle_level_feat = self.middle_conv(middle_level_feat)

        # conv after merge
        x = F.interpolate(x, size=middle_level_feat.size()[2:], mode='linear', align_corners=False)
        x = torch.cat((x, low_level_feat, middle_level_feat), dim=1)
        print("final1", x.shape)
        x = self.last_conv(x)
        print("final", x.shape)
        return x


class DeepLab(nn.Module):
    def __init__(self, backbone='mobilenet', output_stride=16, num_classes=1):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        self.backbone = build_backbone(backbone, in_channels=1)
        self.aspp = ASPP(backbone, output_stride)
        self.decoder = Decoder(num_classes, backbone)

    def forward(self, input):
        low_level_feat, middle_level_feat, x = self.backbone(input)

        x = self.aspp(x)
        x = self.decoder(low_level_feat, middle_level_feat, x)
        # 变为与输入 size 一样
        # print(x.size())
        x = F.interpolate(x, size=input.size()[2:], mode='linear')

        return [x]

    def loss_function(self, *args, gt) -> dict:
        recons = args[0]
        recons_loss_ = MSELoss()
        recons_loss = recons_loss_(recons.flatten(), gt.flatten())
        loss = recons_loss
        return loss


if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(10, 1, 640)
    output = model(input)
    print(output[0].size())
