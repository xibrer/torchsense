import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import HuberLoss

from models.DeepLabV3.backbone import build_backbone
from models.DeepLabV3.backbone.mobilenet import DepthSeparableConv1d
from models.DeepLabV3.aspp import build_aspp


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
        low_level_feat = self.low_conv(low_level_feat)
        middle_level_feat = self.middle_conv(middle_level_feat)

        # conv after merge
        x = F.interpolate(x, size=middle_level_feat.size()[2:], mode='linear', align_corners=False)
        x = torch.cat((x, low_level_feat, middle_level_feat), dim=1)
        x = self.last_conv(x)
        return x


class DeepLab(nn.Module):
    def __init__(self, backbone='mobilenet', output_stride=16, num_classes=1):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        self.backbone = build_backbone(backbone, in_channels=1)
        self.aspp = build_aspp(backbone, output_stride)
        self.decoder = Decoder(num_classes, backbone)
        self._recons_loss = HuberLoss()

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
        return self._recons_loss(recons.flatten(), gt.flatten())


if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(10, 1, 640)
    output = model(input)
    print(output[0].size())
