import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchinfo import summary


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv1d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv1d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv1d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv1d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv1d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nf=64, nb=23, gc=32, sf=4):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.sf = sf
        # print([in_nc, out_nc, nf, nb, gc, sf])

        self.conv_first = nn.Conv1d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
        if self.sf == 4:
            self.upconv2 = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv1d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        if self.sf == 4:
            fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        # 换
        self.conv1 = nn.Conv1d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        x = torch.cat([max_pool, avg_pool], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class AttentionRRDBNet(nn.Module):
    def __init__(self, sf=4, original_generator=RRDBNet(in_nc=1, out_nc=1, sf=4)):
        super(AttentionRRDBNet, self).__init__()
        self.generator = original_generator
        self.attention = SpatialAttention()
        self.upsample = nn.Upsample(scale_factor=4, mode='linear', align_corners=True)  # 可根据需要调整上采样方式

    def forward(self, x):
        attention_map = self.attention(x)
        generated = self.generator(x)
        # print(attention_map.shape,generated.shape)
        # 确保 attention_map 和 generated 形状一致
        if attention_map.size(-1) != generated.size(-1):
            attention_map = self.upsample(attention_map)

        return generated * attention_map


# 测试脚本
if __name__ == "__main__":
    # 创建一个RRDBNet实例
    model = RRDBNet(in_nc=1, out_nc=1, sf=4)

    # 创建一个随机输入张量，形状为 (batch_size, channels, height, width)
    x = torch.randn(10, 1, 100)
    summary(model, input_size=x.shape)
    # 前向传播
    with torch.no_grad():
        output = model(x)

    # 打印输出形状
    print("Output shape:", output.shape)
