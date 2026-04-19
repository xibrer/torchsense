
import time

import torch
import torch.nn as nn
from einops import rearrange


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=None):
        super(BasicBlock, self).__init__()
        self.basic = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.basic(x)


class DSConv1d(nn.Module):

    def __init__(self, input_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.Depthwise = nn.Sequential(
            nn.Conv1d(input_channels, input_channels,
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


class DSConv2d(nn.Module):

    def __init__(self, input_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.Depthwise = nn.Sequential(
            nn.Conv2d(input_channels, input_channels,
                      kernel_size, groups=input_channels, **kwargs),
            nn.BatchNorm2d(input_channels),
            nn.LeakyReLU(inplace=True)
        )

        self.Pointwise = nn.Sequential(
            nn.Conv2d(input_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.Depthwise(x)
        x = self.Pointwise(x)
        return x


class LinearBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, t=6, class_num=100):
        super().__init__()

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


class FTB(nn.Module):

    def __init__(self, input_dim=257, in_channel=96, r_channel=5):
        super(FTB, self).__init__()

        self.in_channel = in_channel
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, r_channel, kernel_size=(1, 1)),
            nn.BatchNorm2d(r_channel),
            nn.ReLU()
        )

        self.conv1d = nn.Sequential(
            nn.Conv1d(r_channel * input_dim, in_channel, kernel_size=9, padding=4),
            nn.BatchNorm1d(in_channel),
            nn.ReLU()
        )

        self.freq_fc = nn.Linear(input_dim, input_dim, bias=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel, kernel_size=(1, 1)),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )

    def forward(self, inputs):
        '''
        inputs should be [Batch, Ca, Dim, Time]
        '''

        # T-F attention
        conv1_out = self.conv1(inputs)

        B, C, D, T = conv1_out.size()
        reshape1_out = torch.reshape(conv1_out, [B, C * D, T])
        conv1d_out = self.conv1d(reshape1_out)
        conv1d_out = torch.reshape(conv1d_out, [B, self.in_channel, 1, T])

        # now is also [B,C,D,T]
        att_out = conv1d_out * inputs

        # tranpose to [B,C,T,D]
        att_out = torch.transpose(att_out, 2, 3)
        freqfc_out = self.freq_fc(att_out)
        att_out = torch.transpose(freqfc_out, 2, 3)

        cat_out = torch.cat([att_out, inputs], 1)
        outputs = self.conv2(cat_out)
        return outputs


class TFS_AttConv(nn.Module):
    def __init__(self, input_dim=257, channel_amp=96):
        super().__init__()
        self.ftb1 = FTB(input_dim=input_dim,
                        in_channel=channel_amp,
                        )
        self.amp_conv1 = nn.Sequential(
            nn.Conv2d(channel_amp, channel_amp, kernel_size=5, padding=2),
            nn.BatchNorm2d(channel_amp),
            nn.LeakyReLU()
        )

        self.amp_conv2 = nn.Sequential(
            nn.Conv2d(channel_amp, channel_amp, kernel_size=(1, 25), padding=(0, 12)),
            nn.BatchNorm2d(channel_amp),
            nn.LeakyReLU()
        )

        self.amp_conv3 = nn.Sequential(
            nn.Conv2d(channel_amp, channel_amp, kernel_size=5, padding=2),
            nn.BatchNorm2d(channel_amp),
            nn.LeakyReLU()
        )

        self.ftb2 = FTB(input_dim=input_dim, in_channel=channel_amp)

    def forward(self, amp):
        '''
        amp should be [Batch, Ca, Dim, Time]
        amp should be [Batch, Cr, Dim, Time]
        '''

        amp_out = self.ftb1(amp)
        amp_out = self.amp_conv1(amp_out)
        amp_out = self.amp_conv2(amp_out)
        amp_out = self.amp_conv3(amp_out)
        amp_out = self.ftb2(amp_out)
        return amp_out


class TF_Domain_Noise(nn.Module):
    def __init__(self, channel_amp=96):
        super(TF_Domain_Noise, self).__init__()
        self.amp_conv1 = nn.Sequential(
            nn.Conv2d(1, channel_amp,
                      kernel_size=[1, 7],
                      padding=(0, 3)
                      ),
            nn.BatchNorm2d(channel_amp),
            nn.LeakyReLU(),
            nn.Conv2d(channel_amp, channel_amp,
                      kernel_size=[7, 1],
                      padding=(3, 0)
                      ),
            nn.BatchNorm2d(channel_amp),
            nn.LeakyReLU(),
        )
        self.amp_conv2 = nn.Sequential(
            nn.Conv2d(channel_amp, 1, kernel_size=[1, 1]),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
        )
        self.tfss = nn.ModuleList()
        for idx in range(3):
            self.tfss.append(TFS_AttConv())

    def forward(self, amp):
        time_start = time.time()
        spec = self.amp_conv1(amp)

        s_spec = [spec]
        for idx, layer in enumerate(self.tfss):
            if idx != 0:
                s_spec.append(s_spec[0] + s_spec[-1])
            s_spec.append(layer(s_spec[-1]))
        out = self.amp_conv2(s_spec[-1])
        # s_spec = spec
        # for idx, layer in enumerate(self.tfss):
        #     if idx != 0:
        #         spec += s_spec
        #     spec = layer(spec)
        # out = self.amp_conv2(spec)

        return out


class TFU_Conv(nn.Module):
    def __init__(self, in_channels=96):
        super(TFU_Conv, self).__init__()
        self.module = nn.Sequential(
            BasicBlock(in_channels=in_channels, kernel_size=(3, 5), out_channels=in_channels, padding=(1, 2)),
            BasicBlock(in_channels=in_channels, kernel_size=(9, 1), out_channels=in_channels, padding=(4, 0)),
            BasicBlock(in_channels=in_channels, kernel_size=(3, 5), out_channels=in_channels, padding=(1, 2), )
        )

    def forward(self, x):
        return self.module(x)


class TF_Domain_ACC(nn.Module):
    def __init__(self, channel_amp=96):
        super(TF_Domain_ACC, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, channel_amp,
                      kernel_size=(1, 5),
                      padding=(0, 2)
                      ),
            nn.BatchNorm2d(channel_amp),
            nn.LeakyReLU(),
            nn.Conv2d(channel_amp, channel_amp,
                      kernel_size=(7, 1),
                      padding=(3, 0)
                      ),
            nn.BatchNorm2d(channel_amp),
            nn.LeakyReLU(),
        )
        self.tfus = nn.ModuleList()
        for idx in range(3):
            self.tfus.append(TFU_Conv())
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel_amp, 1, kernel_size=(1, 1)),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        time_start = time.time()
        spec = self.conv1(x)
        # print("generator-acc:1 conv layers"+str(time_start-time.time()))
        # time_start=time.time()
        s_spec = []
        s_spec.append(spec)

        for idx, layer in enumerate(self.tfus):
            if idx != 0:
                s_spec.append(s_spec[0] + s_spec[-1])
            s_spec.append(layer(s_spec[-1]))
        # print("generator-acc:3 TFU conv layer"+str(time_start-time.time()))
        # time_start=time.time() 
        out = self.conv2(s_spec[-1])
        # print("generator-acc:1 conv layer"+str(time_start-time.time()))
        # time_start=time.time() 
        return out


# 输入的是 带噪声的时频谱（amp）和加速计的时频谱(acc)（1，257，498）  （1，51，498）
# 输出是  增强的amp  groudtruth 纯净的amp
class Generator(nn.Module):
    def __init__(self, lstm_nums=300):
        super().__init__()
        self.noise = TF_Domain_Noise()
        self.acc = TF_Domain_ACC()
        # 注意力融合--
        self.att = FTB(input_dim=257, in_channel=1)
        # TODO 自定义实现lstm
        self.lstm = nn.LSTM(input_size=257 * 1, hidden_size=lstm_nums, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(lstm_nums * 2, 600),
            nn.LeakyReLU(),
            nn.Linear(600, 600),
            nn.LeakyReLU(),
            nn.Linear(600, 257),
            
        )

    def forward(self, inputs, training=True):
        acc, noise = inputs
        noise_extraction = self.noise(noise)
        acc_extraction = self.acc(acc)
        features = torch.concat((noise_extraction, acc_extraction), dim=2)
        att = self.att(noise_extraction)
        att = rearrange(att, 'b 1 f t -> b t f')
        lstm = self.lstm(att)[0]
        out = self.fc(lstm)
        if not training:
            print(f"out:{out.max()}")
            print(f"noise:{noise.max()}")
        out = out.float()
        out = torch.sigmoid(out)
        mask = rearrange(out, 'b t f -> b 1 f t')
        output = mask * noise

        return output
