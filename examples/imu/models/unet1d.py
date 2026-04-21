"""一维 U-Net，用于 SCG→ECG 等序列到序列回归（输入输出与 DeepLab 分支对齐）。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import HuberLoss


class _DoubleConv1d(nn.Module):
    """两次 3×1 卷积 + BN + ReLU。"""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet1d(nn.Module):
    """1D U-Net：输入 (N, 1, L)，输出与输入等长的单通道预测。

    forward 返回 ``[out]``，与 ``examples/imu/models/DeepLabV3`` 中 DeepLab 一致，
    便于 ``train.py`` / ``eval.py`` 共用同一套逻辑。

    Args:
        in_channels (int): 输入通道数。默认为 1。
        out_channels (int): 输出通道数。默认为 1。
        base_channels (int): 最浅层特征通道数。默认为 64。
        depth (int): 下采样级数（MaxPool 次数）。默认为 4。
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        depth: int = 4,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError('depth must be >= 1')

        self._depth = depth
        self._recons_loss = HuberLoss()

        chs = [base_channels * (2**i) for i in range(depth)]
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = in_channels
        for c in chs:
            self.downs.append(_DoubleConv1d(prev, c))
            self.pools.append(nn.MaxPool1d(2))
            prev = c

        self.bottleneck = _DoubleConv1d(prev, prev * 2)
        bridge_ch = prev * 2

        self.upsamples = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        dec_chs = list(reversed(chs))
        in_dec = bridge_ch
        for skip_ch in reversed(chs):
            self.upsamples.append(nn.Upsample(scale_factor=2, mode='linear', align_corners=False))
            self.up_convs.append(_DoubleConv1d(in_dec + skip_ch, skip_ch))
            in_dec = skip_ch

        self.out_conv = nn.Sequential(
            nn.Conv1d(dec_chs[-1], out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        target_len = x.size(2)

        skips: list[torch.Tensor] = []
        for down, pool in zip(self.downs, self.pools, strict=True):
            x = down(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, up_conv, skip in zip(self.upsamples, self.up_convs, reversed(skips), strict=True):
            x = up(x)
            if x.size(2) != skip.size(2):
                x = F.interpolate(x, size=skip.size(2), mode='linear', align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = up_conv(x)

        x = self.out_conv(x)
        if x.size(2) != target_len:
            x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
        return [x]

    def loss_function(self, *args: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """与 DeepLab.loss_function 相同签名：Huber(recons, gt)。"""
        recons = args[0]
        return self._recons_loss(recons.flatten(), gt.flatten())


if __name__ == '__main__':
    m = UNet1d()
    m.eval()
    inp = torch.randn(4, 1, 3840)
    out = m(inp)
    print(out[0].shape)
