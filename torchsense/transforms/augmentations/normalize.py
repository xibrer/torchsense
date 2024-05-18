import numpy as np
import torch
import numpy.random as random


class Normalize(torch.nn.Module):
    def __init__(self, new_min=0, new_max=1):
        """
        :param min_snr: Minimum signal-to-noise ratio
        :param max_snr: Maximum signal-to-noise ratio
        """
        super().__init__()
        self.min_snr = new_min
        self.max_snr = new_max

    def forward(self, x):

        x = (x - x.min()) / (x.max() - x.min())  # 先将x归一化到0~1
        x = x * (self.min_snr - self.max_snr) + self.max_snr  # 然后将x的范围调整到new_min~new_max
        return x
