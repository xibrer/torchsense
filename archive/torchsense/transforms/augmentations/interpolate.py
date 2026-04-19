import torch
import numpy.random as random
from .utils import *


class Interpolate(torch.nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='linear'):
        """
        :param min_snr: Minimum signal-to-noise ratio
        :param max_snr: Maximum signal-to-noise ratio
        """
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        if not has_batch_dimension(x):
            x = add_batch_dimension(x)
        x = torch.nn.functional.interpolate(x, size=self.size, mode=self.mode)
        x = remove_batch_dimension(x)
        return x
