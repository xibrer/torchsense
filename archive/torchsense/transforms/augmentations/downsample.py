import torch
import numpy.random as random
from .utils import *


class DownSample(torch.nn.Module):
    def __init__(self, scale_factor=1):
        """
        :param scale_factor: Scale factor
        """
        super().__init__()
        
        self.scale_factor = scale_factor
        

    def forward(self, x):
        
        x = x.squeeze(0)
        x = x[::self.scale_factor]
        x = x.unsqueeze(0)
        return x
