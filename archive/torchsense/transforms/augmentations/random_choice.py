import numpy as np
import torch
import numpy.random as random
from .utils import has_batch_dimension,remove_batch_dimension,add_batch_dimension

class RandomChoice(torch.nn.Module):
    def __init__(self, length=1000):
        """
        :param min_snr: Minimum signal-to-noise ratio
        :param max_snr: Maximum signal-to-noise ratio
        """
        super().__init__()
        self.length = length

    def forward(self, x):

        # begin_point = random.randint(0, x.shape[1] - self.length)
        begin_point = 1
        if has_batch_dimension(x):
            x = remove_batch_dimension(x)
            x = x[:, begin_point:begin_point + self.length]
            x = add_batch_dimension(x)
        else:
            x = x[:, begin_point:begin_point + self.length]
        return x