import torch
from .functional import negative_si_snr
import torch.nn.functional as F

class ScaleInvariantSignalNoiseRatio(torch.nn.Module):
    def __init__(self):
        super(ScaleInvariantSignalNoiseRatio, self).__init__()

    def forward(self, x, s):
        return negative_si_snr(x, s)

