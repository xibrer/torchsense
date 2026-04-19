import numpy as np
import torch
import numpy.random as random
import torchaudio


class HighPassBiquad(torch.nn.Module):
    """
    A high-pass biquad filter module for audio signal processing.

    Args:
        Fs (int): The sample rate of the input audio signal. Default is 400.
        cutoff_freq (int): The cutoff frequency of the high-pass filter in Hz. Default is 25.

    Returns:
        torch.Tensor: The filtered audio signal.

    Examples:
        >>> filter = HighPassBiquad(Fs=8000, cutoff_freq=1000)
        >>> input_audio = torch.randn(1, 1, 8000)
        >>> output_audio = filter(input_audio)
    """

    def __init__(self, Fs=400, cutoff_freq=25):
        super().__init__()
        self.Fs = Fs
        self.cutoff_freq = cutoff_freq

    def forward(self, x):
        # 进行数据填充以减轻边界效应
        x = x.reshape(-1)
        padding_size = 100  # 选择合适的填充大小
        padded_audio = torch.cat([x[:padding_size].flip(0), x, x[-padding_size:].flip(0)])
        x = torchaudio.functional.highpass_biquad(padded_audio, self.Fs, self.cutoff_freq)
        x = x[padding_size:-padding_size]
        x = x.reshape(1,-1)
        return x
