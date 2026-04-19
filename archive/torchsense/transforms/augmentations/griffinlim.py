import torch
from torch import Tensor
from typing import Callable, Optional, Sequence, Tuple, Union
from torchaudio import functional as F


class GriffinLim(torch.nn.Module):
    r"""Compute waveform from a linear scale magnitude spectrogram using the Griffin-Lim transformation.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Implementation ported from
    *librosa* :cite:`brian_mcfee-proc-scipy-2015`, *A fast Griffin-Lim algorithm* :cite:`6701851`
    and *Signal estimation from modified short-time Fourier transform* :cite:`1172092`.

    Args:
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``400``)
        n_iter (int, optional): Number of iteration for phase recovery process. (Default: ``32``)
        win_length (int or None, optional): Window size. (Default: ``n_fft``)
        hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
        window_fn (Callable[..., Tensor], optional): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        power (float, optional): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for magnitude, 2 for power, etc. (Default: ``2``)
        wkwargs (dict or None, optional): Arguments for window function. (Default: ``None``)
        momentum (float, optional): The momentum parameter for fast Griffin-Lim.
            Setting this to 0 recovers the original Griffin-Lim method.
            Values near 1 can lead to faster convergence, but above 1 may not converge. (Default: ``0.99``)
        length (int, optional): Array length of the expected output. (Default: ``None``)
        rand_init (bool, optional): Initializes phase randomly if True and to zero otherwise. (Default: ``True``)

    Example
        >>> batch, freq, time = 2, 257, 100
        >>> spectrogram = torch.randn(batch, freq, time)
        >>> transform = transforms.GriffinLim(n_fft=512)
        >>> waveform = transform(spectrogram)
    """
    __constants__ = ["n_fft", "n_iter", "win_length", "hop_length", "power", "length", "momentum", "rand_init"]

    def __init__(
            self,
            n_fft: int = 400,
            n_iter: int = 32,
            win_length: Optional[int] = None,
            hop_length: Optional[int] = None,
            window_fn: Callable[..., Tensor] = torch.hann_window,
            power: float = 2.0,
            wkwargs: Optional[dict] = None,
            momentum: float = 0.99,
            length: Optional[int] = None,
            rand_init: bool = True,
    ) -> None:
        super(GriffinLim, self).__init__()

        if not (0 <= momentum < 1):
            raise ValueError("momentum must be in the range [0, 1). Found: {}".format(momentum))

        self.n_fft = n_fft
        self.n_iter = n_iter
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer("window", window.to(device))
        self.length = length
        self.power = power
        self.momentum = momentum
        self.rand_init = rand_init

    def forward(self, specgram: Tensor) -> Tensor:
        r"""
        Args:
            specgram (Tensor):
                A magnitude-only STFT spectrogram of dimension (..., freq, frames)
                where freq is ``n_fft // 2 + 1``.

        Returns:
            Tensor: waveform of (..., time), where time equals the ``length`` parameter if given.
        """
        return F.griffinlim(
            specgram,
            self.window,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.power,
            self.n_iter,
            self.momentum,
            self.length,
            self.rand_init,
        )
