import numpy as np
import numpy.random as random
import torch

from .utils import has_batch_dimension, remove_batch_dimension, add_batch_dimension


class AddRandomNoise(torch.nn.Module):
    def __init__(self, prob=0.9, noise_factor=0.1):
        super().__init__()
        self.prob = prob
        self.noise_factor = noise_factor  # Factor to scale noise relative to the input

    def forward(self, x):
        if not self.training:
            return x

        # Check if we should add noise based on the probability
        if random.rand() > self.prob:
            return x

        # Calculate noise standard deviation based on the input tensor's statistics
        mean = x.mean()
        std = x.std()
        noise_std = self.noise_factor * std

        # Generate random noise
        noise = torch.randn_like(x) * noise_std

        # Add noise to the input tensor
        return x + noise
