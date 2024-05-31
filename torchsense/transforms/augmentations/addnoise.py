import numpy as np
import torch
import numpy.random as random
from .utils import load_file
import os
from pathlib import Path


class AddNoise(torch.nn.Module):

    def __init__(self, noise_file_paths: str = None,
                 key="acc[2]",
                 add_prob: float = 0.8):
        """
        :param noise_file_paths: file path for noise
        """
        super().__init__()
        self.noise_list = []
        self.key = key
        self.add_prob = add_prob
        noise_file_paths = Path(noise_file_paths)
        for path in sorted(noise_file_paths.rglob('*')):
            full_path_str = str(path)  # Get the full path string
            if path.is_file() and "clean" not in full_path_str:
                # print(full_path_str)
                self.noise_list.append(full_path_str)

    def forward(self, x):
        if random.rand() > self.add_prob:
            # print("No noise added")
            return x
        index = random.randint(0, len(self.noise_list))
        noise_file_path = self.noise_list[index]
        noise = load_file(noise_file_path, self.key)
        x = x + noise[0].reshape(1, -1)
        return x
