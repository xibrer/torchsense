import numpy as np
import torch
import numpy.random as random
from .utils import load_file
import os
from pathlib import Path


class AddNoise(torch.nn.Module):

    def __init__(self, 
                 noise_file_paths: str = None,
                 key="acc[2]",
                 add_prob: float = 1
                 ):
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
            if path.is_file():
                # print(full_path_str)
                self.noise_list.append(full_path_str)

    def forward(self, x):
        if random.rand() > self.add_prob:
            # print("No noise added")
            return x
        index = random.randint(0, len(self.noise_list))
        enhance_amptitude = random.uniform(1, 3)
        enhance_amptitude = 3
        noise_file_path = self.noise_list[index]
        noise = load_file(noise_file_path, self.key)
        # print(noise[0])
        noise = noise[0]*enhance_amptitude
        x = x + noise.reshape(1, -1)
        return x
