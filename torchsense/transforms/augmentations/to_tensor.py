from torch import nn
import torch
import numpy as np


class ToTensor:
    """Convert a PIL Image or ndarray to tensor and scale the values accordingly.

    This transform does not support torchscript.

   """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            pic = torch.from_numpy(pic.reshape(1, -1).astype(np.float32))
            # backward compatibility
            return pic
        else:
            return torch.tensor(pic, dtype=torch.float32)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
