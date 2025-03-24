import numpy as np
import numpy.random as random
import torch

from .utils import add_batch_dimension, has_batch_dimension, remove_batch_dimension


class PreModule(torch.nn.Module):
    """
    A module that applies preprocessing operations to input data before passing it through a model.
    
    Args:
        model (torch.nn.Module): The model to be applied to the input data.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model.cuda()
        self.model.eval()
        
    def forward(self, x):
        """
        Forward pass of the PreModule.
        
        Args:
            x (torch.Tensor): The input data.
        
        Returns:
            torch.Tensor: The output of the model after applying preprocessing operations.
        """
        if not has_batch_dimension(x):
            x = add_batch_dimension(x)
        x = x.cuda()
        x = self.model(x)
        x = x.cpu()
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0].detach()
        else:
            x = x.detach()
        if has_batch_dimension:
            remove_batch_dimension(x)
        return x
