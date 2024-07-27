import torch
from typing import Any, Optional, Union
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_from_ckpt(model, path: Union[str, Path], load_location=None):
    """
    Load model weights from a checkpoint file.

    Args:
        model (nn.Module): The model to load the weights into.
        path (str or Path): The path to the checkpoint file.
        load_location (str, optional): Location to load the model, either 'cpu' or 'cuda'. Defaults to None.

    Returns:
        nn.Module: The model with loaded weights.
    """
    if load_location == 'cpu':
        checkpoint = torch.load(path, map_location='cpu')
    else:
        checkpoint = torch.load(path, map_location=torch.device('cuda') if torch.cuda.is_available() else 'cpu')
    
    # Check if 'state_dict' is in the checkpoint, otherwise assume it's the state dict itself
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Remove prefix 'model.' if present
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    # Remove prefix 'student.' if present
    state_dict = {k.replace("student.", ""): v for k, v in state_dict.items()}
    # Remove keys containing 'teacher'
    state_dict = {k: v for k, v in state_dict.items() if "teacher" not in k}

    # Load the state dictionary into the model
    model.load_state_dict(state_dict)
    model.eval()
    return model
