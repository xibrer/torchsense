import torch
from pathlib import Path
from typing import Any, Optional, Union

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_from_ckpt(model, path: Union[str, Path], load_location=None):
    """
    Load model weights from a checkpoint file.

    Args:
        model (nn.Module): The model to load the weights into.
        path (str or Path): The path to the checkpoint file.
        load_location (str, optional): Location to map the checkpoint. Defaults to None.

    Returns:
        nn.Module: The model with loaded weights.
    """
    # Load the checkpoint
    if load_location == 'cpu':
        checkpoint = torch.load(path, map_location='cpu')
    else:
        checkpoint = torch.load(path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Check if 'state_dict' is in the checkpoint
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

        # Remove 'model.' prefix from keys if present
        new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

        # Remove 'student.' prefix from keys if present
        new_state_dict = {k.replace("student.", ""): v for k, v in new_state_dict.items()}

        # Remove keys containing 'teacher'
        final_state_dict = {k: v for k, v in new_state_dict.items() if "teacher" not in k}

        # Load the state dictionary into the model
        model.load_state_dict(final_state_dict, strict=False)
    else:
        # If no 'state_dict', assume checkpoint is the state_dict
        model.load_state_dict(checkpoint, strict=False)

    # Set model to evaluation mode
    model.eval()

    return model
