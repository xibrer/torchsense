import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_from_ckpt(model, path,load_location=None):
    """
    Load model weights from a checkpoint file.

    Args:
        model (nn.Module): The model to load the weights into.
        path (str): The path to the checkpoint file.

    Returns:
        nn.Module: The model with loaded weights.
    """
    if load_location == 'cpu':
        checkpoint = torch.load(path,map_location='cpu')
    else:
        checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["state_dict"]
    new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    # Load the new state dictionary into the model
    model.load_state_dict(new_state_dict)
    return model
