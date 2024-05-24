import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_from_ckpt(model, path):
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["state_dict"]
    new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    # Load the new state dictionary into the model
    model.load_state_dict(new_state_dict)
    return model
