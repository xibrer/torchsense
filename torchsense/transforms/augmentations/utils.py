import torch
from typing import Union, List, Tuple


def tensor_has_valid_audio_batch_dimension(tensor) -> bool:
    if isinstance(tensor, list) or isinstance(tensor, tuple):
        # Check if each element in the list has a valid length (3 dimensions)
        for element in tensor:
            if not hasattr(element, 'ndim') or element.ndim != 3:
                return False
        return True
    if tensor.ndim == 3:
        return True
    return False


def add_audio_batch_dimension(tensor: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]) -> Union[
    torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]:
    # Function to unsqueeze tensor along the specified dimension
    def unsqueeze_element(element: torch.Tensor) -> torch.Tensor:
        if isinstance(element, torch.Tensor):
            return element.unsqueeze(dim=0)
        raise ValueError("All elements must be torch tensors")

    # Check if the input is a list
    if isinstance(tensor, list):
        return [unsqueeze_element(element) for element in tensor]

    # Check if the input is a tuple
    if isinstance(tensor, tuple):
        return tuple(unsqueeze_element(element) for element in tensor)

    # Check if the input is a tensor
    if isinstance(tensor, torch.Tensor):
        return tensor.unsqueeze(dim=0)

    # If the input is neither a list, tuple, nor tensor, raise an error
    raise ValueError("Input must be a torch tensor, a list of tensors, or a tuple of tensors")


def remove_audio_batch_dimension(tensor: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]:
    # Function to squeeze tensor along the specified dimension
    def squeeze_element(element: torch.Tensor) -> torch.Tensor:
        if isinstance(element, torch.Tensor):
            return element.squeeze(dim=0)
        raise ValueError("All elements must be torch tensors")

    # Check if the input is a list
    if isinstance(tensor, list):
        return [squeeze_element(element) for element in tensor]

    # Check if the input is a tuple
    if isinstance(tensor, tuple):
        return tuple(squeeze_element(element) for element in tensor)

    # Check if the input is a tensor
    if isinstance(tensor, torch.Tensor):
        return tensor.squeeze(dim=0)

    # If the input is neither a list, tuple, nor tensor, raise an error
    raise ValueError("Input must be a torch tensor, a list of tensors, or a tuple of tensors")
