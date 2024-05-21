from .utils import load_file
from .folder import DatasetFolder
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from torchsense.transforms import (tensor_has_valid_audio_batch_dimension,
                                   add_audio_batch_dimension, remove_audio_batch_dimension)
IMG_EXTENSIONS = (".mat", ".jpeg", ".npz")


def default_loader(path: str, params: list) -> Any:
    return load_file(path, params)


class SensorFolder(DatasetFolder):
    """A generic data1 loader where the images are arranged in this way by default: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png

    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory path.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
        allow_empty(bool, optional): If True, empty folders are considered to be valid classes.
            An error is raised on empty folders if False (default).

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            params: Tuple[List[str], Optional[List[str]]],
            pre_model=None,
            stage_transform: Optional[Callable] = None,
            transform: Union[Optional[Callable], List[Callable]] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str, Any], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            allow_empty: bool = False,
    ):
        super().__init__(
            root,
            params,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
            allow_empty=allow_empty,
        )
        self.imgs = self.samples
        if pre_model is None:
            self.pre_model = None
            self.stage_transform = None

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, labels = self.samples[index]
        sample, target = self.loader(path, self.params)
        if len(sample) == 1:
            sample = sample[0]
        if len(target) == 1:
            target = target[0]
        if self.transform is not None:
            for i in range(len(self.transform)):
                sample[i] = self.transform[i](sample[i])
        if self.pre_model is not None:
            if tensor_has_valid_audio_batch_dimension(sample):
                sample = add_audio_batch_dimension(sample)
            stage_out = self.pre_model(tuple(sample))
            stage_out = remove_audio_batch_dimension(stage_out)
            if self.stage_transform is not None:
                sample = self.stage_transform(stage_out)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, labels


class AudioFolder(DatasetFolder):
    """A generic data1 loader where the images are arranged in this way by default: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png

    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory path.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
        allow_empty(bool, optional): If True, empty folders are considered to be valid classes.
            An error is raised on empty folders if False (default).

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            params: Tuple[List[str], Optional[List[str]]] = None,
            pre_model=None,
            stage_transform: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str, Any], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            allow_empty: bool = False,
    ):
        super().__init__(
            root,
            params,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
            allow_empty=allow_empty,
        )
        self.imgs = self.samples
        self.pre_model = pre_model
        self.stage_transform = stage_transform
