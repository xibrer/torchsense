import os
import sys
import re
import datetime
from typing import Any

_IMU_DIR = os.path.dirname(os.path.abspath(__file__))
if _IMU_DIR not in sys.path:
    sys.path.insert(0, _IMU_DIR)

import numpy as np
import scipy.io as io
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset
import pandas as pd


def minmax_normalize(x: np.ndarray) -> np.ndarray:
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + 1e-8)


class MyDataset(Dataset):
    def __init__(
        self,
        mat_path,
        augment_cfg: dict[str, Any] | None = None,
        is_train: bool = False,
    ):
        """从单个目录或目录列表扫描 .mat，并在初始化时全部读入内存。

        Args:
            mat_path (str | list[str] | tuple[str, ...]): 单个受试者目录，或多个目录组成的
                列表/元组；列表内顺序决定样本在数据集中的拼接顺序。
            augment_cfg (dict[str, Any] | None): 增强配置字典。Defaults to None.
            is_train (bool): 是否为训练集。仅训练集应用增强。Defaults to False.
        """
        mat_paths = []
        self.is_train = is_train
        self.augment_cfg = augment_cfg or {}
        self._aug_enabled = bool(self.augment_cfg.get('enabled', False))
        self._jitter_cfg = self.augment_cfg.get('jitter', {})
        self._time_warp_cfg = self.augment_cfg.get('time_warp', {})
        self._random_mask_cfg = self.augment_cfg.get('random_mask', {})

        folders = mat_path if isinstance(mat_path, (list, tuple)) else [mat_path]
        for folder in folders:
            for file in os.listdir(folder):
                if not file.lower().endswith('.mat'):
                    continue
                mat_paths.append(os.path.join(folder, file))

        self.samples = []
        for path in mat_paths:
            data = io.loadmat(path)
            ecg = minmax_normalize(data['ecg'].astype(np.float32))
            scg = minmax_normalize(data['y'].astype(np.float32))
            self.samples.append((torch.from_numpy(scg), torch.from_numpy(ecg)))

    def __getitem__(self, index):
        scg, ecg = self.samples[index]
        if self.is_train and self._aug_enabled:
            scg = self._apply_augmentations(scg)
        return scg, ecg

    def __len__(self):
        return len(self.samples)

    def _apply_augmentations(self, x: torch.Tensor) -> torch.Tensor:
        """Apply enabled augmentations to one input sample."""
        out = x.clone().float()

        if self._jitter_cfg.get('enabled', False):
            out = self._augment_jitter(out, self._jitter_cfg)
        if self._time_warp_cfg.get('enabled', False):
            out = self._augment_time_warp(out, self._time_warp_cfg)
        if self._random_mask_cfg.get('enabled', False):
            out = self._augment_random_mask(out, self._random_mask_cfg)
        return out

    @staticmethod
    def _flatten_signal(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        """Flatten tensor to 1D while keeping original shape."""
        orig_shape = tuple(x.shape)
        x1d = x.reshape(-1)
        return x1d, orig_shape

    @staticmethod
    def _restore_signal(x1d: torch.Tensor, orig_shape: tuple[int, ...]) -> torch.Tensor:
        """Restore flattened 1D tensor to original shape."""
        return x1d.reshape(orig_shape)

    def _augment_jitter(self, x: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
        """Add Gaussian jitter noise."""
        sigma = float(cfg.get('sigma', 0.01))
        if sigma <= 0:
            return x
        noise = torch.randn_like(x) * sigma
        return torch.clamp(x + noise, 0.0, 1.0)

    def _augment_time_warp(self, x: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
        """Randomly stretch/compress time axis and resample back."""
        min_scale = float(cfg.get('min_scale', 0.9))
        max_scale = float(cfg.get('max_scale', 1.1))
        if min_scale <= 0 or max_scale <= 0:
            return x
        if min_scale > max_scale:
            min_scale, max_scale = max_scale, min_scale

        x1d, orig_shape = self._flatten_signal(x)
        length = int(x1d.shape[0])
        if length < 4:
            return x

        scale = float(np.random.uniform(min_scale, max_scale))
        warped_len = max(2, int(round(length * scale)))
        sig = x1d.view(1, 1, length)
        warped = torch.nn.functional.interpolate(sig, size=warped_len, mode='linear', align_corners=False)
        restored = torch.nn.functional.interpolate(warped, size=length, mode='linear', align_corners=False)
        out = restored.view(-1)
        return self._restore_signal(torch.clamp(out, 0.0, 1.0), orig_shape)

    def _augment_random_mask(self, x: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
        """Apply random temporal masking to segments."""
        max_masks = int(cfg.get('max_masks', 2))
        min_ratio = float(cfg.get('min_ratio', 0.01))
        max_ratio = float(cfg.get('max_ratio', 0.08))
        fill_mode = str(cfg.get('fill_mode', 'zero'))

        if max_masks <= 0:
            return x
        if min_ratio < 0:
            min_ratio = 0.0
        if max_ratio < min_ratio:
            max_ratio = min_ratio

        x1d, orig_shape = self._flatten_signal(x)
        length = int(x1d.shape[0])
        if length < 8:
            return x

        num_masks = int(np.random.randint(1, max_masks + 1))
        out = x1d.clone()
        fill_val = 0.0 if fill_mode == 'zero' else float(out.mean().item())

        for _ in range(num_masks):
            ratio = float(np.random.uniform(min_ratio, max_ratio))
            seg_len = max(1, int(round(length * ratio)))
            if seg_len >= length:
                seg_len = length - 1
            start = int(np.random.randint(0, max(1, length - seg_len)))
            end = start + seg_len
            out[start:end] = fill_val

        return self._restore_signal(torch.clamp(out, 0.0, 1.0), orig_shape)


patient_list = ['20-m_train', '21-m_train', '22-f_train', '23-m_train',
                '27-f_train', '34-f_train', '35-m-2_train', '36-f_train',
                '37-f_train', '52-f_train', '54-m_train', '58-m_train',
                'AMI_train', 'FC_train', ]


def get_network(args):
    """ return given network
    """

    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'lstm':
        from models.rnn import lstm 
        net =lstm()
    elif args.net == 'gru':
        from models.rnn import gru 
        net =gru()
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.net == 'ae64':
        from models.Autoencoder import Autoencoder
        net = Autoencoder(1, 64)
    elif args.net == 'ae128':
        from models.Autoencoder import Autoencoder
        net = Autoencoder(1, 128)
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.net == 'xception':
        from models.xception import xception
        net = xception()
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.net in ('deeplab', 'deeplab1', 'deeplab2', 'deeplab3', 'deeplab4', 'deeplab5', 'deeplab6'):
        from models.DeepLabV3.DeepLabV3Plus import DeepLab
        net = DeepLab()
    elif args.net == 'unet':
        from models.unet1d import UNet1d
        net = UNet1d()
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu:  # use_gpu
        net = net.cuda()

    return net


def show_parameter(_model):
    inputs = torch.rand(1, 1, 3840).cuda()
    flops, params = profile(_model, inputs=(inputs,))
    print('Total params: %.2fM' % (params / 1000000.0))
    print('Total flops: %.2fM' % (flops / 1000000.0))


def merge_excel(file_dir):
    # 找到文件路径下的所有表格名称，返回列表
    file_list = os.listdir(file_dir)
    print(file_list)
    new_list = {"rr": [], "pr": [], "qrs": [], "qt": [], "rmse": [], "cs": []}
    wave_list = ["rr", "pr", "qrs", "qt", "rmse", "cs"]
    for file in file_list:
        # 重构文件路径
        file_path = os.path.join(file_dir, file)
        # 将excel转换成DataFrame
        dataframe = pd.read_excel(file_path)
        for wave in wave_list:
            df = dataframe[wave]
            # 保存到新列表中
            new_list[wave].append(df)

    # 多个DataFrame合并为一个
    for key in new_list:
        print(key)
        df = pd.concat(new_list[key], axis=1)
        # 写入到一个新excel表中
        # print(df)
        patt4 = r'rec_data'
        name = re.sub(patt4, "split_data", file_dir)
        if not os.path.exists(name):
            os.makedirs(name)
        new_filename = name + key + ".csv"
        df.to_csv(new_filename, index=False, mode='w')
    print("已保存", new_filename)
