"""Visualize before/after comparisons for data augmentation samples."""

from __future__ import annotations

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import MyDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='可视化数据增强前后对比')
    p.add_argument(
        '--train_dir',
        type=str,
        default='',
        help='ecg_train_set 根目录；默认 examples/imu/dataset/ecg_train_set',
    )
    p.add_argument(
        '--subjects',
        nargs='+',
        default=['wxw_train'],
        help='要读取的受试者目录列表',
    )
    p.add_argument(
        '--augment_cfg',
        type=str,
        default='',
        help='增强 YAML 路径；默认 examples/imu/augment.yaml',
    )
    p.add_argument('--num_samples', type=int, default=10, help='可视化样本数')
    p.add_argument('--seed', type=int, default=42, help='随机种子')
    p.add_argument(
        '--out_path',
        type=str,
        default='',
        help='输出图片路径；默认 examples/imu/eval_vis/augment_before_after_10.png',
    )
    return p.parse_args()


def load_augmentation_config(cfg_path: str) -> dict:
    """Load YAML augmentation config and return augmentation section."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError('未安装 PyYAML，请先执行: pip install pyyaml') from exc

    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f'未找到增强配置: {cfg_path}')
    with open(cfg_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    aug = cfg.get('augmentation', {})
    if not isinstance(aug, dict):
        raise ValueError('增强配置格式错误，需包含 augmentation 字典')
    return aug


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    imu_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = args.train_dir or os.path.join(imu_dir, 'dataset', 'ecg_train_set')
    cfg_path = args.augment_cfg or os.path.join(imu_dir, 'augment.yaml')
    out_path = args.out_path or os.path.join(imu_dir, 'eval_vis', 'augment_before_after_10.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    subjects = [os.path.join(train_dir, s) for s in args.subjects]
    aug_cfg = load_augmentation_config(cfg_path)

    raw_ds = MyDataset(subjects, augment_cfg=None, is_train=False)
    aug_ds = MyDataset(subjects, augment_cfg=aug_cfg, is_train=True)
    if len(raw_ds) == 0:
        raise RuntimeError('数据集为空，请检查 train_dir 与 subjects。')

    n = min(max(args.num_samples, 1), len(raw_ds))
    indices = random.sample(range(len(raw_ds)), n)

    fig, axes = plt.subplots(n, 1, figsize=(14, 2.2 * n), squeeze=False)
    for row, idx in enumerate(indices):
        before_x, _ = raw_ds[idx]
        after_x, _ = aug_ds[idx]
        before = np.asarray(before_x.detach().cpu()).ravel()
        after = np.asarray(after_x.detach().cpu()).ravel()
        t = np.arange(min(len(before), len(after)))

        ax = axes[row][0]
        ax.plot(t, before[: len(t)], color='tab:blue', linewidth=1.0, alpha=0.9, label='before')
        ax.plot(t, after[: len(t)], color='tab:red', linewidth=1.0, alpha=0.8, label='after')
        ax.set_title(f'sample idx={idx}')
        if row == 0:
            ax.legend(loc='upper right')
        ax.set_xlabel('t')

    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f'saved figure: {out_path}')


if __name__ == '__main__':
    main()
