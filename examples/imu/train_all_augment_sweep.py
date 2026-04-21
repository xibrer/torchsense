"""Batch train IMU models with and without augmentation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Sequence


NETS = ['deeplab6', 'unet']


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='批量训练：每个模型跑无增强+有增强两轮')
    p.add_argument('--max_epoch', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--save_name', type=str, default='sweep')
    p.add_argument('--tag', type=str, default='sweep')
    p.add_argument('--log_root', type=str, default='log')
    p.add_argument('--log_name', type=str, default='sweep.txt')
    p.add_argument('--gpu', action='store_true', default=True)
    p.add_argument('--dry_run', action='store_true', help='仅打印命令，不执行')
    return p.parse_args()


def _build_cmd(
    train_script: str,
    net: str,
    augment_cfg: str,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        sys.executable,
        train_script,
        '--net',
        net,
        '--max_epoch',
        str(args.max_epoch),
        '--batch_size',
        str(args.batch_size),
        '--save_name',
        args.save_name,
        '--tag',
        args.tag,
        '--log_root',
        args.log_root,
        '--log_name',
        args.log_name,
        '--augment_cfg',
        augment_cfg,
    ]
    if args.gpu:
        cmd.append('-gpu')
    return cmd


def run_once(
    train_script: str,
    net: str,
    augment_cfg: str,
    args: argparse.Namespace,
) -> None:
    cmd = _build_cmd(train_script, net, augment_cfg, args)
    print(f'\n[RUN] net={net} cfg={augment_cfg}')
    print(' '.join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    imu_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(imu_dir, 'train.py')
    cfg_off = os.path.join(imu_dir, 'augment_off.yaml')
    cfg_on = os.path.join(imu_dir, 'augment.yaml')

    missing = [p for p in (train_script, cfg_off, cfg_on) if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f'缺少必要文件: {missing}')

    for net in NETS:
        run_once(train_script=train_script, net=net, augment_cfg=cfg_off, args=args)
        run_once(train_script=train_script, net=net, augment_cfg=cfg_on, args=args)

    print('\nAll sweeps completed.')


if __name__ == '__main__':
    main()
