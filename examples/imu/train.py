import argparse
import os
import subprocess
import sys
import time
from typing import Any
import torch
import numpy as np
from torch import optim

from torch.nn import HuberLoss
from rich.progress import track

from utils import get_network, MyDataset

workdir = os.path.abspath('.')
sys.setrecursionlimit(2000)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--net",
    default='deeplab6',
    type=str,
    help="网络名：deeplab / deeplab1–6、unet 等",
)
parser.add_argument('-gpu', action='store_true',
                    default=True, help='use gpu or not')
parser.add_argument("--max_epoch", default=50, type=int,
                    help="display a cubic of a given number")

parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--presicion', type=str, default='cuda', help='log tag')

parser.add_argument('--log_root', type=str, default='log')
log_name = str(1)
parser.add_argument('--log_name', type=str, default=log_name + '.txt')
parser.add_argument('--tag', type=str, default='default', help='log tag')
parser.add_argument('--save_name', type=str, default='train', help='checkpoint filename prefix')
parser.add_argument(
    '--augment_cfg',
    type=str,
    default='',
    help='YAML 配置路径；默认 examples/imu/augment.yaml',
)
parser.add_argument(
    '--skip_post_eval',
    action='store_true',
    help='训练后跳过自动 eval/onnx 导出（默认不跳过）',
)
args = parser.parse_args()

use_cuda = args.gpu and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

np.random.seed(42)
torch.manual_seed(42)
if use_cuda:
    torch.cuda.manual_seed_all(42)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

_HUBER_LOSS = HuberLoss()


def regularization_loss(model, lambda_value, p):
    """
       regularization_loss，只惩罚含有 weight 的参数
       model: 传入模型
       factor: 正则化惩罚系数
       p: p-范数
    """
    if lambda_value == 0:
        p0 = next(model.parameters())
        return torch.zeros((), device=p0.device, dtype=p0.dtype)

    p0 = next(model.parameters())
    reg_loss = torch.zeros((), device=p0.device, dtype=p0.dtype)
    for name, w in model.named_parameters():
        if 'weight' in name:  # 只对 参数名 含有 weight 的参数 正则化
            reg_loss = reg_loss + torch.norm(w, p)
    reg_loss = lambda_value * reg_loss
    return reg_loss


def loss_function(recons, gt):
    return _HUBER_LOSS(recons.flatten(), gt.flatten())


def load_augmentation_config(cfg_path: str) -> dict[str, Any]:
    """Load augmentation YAML config.

    Returns empty dict when config cannot be loaded.
    """
    if not os.path.isfile(cfg_path):
        print(f'augmentation config not found, disabled: {cfg_path}')
        return {}

    try:
        import yaml
    except ImportError:
        print('PyYAML not installed, augmentation disabled. install: pip install pyyaml')
        return {}

    try:
        with open(cfg_path, encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as exc:
        print(f'failed to parse augmentation config, disabled: {exc}')
        return {}

    aug = cfg.get('augmentation', {})
    if not isinstance(aug, dict):
        print('invalid augmentation config format, disabled.')
        return {}
    return aug


def build_aug_suffix(aug_cfg: dict[str, Any]) -> str:
    """Build concise suffix from enabled augmentation methods."""
    if not aug_cfg.get('enabled', False):
        return ''

    method_alias = {
        'jitter': 'jit',
        'time_warp': 'tw',
        'random_mask': 'rm',
    }
    enabled_aliases: list[str] = []
    for key, alias in method_alias.items():
        method_cfg = aug_cfg.get(key, {})
        if isinstance(method_cfg, dict) and method_cfg.get('enabled', False):
            enabled_aliases.append(alias)
    if not enabled_aliases:
        return ''
    return '_' + '-'.join(enabled_aliases)


def train(model, save_name):
    min_loss = 10.0
    total_loss = 0.0
    val_loss = 0.0
    nb = device.type == 'cuda'

    log = open(os.path.join(args.log_root, args.log_name), 'a')
    try:
        log.write(save_name + '\n')
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_epoch / 2)
        epoch_start_time = time.time()
        eval_dir = os.path.join(workdir, 'eval')
        ckpt_base = os.path.join(eval_dir, f'{save_name}_{args.net}')

        for epoch in range(args.max_epoch):
            model.train()
            for i, data in enumerate(
                track(
                    train_dataloader,
                    description=f"Epoch {epoch + 1}/{args.max_epoch}",
                ),
                0,
            ):
                inputs_i, labels = data
                optimizer.zero_grad(set_to_none=True)
                inputs_i = inputs_i.to(device, non_blocking=nb)
                labels = labels.to(device, non_blocking=nb)
                lambda_value = 0.00  # L2 正则化系数
                outputs = model(inputs_i)
                if args.net == 'lstm' or args.net == 'gru':
                    loss = loss_function(outputs, labels)
                elif args.net.startswith('deeplab') or args.net in ('unet',):
                    loss = model.loss_function(*outputs, gt=labels)
                else:
                    loss = model.get_reconstruction_loss(*outputs, labels=labels)
                reg_loss = regularization_loss(model, lambda_value, 2)
                total_loss = reg_loss + loss
                total_loss.backward()
                optimizer.step()
            scheduler.step()
            with torch.no_grad():
                model.eval()
                val_loss_sum = 0.0
                val_batches = 0
                _val_loss = HuberLoss()
                for val_data in track(val_dataloader, description="Validation"):
                    inputs_i, labels = val_data
                    inputs_i = inputs_i.to(device, non_blocking=nb)
                    labels = labels.to(device, non_blocking=nb)
                    outputs = model(inputs_i)
                    if args.net == 'lstm' or args.net == 'gru':
                        batch_loss = _val_loss(outputs.flatten(), labels.flatten())
                    else:
                        batch_loss = _val_loss(outputs[0].flatten(), labels.flatten())
                    val_loss_sum += batch_loss.item()
                    val_batches += 1
                val_loss = val_loss_sum / val_batches if val_batches > 0 else 0.0

            train_loss_val = total_loss.detach().item() if torch.is_tensor(total_loss) else float(total_loss)
            info = 'Train: {} batch @ epoch:{}/{} loss: {:.4f} val_loss: {:.4f} time elapsed: {:.1f}h {:.1f}min' \
                .format(i, epoch + 1, args.max_epoch, train_loss_val, val_loss, (time.time() - epoch_start_time) // 3600,
                        (time.time() - epoch_start_time) / 60)
            info = '{}\t'.format(time.asctime(time.localtime())) + info
            print(info)

            log.write(info + '\n')
            log.flush()

            torch.save(model.state_dict(), f'{ckpt_base}_last.pth')

            is_best = val_loss < min_loss
            min_loss = min(val_loss, min_loss)
            if is_best:
                torch.save(model.state_dict(), f'{ckpt_base}_best.pth')
                print('best checkpoint saved:', f'{ckpt_base}_best.pth')

        print('Finished Training')
    finally:
        log.close()


def run_post_train_eval(
    save_name: str,
    train_dir: str,
    val_subjects: list[str],
) -> None:
    """训练完成后自动评估并导出 best/last 两份 ONNX。"""
    imu_dir = os.path.dirname(os.path.abspath(__file__))
    eval_script = os.path.join(imu_dir, 'eval.py')
    out_dir = os.path.join(imu_dir, 'eval_vis')

    for checkpoint in ('best', 'last'):
        onnx_path = os.path.join(out_dir, f'{save_name}_{args.net}_{checkpoint}.onnx')
        cmd = [
            sys.executable,
            eval_script,
            '--net', args.net,
            '--save_name', save_name,
            '--checkpoint', checkpoint,
            '--train_dir', train_dir,
            '--batch_size', str(args.batch_size),
            '--out_dir', out_dir,
            '--onnx_path', onnx_path,
            '--skip_onnx_verify',
        ]
        if val_subjects:
            cmd.extend(['--val_subjects', *val_subjects])

        print(f'Auto-eval ({checkpoint}) and export ONNX...')
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f'自动导出 {checkpoint} ONNX 失败: {exc}')


if __name__ == "__main__":
    train_dir = os.path.join(workdir, 'examples', 'imu', 'dataset', 'ecg_train_set')
    imu_dir = os.path.dirname(os.path.abspath(__file__))
    aug_cfg_path = args.augment_cfg or os.path.join(imu_dir, 'augment.yaml')
    aug_cfg = load_augmentation_config(aug_cfg_path)
    if aug_cfg.get('enabled', False):
        print(f'augmentation enabled from: {aug_cfg_path}')
    else:
        print('augmentation disabled.')
    aug_suffix = build_aug_suffix(aug_cfg)
    effective_save_name = f'{args.save_name}{aug_suffix}'
    print(f'effective save_name: {effective_save_name}')

    val_list = ['wxw_train']
    train_list = [
        # '20-m_train', '21-m_train', '22-f_train', '23-m_train',
        # '27-f_train', '37-f_train', '34-f_train', '35-m-2_train', '36-f_train',
        # '52-f_train', '58-m_train', 'ND_train',
        'dkh_train', 'zch_train', 'yy_train',
        'lsl_train', 'hpy_train', 'ws_train',
        'wxw_train', 'yf_train', 'zzj_train','chy_train','jx_train','lc_train','zyj_train'
    ]

    train_subjects = [s for s in train_list if s not in val_list]
    train_dataset = MyDataset(
        [os.path.join(train_dir, name) for name in train_subjects],
        augment_cfg=aug_cfg,
        is_train=True,
    )
    val_dataset = MyDataset(
        [os.path.join(train_dir, name) for name in val_list],
        augment_cfg=None,
        is_train=False,
    )

    loader_kw = {'num_workers': 4, 'pin_memory': use_cuda}
    if loader_kw['num_workers'] > 0:
        loader_kw['persistent_workers'] = True

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kw,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kw,
    )

    log_dir = os.path.join(args.log_root, args.tag)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(workdir, 'eval'), exist_ok=True)

    args.gpu = use_cuda
    model = get_network(args)
    train(model, effective_save_name)
    if args.skip_post_eval:
        print('skip_post_eval enabled: skip auto eval/onnx export.')
    else:
        run_post_train_eval(save_name=effective_save_name, train_dir=train_dir, val_subjects=val_list)
