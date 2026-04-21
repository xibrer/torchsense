"""批量将 eval 目录下的 .pth 权重转为 ONNX（与 train 后 run_post_train_eval 命名一致）。"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from types import SimpleNamespace
import torch
import torch.nn as nn

_IMU_DIR = os.path.dirname(os.path.abspath(__file__))
if _IMU_DIR not in sys.path:
    sys.path.insert(0, _IMU_DIR)

from utils import MyDataset, get_network

# 与 utils.get_network 中 IMU 常用名一致；按长度降序匹配，避免 deeplab 误匹配 deeplab6
_KNOWN_NETS: tuple[str, ...] = tuple(
    sorted(
        (
            'deeplab6',
            'deeplab5',
            'deeplab4',
            'deeplab3',
            'deeplab2',
            'deeplab1',
            'deeplab',
            'unet',
            'lstm',
            'gru',
        ),
        key=len,
        reverse=True,
    ),
)


class OnnxExportWrapper(nn.Module):
    """与 eval.py 一致：list/tuple 输出取首元素。"""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.inner(x)
        if isinstance(out, (list, tuple)):
            return out[0]
        return out


def _ensure_channel(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x.unsqueeze(1)
    if x.dim() == 1:
        return x.unsqueeze(0).unsqueeze(0)
    return x


def parse_ckpt_stem(stem: str) -> tuple[str, str, str] | None:
    """从文件名 stem（无 .pth）解析 save_name、net、checkpoint。

    期望格式：{save_name}_{net}_{best|last}
    """
    for suffix in ('_best', '_last'):
        if stem.endswith(suffix):
            rest = stem[: -len(suffix)]
            break
    else:
        return None

    for net in _KNOWN_NETS:
        suf = f'_{net}'
        if rest.endswith(suf):
            save_name = rest[: -len(suf)]
            if save_name:
                return save_name, net, suffix.lstrip('_')
    return None


def export_one(
    pth_path: str,
    out_dir: str,
    train_dir: str,
    val_subjects: list[str],
    opset: int,
    skip_verify: bool,
    net_override: str,
) -> bool:
    base = os.path.basename(pth_path)
    stem, _ext = os.path.splitext(base)
    parsed = parse_ckpt_stem(stem)
    if parsed is None:
        print(f'[skip] 无法解析文件名: {base}（期望 *_{net}_best.pth 或 *_last.pth）')
        return False
    save_name, net, ckpt_tag = parsed
    if net_override:
        net = net_override

    # 不可在 class 体内写 net=net：类作用域会把 net 当局部名，右侧 net 未定义。
    args_net = SimpleNamespace(net=net, gpu=False)

    try:
        model = get_network(args_net)
    except SystemExit:
        print(f'[skip] 不支持的 net={net}: {pth_path}')
        return False

    device = torch.device('cpu')
    state = torch.load(pth_path, map_location=device)
    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        print(f'[skip] load_state_dict 失败 {base}: {exc}')
        return False

    model.eval()
    val_paths = [os.path.join(train_dir, name) for name in val_subjects]
    val_dataset = MyDataset(val_paths)
    sample_inp, _ = val_dataset[0]
    dummy = _ensure_channel(sample_inp.unsqueeze(0)).float().cpu()
    seq_len = dummy.shape[-1]

    onnx_name = f'{save_name}_{net}_{ckpt_tag}.onnx'
    onnx_out = os.path.join(out_dir, onnx_name)

    try:
        import onnx  # noqa: F401
    except ImportError:
        print('未安装 onnx，跳过。可执行: uv pip install onnx onnxscript')
        return False

    wrapper = OnnxExportWrapper(model).cpu().eval()
    export_kw = dict(
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch', 2: 'length'},
            'output': {0: 'batch', 2: 'length'},
        },
        opset_version=opset,
    )
    try:
        torch.onnx.export(wrapper, dummy, onnx_out, dynamo=False, **export_kw)
    except TypeError:
        torch.onnx.export(wrapper, dummy, onnx_out, **export_kw)
    print(f'[ok] {base} -> {onnx_out}  (dummy_len={seq_len})')

    if not skip_verify:
        try:
            from eval import verify_onnx_random
        except ImportError:
            pass
        else:
            verify_onnx_random(onnx_out, wrapper, seq_len)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description='批量 pth -> ONNX')
    p.add_argument(
        '--input',
        type=str,
        default='',
        help='单个 .pth 路径，或与 --glob 二选一',
    )
    p.add_argument(
        '--glob',
        type=str,
        default='',
        help='glob 模式，如 eval/*_best.pth',
    )
    p.add_argument(
        '--input_dir',
        type=str,
        default='',
        help='目录内所有 .pth；默认仓库根下 eval/',
    )
    p.add_argument(
        '--out_dir',
        type=str,
        default='',
        help='ONNX 输出目录；默认 examples/imu/eval_vis',
    )
    p.add_argument(
        '--train_dir',
        type=str,
        default='',
        help='用于取一条样本作 dummy；默认 examples/imu/dataset/ecg_train_set',
    )
    p.add_argument(
        '--val_subjects',
        nargs='+',
        default=['wxw_train'],
        help='与 eval 一致',
    )
    p.add_argument('--opset', type=int, default=17)
    p.add_argument('--skip_verify', action='store_true', help='跳过 onnxruntime 随机自检')
    p.add_argument('--net', type=str, default='', help='强制指定 net（解析失败或需覆盖时）')
    args = p.parse_args()

    repo_root = os.path.abspath(os.path.join(_IMU_DIR, '..', '..'))
    default_eval = os.path.join(repo_root, 'eval')
    out_dir = args.out_dir or os.path.join(_IMU_DIR, 'eval_vis')
    train_dir = args.train_dir or os.path.join(_IMU_DIR, 'dataset', 'ecg_train_set')
    os.makedirs(out_dir, exist_ok=True)

    paths: list[str] = []
    if args.input:
        paths = [args.input]
    elif args.glob:
        paths = sorted(glob.glob(args.glob, recursive=False))
    elif args.input_dir:
        for name in sorted(os.listdir(args.input_dir)):
            if name.lower().endswith('.pth'):
                paths.append(os.path.join(args.input_dir, name))
    else:
        if os.path.isdir(default_eval):
            for name in sorted(os.listdir(default_eval)):
                if name.lower().endswith('.pth'):
                    paths.append(os.path.join(default_eval, name))
        else:
            print(f'默认 eval 目录不存在: {default_eval}，请指定 --input_dir 或 --glob')
            sys.exit(1)

    if not paths:
        print('未找到任何 .pth 文件')
        sys.exit(1)

    ok = 0
    for pth in paths:
        if not os.path.isfile(pth):
            print(f'[skip] 不是文件: {pth}')
            continue
        if export_one(
            pth_path=pth,
            out_dir=out_dir,
            train_dir=train_dir,
            val_subjects=args.val_subjects,
            opset=args.opset,
            skip_verify=args.skip_verify,
            net_override=args.net,
        ):
            ok += 1
    print(f'完成: {ok}/{len(paths)} 成功')


if __name__ == '__main__':
    main()
