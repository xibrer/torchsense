"""
在验证集上评估模型、保存波形对比图，并导出 ONNX（DeepLab 等返回 list 的模型会先包一层再导出）。
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import L1Loss, MSELoss

_IMU_DIR = os.path.dirname(os.path.abspath(__file__))
if _IMU_DIR not in sys.path:
    sys.path.insert(0, _IMU_DIR)

from utils import MyDataset, get_network

workdir = os.path.abspath('.')


class OnnxExportWrapper(nn.Module):
    """将 forward 返回 list/tuple 的模型包装为单张量输出，便于 ONNX 导出。"""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.inner(x)
        if isinstance(out, (list, tuple)):
            return out[0]
        return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='IMU ECG 模型验证与 ONNX 导出')
    p.add_argument(
        '--net',
        type=str,
        default='deeplab6',
        help='与 train 一致：deeplab* / unet',
    )
    p.add_argument('--save_name', type=str, default='train', help='与 train 一致的 checkpoint 前缀')
    p.add_argument(
        '--checkpoint',
        type=str,
        choices=('best', 'last'),
        default='best',
        help='加载 eval/{save_name}_{net}_{checkpoint}.pth',
    )
    p.add_argument('--ckpt', type=str, default='', help='若指定则直接使用该权重路径，忽略 checkpoint')
    p.add_argument(
        '--train_dir',
        type=str,
        default='',
        help='ecg_train_set 根目录；默认 examples/imu/dataset/ecg_train_set（相对 workdir）',
    )
    p.add_argument(
        '--val_subjects',
        nargs='+',
        default=['wxw_train'],
        help='验证集子目录名，与 train 中 val_list 一致',
    )
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--gpu', action='store_true', default=True)
    p.add_argument('--no_gpu', action='store_true', help='强制 CPU')
    p.add_argument('--out_dir', type=str, default='', help='可视化输出目录，默认 examples/imu/eval_vis')
    p.add_argument('--num_plots', type=int, default=6, help='保存多少条样本的波形对比图')
    p.add_argument('--onnx_path', type=str, default='', help='ONNX 保存路径；默认 out_dir/model.onnx')
    p.add_argument('--skip_onnx', action='store_true', help='不导出 ONNX')
    p.add_argument(
        '--skip_onnx_verify',
        action='store_true',
        help='导出 ONNX 后不进行 onnxruntime 随机数据推理自检',
    )
    p.add_argument('--opset', type=int, default=17)
    return p.parse_args()


def _ensure_channel(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x.unsqueeze(1)
    if x.dim() == 1:
        return x.unsqueeze(0).unsqueeze(0)
    return x


def verify_onnx_random(
    onnx_path: str,
    torch_cpu: nn.Module,
    seq_len: int,
) -> None:
    """onnx 结构检查 + 随机张量经 onnxruntime 推理，并与 PyTorch 对比。"""
    import onnx

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print('onnx.check_model: OK')

    try:
        import onnxruntime as ort
    except ImportError:
        print('未安装 onnxruntime，跳过随机推理自检。安装: pip install onnxruntime')
        return

    torch_cpu.eval()
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    rng = np.random.default_rng(42)
    cases = [
        ('same_len', 2, seq_len),
        ('other_len', 1, max(32, seq_len // 2)),
    ]
    for tag, batch, length in cases:
        x = rng.standard_normal((batch, 1, length)).astype(np.float32)
        xt = torch.from_numpy(x)
        with torch.no_grad():
            pt = torch_cpu(xt).numpy()
        ort_out = sess.run([out_name], {inp_name: x})[0]
        if ort_out.shape != pt.shape:
            raise RuntimeError(f'{tag}: shape mismatch pt={pt.shape} onnx={ort_out.shape}')
        max_diff = float(np.abs(pt - ort_out).max())
        ok = np.allclose(pt, ort_out, rtol=1e-3, atol=1e-4)
        print(
            f'ONNXRuntime [{tag}] batch={batch} len={length} out={ort_out.shape} '
            f'max_abs_diff={max_diff:.6f} allclose={ok}',
        )


def main() -> None:
    args = parse_args()
    use_cuda = args.gpu and not args.no_gpu and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    train_dir = args.train_dir or os.path.join(_IMU_DIR, 'dataset', 'ecg_train_set')
    out_dir = args.out_dir or os.path.join(_IMU_DIR, 'eval_vis')
    os.makedirs(out_dir, exist_ok=True)

    val_paths = [os.path.join(train_dir, name) for name in args.val_subjects]
    val_dataset = MyDataset(val_paths)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
    )

    ckpt_name = f'{args.save_name}_{args.net}_{args.checkpoint}.pth'
    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        ckpt_path = os.path.join(workdir, 'eval', ckpt_name)
        if not os.path.isfile(ckpt_path):
            alt = os.path.join(_IMU_DIR, 'eval', ckpt_name)
            if os.path.isfile(alt):
                ckpt_path = alt
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'未找到权重: {ckpt_path}（可用 --ckpt 指定）')

    class _A:
        net = args.net
        gpu = use_cuda

    model = get_network(_A)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    l1 = L1Loss(reduction='sum')
    mse = MSELoss(reduction='sum')
    n_elems = 0
    l1_sum = 0.0
    mse_sum = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = _ensure_channel(inputs.to(device, non_blocking=True))
            labels = _ensure_channel(labels.to(device, non_blocking=True))
            outputs = model(inputs)
            pred = outputs if not isinstance(outputs, (list, tuple)) else outputs[0]
            b_elems = pred.numel()
            n_elems += b_elems
            l1_sum += l1(pred, labels).item()
            mse_sum += mse(pred, labels).item()

    mae = l1_sum / max(n_elems, 1)
    rmse = (mse_sum / max(n_elems, 1)) ** 0.5
    print(f'checkpoint: {ckpt_path}')
    print(f'val samples: {len(val_dataset)}  |  MAE: {mae:.6f}  RMSE: {rmse:.6f}')

    n_plot = min(args.num_plots, len(val_dataset))
    fig, axes = plt.subplots(n_plot, 1, figsize=(12, 2.2 * n_plot), squeeze=False)
    with torch.no_grad():
        for k in range(n_plot):
            inp, lab = val_dataset[k]
            inp_b = _ensure_channel(inp.unsqueeze(0)).to(device)
            out = model(inp_b)
            pred = out[0] if isinstance(out, (list, tuple)) else out
            pred_np = pred[0, 0].detach().cpu().numpy().ravel()
            lab_np = np.asarray(lab.detach().cpu()).ravel()
            inp_np = np.asarray(inp.detach().cpu()).ravel()
            n = min(len(pred_np), len(lab_np), len(inp_np))
            pred_np, lab_np, inp_np = pred_np[:n], lab_np[:n], inp_np[:n]
            ax = axes[k][0]
            t = np.arange(n)
            ax.plot(t, inp_np, 'b', alpha=0.5, linewidth=0.8, label='SCG in')
            ax.plot(t, lab_np, 'g', linewidth=1.0, label='ECG gt')
            ax.plot(t, pred_np, 'r--', linewidth=1.0, label='ECG pred')
            ax.set_title(f'sample {k}')
            ax.legend(loc='upper right', fontsize=8)
            ax.set_xlabel('t')
    plt.tight_layout()
    vis_path = os.path.join(out_dir, f'{args.save_name}_{args.net}_val_compare.png')
    fig.savefig(vis_path, dpi=150)
    plt.close(fig)
    print(f'saved figure: {vis_path}')

    if args.skip_onnx:
        return

    onnx_out = args.onnx_path or os.path.join(out_dir, f'{args.save_name}_{args.net}.onnx')
    try:
        import onnx  # noqa: F401
    except ImportError:
        print('未安装 onnx，跳过导出。可执行: uv pip install onnx onnxscript')
        return

    wrapper = OnnxExportWrapper(model).cpu().eval()
    sample_inp, _ = val_dataset[0]
    dummy = _ensure_channel(sample_inp.unsqueeze(0)).float().cpu()
    seq_len = dummy.shape[-1]

    export_kw = dict(
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch', 2: 'length'},
            'output': {0: 'batch', 2: 'length'},
        },
        opset_version=args.opset,
    )
    try:
        torch.onnx.export(wrapper, dummy, onnx_out, dynamo=False, **export_kw)
    except TypeError:
        torch.onnx.export(wrapper, dummy, onnx_out, **export_kw)
    print(f'saved onnx: {onnx_out}  (dummy length={seq_len}, dynamic batch/length)')

    if not args.skip_onnx_verify:
        verify_onnx_random(onnx_out, wrapper, seq_len)


if __name__ == '__main__':
    main()
