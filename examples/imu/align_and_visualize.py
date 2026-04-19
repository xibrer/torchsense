"""Read paired IMU/ECG CSV samples, align by timestamp, and plot.

This script loads ``*_imu.csv`` and ``*_ecg.csv`` from ``examples/imu``, finds their
time overlap, collapses duplicate timestamps, interpolates both streams onto a
common time grid, and visualizes ECG (microvolts) with IMU ``acc_y`` only on one
figure (dual *y* axes for different units). By default writes
``align_preview.png`` next to this script so the file updates on each run; use
``--no-save`` to skip.

Example:
    python examples/imu/align_and_visualize.py
    python examples/imu/align_and_visualize.py --target-hz 500 --show
    python examples/imu/align_and_visualize.py --no-save --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def load_session_csv(path: Path) -> pd.DataFrame:
    """Load a session CSV and sort by ``timestamp_ms``.

    Args:
        path (Path): CSV path with a ``timestamp_ms`` column.

    Returns:
        pd.DataFrame: Sorted rows.
    """
    df = pd.read_csv(path)
    if "timestamp_ms" not in df.columns:
        msg = f"{path}: missing 'timestamp_ms' column"
        raise ValueError(msg)
    return df.sort_values("timestamp_ms").reset_index(drop=True)


def collapse_duplicate_timestamps(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Average samples that share the same ``timestamp_ms``.

    Args:
        df (pd.DataFrame): Input with ``timestamp_ms`` and numeric ``value_cols``.
        value_cols (list[str]): Columns to aggregate with mean.

    Returns:
        pd.DataFrame: One row per unique timestamp.
    """
    return df.groupby("timestamp_ms", as_index=False)[value_cols].mean()


def overlap_bounds(
    t_a: np.ndarray,
    t_b: np.ndarray,
) -> tuple[int, int]:
    """Return inclusive overlap ``[start_ms, end_ms]`` for two timestamp arrays.

    Args:
        t_a (np.ndarray): Sorted timestamps (ms).
        t_b (np.ndarray): Sorted timestamps (ms).

    Returns:
        tuple[int, int]: Overlap start and end in ms.

    Raises:
        ValueError: If there is no overlap.
    """
    start = int(max(t_a.min(), t_b.min()))
    end = int(min(t_a.max(), t_b.max()))
    if start > end:
        msg = "No time overlap between the two series."
        raise ValueError(msg)
    return start, end


def interpolate_linear(t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Linearly interpolate ``y_src`` defined at ``t_src`` onto ``t_dst``.

    Args:
        t_src (np.ndarray): Source timestamps (strictly increasing after dedup).
        y_src (np.ndarray): Source values, same length as ``t_src``.
        t_dst (np.ndarray): Destination timestamps inside ``[t_src[0], t_src[-1]]``.

    Returns:
        np.ndarray: Interpolated values at ``t_dst``.
    """
    return np.interp(t_dst.astype(np.float64), t_src.astype(np.float64), y_src.astype(np.float64))


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Write ``fig`` to ``path``, using a temp file then replace so viewers see a new file.

    Args:
        fig (plt.Figure): Matplotlib figure.
        path (Path): Destination PNG path.
    """
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.write{path.suffix}")
    try:
        fig.savefig(tmp, dpi=150)
        tmp.replace(path)
    except OSError:
        tmp.unlink(missing_ok=True)
        fig.savefig(path, dpi=150)


def build_uniform_grid(t_start_ms: int, t_end_ms: int, target_hz: float) -> np.ndarray:
    """Build a uniform time grid in milliseconds over ``[t_start_ms, t_end_ms]``.

    Args:
        t_start_ms (int): Interval start (ms).
        t_end_ms (int): Interval end (ms).
        target_hz (float): Samples per second.

    Returns:
        np.ndarray: Timestamps in ms, shape ``(N,)``.
    """
    duration_s = (t_end_ms - t_start_ms) / 1000.0
    n = max(2, int(np.ceil(duration_s * target_hz)) + 1)
    return np.linspace(t_start_ms, t_end_ms, n, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Align IMU/ECG CSVs by timestamp and plot.")
    parser.add_argument(
        "--imu",
        type=Path,
        default=_script_dir() / "session_20260419_134121_imu.csv",
        help="Path to IMU CSV.",
    )
    parser.add_argument(
        "--ecg",
        type=Path,
        default=_script_dir() / "session_20260419_134121_ecg.csv",
        help="Path to ECG CSV.",
    )
    parser.add_argument(
        "--target-hz",
        type=float,
        default=400.0,
        help="Common sampling rate (Hz) for interpolation grid. Defaults to 400.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Call plt.show() to display the figure interactively.",
    )
    out_group = parser.add_mutually_exclusive_group()
    out_group.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write a PNG (only useful with --show).",
    )
    out_group.add_argument(
        "--save",
        type=Path,
        default=None,
        metavar="PATH",
        help="PNG output path. Default: align_preview.png next to this script.",
    )
    args = parser.parse_args()

    imu_raw = load_session_csv(args.imu)
    ecg_raw = load_session_csv(args.ecg)

    if "acc_y" not in imu_raw.columns:
        msg = f"{args.imu}: need acc_y column"
        raise ValueError(msg)
    if "sample_uv" not in ecg_raw.columns:
        msg = f"{args.ecg}: need sample_uv column"
        raise ValueError(msg)

    imu_dedup = collapse_duplicate_timestamps(imu_raw, ["acc_y"])
    ecg_dedup = collapse_duplicate_timestamps(ecg_raw, ["sample_uv"])

    t_imu = imu_dedup["timestamp_ms"].to_numpy()
    t_ecg = ecg_dedup["timestamp_ms"].to_numpy()

    t0, t1 = overlap_bounds(t_imu, t_ecg)
    grid_t = build_uniform_grid(t0, t1, args.target_hz)

    time_s = (grid_t - t0) / 1000.0

    ecg_y = interpolate_linear(t_ecg, ecg_dedup["sample_uv"].to_numpy(), grid_t)
    acc_y = interpolate_linear(t_imu, imu_dedup["acc_y"].to_numpy(), grid_t)

    fig, ax_ecg = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ax_acc = ax_ecg.twinx()

    line_ecg, = ax_ecg.plot(time_s, ecg_y, color="C0", lw=0.8, label="ECG (µV)")
    line_acc, = ax_acc.plot(time_s, acc_y, color="C1", lw=0.8, label="acc_y (m/s²)")

    ax_ecg.set_xlabel("Time from overlap start (s)")
    ax_ecg.set_ylabel("ECG (µV)", color="C0")
    ax_acc.set_ylabel("acc_y (m/s²)", color="C1")
    ax_ecg.tick_params(axis="y", labelcolor="C0")
    ax_acc.tick_params(axis="y", labelcolor="C1")
    ax_ecg.set_title("Aligned session: ECG + IMU acc_y (linear interp on overlap)")
    ax_ecg.grid(True, alpha=0.3)

    ax_ecg.legend(handles=[line_ecg, line_acc], loc="upper right", fontsize=8)

    if args.no_save:
        save_path: Path | None = None
    else:
        save_path = args.save if args.save is not None else _script_dir() / "align_preview.png"
    if save_path is not None:
        save_figure(fig, save_path)
        print(f"Saved figure to {save_path}", flush=True)
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
