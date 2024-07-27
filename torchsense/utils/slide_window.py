import os
import numpy as np
from hdf5storage import loadmat
from pathlib import Path


def slide_window(save_name, data, window_size, step, vars_to_slide=None):
    if vars_to_slide is None:
        vars_to_slide = data.keys()

    # 找到最长变量的长度
    max_length = max(data[var].shape[-1] for var in vars_to_slide)

    for i in range(0, max_length - window_size + 1, step):
        save_data = {}
        for var in data:
            if var in vars_to_slide:
                current_length = data[var].shape[-1]
                factor = max_length / current_length
                adjusted_window_size = int(window_size / factor)
                adjusted_step = int(step / factor)

                # 避免adjusted_window_size和adjusted_step过小
                if adjusted_window_size < 1 or adjusted_step < 1:
                    continue

                start_idx = int(i / factor)
                end_idx = start_idx + adjusted_window_size

                # 确保索引不超出范围
                if end_idx > current_length:
                    end_idx = current_length
                save_data[var] = data[var][..., start_idx:end_idx]
            else:
                save_data[var] = data[var]

        # 创建新的文件名，由原始文件名（去掉扩展名）和当前窗口的起始索引组成
        file_name = f"{save_name.split('.')[0]}_{i}.npz"
        # 保存为.npz文件
        np.savez(file_name, **save_data)
        # print(f"{file_name} saved.")


