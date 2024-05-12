from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
import numpy as np
from h5py import File


def load_file(file_path, keys):
    file_extension = file_path.split('.')[-1].lower()

    if file_extension == '.txt':
        return load_text_file(file_path)
    elif file_extension == 'csv':
        return load_csv_file(file_path)
    elif file_extension == 'json':
        return load_json_file(file_path)
    elif file_extension == 'mat':
        return load_mat_file(file_path, keys)
    elif file_extension == 'npz':
        return load_npz_file(file_path, keys)
    else:
        raise ValueError("Unsupported file format")


def load_text_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()


def load_csv_file(file_path):
    import csv
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)
    return data


def load_json_file(file_path):
    import json
    with open(file_path, 'r') as file:
        return json.load(file)


def load_mat_file(path, list_j):
    from scipy import io
    try:
        all_mat_data = io.loadmat(path)
        return get_meta_data(all_mat_data, list_j)

    except NotImplementedError:
        all_mat_data = File(path)
        return get_meta_data(all_mat_data, list_j)


def get_meta_data(raw_data: Union[Dict[str, Any], File], keys: List[str] = None) -> Tuple:
    if keys is None:
        # If list_j is empty, return all values in data as a tuple
        # Remove meta info
        if '__header__' in raw_data:
            del raw_data['__header__']
        if '__version__' in raw_data:
            del raw_data['__version__']
        if '__globals__' in raw_data:
            del raw_data['__globals__']
        return tuple(raw_data.values())
    else:
        # 初始化一个空列表
        values = []
        for key in keys:
            # 检查当前的键是否在 raw_data 字典中
            if key in raw_data:
                value = raw_data[key]
                values.append(value)
        tuple_values = tuple(values)
        # 将列表转换为元组
        return tuple_values


def load_npz_file(path, keys):
    npz_data = np.load(path)
    return tuple(npz_data[key] for key in keys if key in npz_data)

# 使用示例
# file_path = 'example.txt'
# data = load_file(file_path)
# print(data)
