from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
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


def get_meta_data(
        raw_data: Union[Dict[str, Any], File],
        keys: Tuple[List[str], Optional[List[str]]]
) -> Tuple:
    def parse_key(key: str) -> Tuple[str, Optional[int]]:
        if '[' in key and ']' in key:
            key_name = key[:key.index('[')]
            row_str = key[key.index('[') + 1:key.index(']')]
            if row_str.isdigit():
                return key_name, int(row_str)
        return key, None

    if keys is None:
        # If keys is None, return all values in raw_data as a tuple
        # Remove meta info
        if '__header__' in raw_data:
            del raw_data['__header__']
        if '__version__' in raw_data:
            del raw_data['__version__']
        if '__globals__' in raw_data:
            del raw_data['__globals__']
        return tuple(raw_data.values())
    else:
        inputs, targets = keys
        input_values = []
        target_values = []

        # 提取inputs并处理特定行
        for key in inputs:
            key_name, row = parse_key(key)
            if key_name in raw_data:
                value = raw_data[key_name]
                if row is not None and len(value) > row:
                    input_values.append(value[row])
                else:
                    input_values.append(value)

        # 提取targets
        for key in targets:
            key_name, row = parse_key(key)
            if key_name in raw_data:
                value = raw_data[key_name]
                if row is not None and len(value) > row:
                    target_values.append(value[row])
                else:
                    target_values.append(value)

        # 将列表转换为元组
        return input_values, target_values


def load_npz_file(path, keys):
    raw_data = np.load(path)
    def parse_key(key: str) -> Tuple[str, Optional[int]]:
        if '[' in key and ']' in key:
            key_name = key[:key.index('[')]
            row_str = key[key.index('[') + 1:key.index(']')]
            if row_str.isdigit():
                return key_name, int(row_str)
        return key, None

    if keys is None:
        # If keys is None, return all values in raw_data as a tuple
        # Remove meta info
        if '__header__' in raw_data:
            del raw_data['__header__']
        if '__version__' in raw_data:
            del raw_data['__version__']
        if '__globals__' in raw_data:
            del raw_data['__globals__']
        return tuple(raw_data.values())
    else:
        inputs, targets = keys
        input_values = []
        target_values = []

        # 提取inputs并处理特定行
        for key in inputs:
            key_name, row = parse_key(key)
            if key_name in raw_data:
                value = raw_data[key_name]
                if row is not None and len(value) > row:
                    input_values.append(value[row])
                else:
                    input_values.append(value)

        # 提取targets
        for key in targets:
            key_name, row = parse_key(key)
            if key_name in raw_data:
                value = raw_data[key_name]
                if row is not None and len(value) > row:
                    target_values.append(value[row])
                else:
                    target_values.append(value)

        # 将列表转换为元组
        return input_values, target_values


