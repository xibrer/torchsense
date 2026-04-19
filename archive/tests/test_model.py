# test_unet.py

import torch
import pytest
from torchsense.models import UNet, resnet18

n_channels = 1


def test_unet_forward():
    # 定义模型参数

    n_classes = 2
    bilinear = True

    batch_size = 4
    seq_length = 128
    # 创建模型实例
    model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
    # 创建一个随机的输入张量，形状为 (batch_size, n_channels, seq_length)
    input_tensor = torch.randn(batch_size, n_channels, seq_length)

    # 运行模型前向传播
    output_tensor = model(input_tensor)

    # 检查输出张量的形状是否正确
    assert output_tensor.shape == (batch_size, n_classes, seq_length), \
        f"Expected output shape (batch_size, n_classes, seq_length), but got {output_tensor.shape}"


def test_resnet_forward():
    batch_size = 4

    seq_length = 128

    # 创建一个随机的输入张量，形状为 (batch_size, n_channels, seq_length)
    input_tensor = torch.randn(batch_size, n_channels, seq_length)
    model = resnet18(128)
    # 运行模型前向传播
    output_tensor = model(input_tensor)

    # 检查输出张量的形状是否正确
    assert output_tensor.shape == (batch_size, n_channels, seq_length), \
        f"Expected output shape (batch_size, n_classes, seq_length), but got {output_tensor.shape}"


# 运行测试
if __name__ == "__main__":
    pytest.main()
