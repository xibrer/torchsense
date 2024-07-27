import torch

def mape_loss(output, target, epsilon=1e-8):
    # 确保输出和目标的形状一致
     # 确保输出和目标的形状一致
    assert output.shape == target.shape, "输出和目标的形状必须一致"
    
    # 计算sMAPE
    numerator = torch.abs(target - output)
    denominator = torch.abs(target) + torch.abs(output) + epsilon
    loss = 2 * numerator / denominator
    
    # 计算均值
    loss = torch.mean(loss)
    
    return loss