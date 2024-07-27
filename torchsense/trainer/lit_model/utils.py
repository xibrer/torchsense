import torch
from torchsense.metrics import *
from torchsense.metrics.kd_losses import StructuredLoss
from torchsense.metrics.triplet import TripletLoss

def get_loss_fn(loss):
    if loss is not None:
        loss = loss.lower()
    match loss:
        case "mse":
            loss_fn = torch.nn.MSELoss()
        case "kd_loss":
            loss_fn = StructuredLoss()
        case "cross_entropy":
            loss_fn = torch.nn.CrossEntropyLoss()
        case "l1" | "mae":
            loss_fn = torch.nn.L1Loss()
        case "sisnr":
            loss_fn = ScaleInvariantSignalNoiseRatio()
        case "huber":
            loss_fn = torch.nn.SmoothL1Loss()
        case "bce":
            loss_fn = torch.nn.BCELoss()
        case "bce_with_logits":
            loss_fn = torch.nn.BCEWithLogitsLoss()
        case "triplet":
            loss_fn = TripletLoss()
        case "clt":
            loss_fn = TripletLoss()
        case "hinge":
            loss_fn = torch.nn.HingeEmbeddingLoss()
        case None:
            loss_fn = None  # 当未提供损失函数时处理此情况
        case _:
            raise ValueError(f"Unknown loss function: {loss}")
    return loss_fn
