import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsense.utils import get_audio_timestamps


class PiecewiseLoss(torch.nn.Module):
    def __init__(self):
        super(PiecewiseLoss, self).__init__()
        self.mae_loss = nn.L1Loss()

    def forward(self, out_images, target_images, mic):

        # 处理输入数据
        mic = mic.squeeze(1)
        out_images = out_images.squeeze(1)
        target_images = target_images.squeeze(1)
        batch_size = mic.shape[0]

        speech_timestamps = []
        for i in range(batch_size):
            speech_timestamp = get_audio_timestamps(mic[i])
            speech_timestamps.append(speech_timestamp)

        # 初始化权重矩阵
        gt_weights = torch.full_like(out_images, 0.001)  # 先全部填充为0.1
        for i, timestamps in enumerate(speech_timestamps):
            for ts in timestamps:
                start = int(ts[0] * 392)
                end = int(ts[1] * 392)
                gt_weights[i, start:end] = 0.999  # 设置语音段为0.9

        # 计算加权的图像损失
        # sample_loss = F.l1_loss(out_images, target_images, reduction='none')

        # sample_loss = sample_loss * gt_weights
        # part_image_loss = sample_loss.mean()

        # all_image_loss = F.l1_loss(out_images, target_images)
        # 最终损失
        # loss = part_image_loss +all_image_loss

        loss = F.l1_loss(out_images, target_images * gt_weights)

        return loss
