import torch
import torch.nn.functional as F
from .functional import negative_si_snr


class StructuredLoss(torch.nn.Module):
    def __init__(self):
        super(StructuredLoss, self).__init__()

    def forward(self, student_preds, y, teacher_preds, activations, register_layers):
        kd_loss = 0
        for regisiter_layer in register_layers:
            student_key = f"student_{regisiter_layer}"
            teacher_key = f"teacher_{regisiter_layer}"
            # print(student_key, teacher_key)
            # Convert logits to log probabilities for the student
            kd_loss += self.mse_loss(student_key, teacher_key,activations=activations)
        si_snr = negative_si_snr(student_preds, y)
        # student_mse_loss = F.mse_loss(student_preds, y)
        loss_to_teacher = F.mse_loss(student_preds, teacher_preds[0])
        alpha = 0.2  # 权重系数
        total_loss = si_snr + alpha *kd_loss
        return total_loss


    def kl_loss(self,student_key, teacher_key, activations):
        log_probs_student = F.log_softmax(activations[student_key], dim=1)

        # Convert logits to probabilities for the teacher
        probs_teacher = F.softmax(activations[teacher_key], dim=1)

        # Compute the KL divergence loss
        kl_value += F.kl_div(log_probs_student, probs_teacher, reduction='batchmean')

        return kl_value
    
    def mse_loss(self,student_key, teacher_key, activations):
        mse_value = F.mse_loss(activations[student_key], activations[teacher_key])
        return mse_value