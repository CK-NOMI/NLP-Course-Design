"""蒸馏损失函数"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """知识蒸馏损失：CE + KL divergence。

    Loss = alpha * CE(student_logits, labels)
         + (1 - alpha) * T^2 * KL(log_softmax(student/T), softmax(teacher/T))
    """

    def __init__(self, alpha: float = 0.5, temperature: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, labels):
        """
        Args:
            student_logits: (batch, num_classes)
            teacher_logits: (batch, num_classes)
            labels: (batch,) hard labels

        Returns:
            total_loss, ce_loss, kd_loss
        """
        # 硬标签交叉熵
        ce = self.ce_loss(student_logits, labels)

        # 软标签 KL 散度
        T = self.temperature
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)
        kd = self.kl_loss(student_soft, teacher_soft) * (T * T)

        # 总损失
        total = self.alpha * ce + (1 - self.alpha) * kd

        return total, ce, kd
