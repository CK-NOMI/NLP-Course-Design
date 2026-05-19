"""TextCNN 模型"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """多核卷积文本分类模型。

    Embedding → 多尺度 Conv1d → ReLU → MaxPool → Concat → Dropout → FC
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_filters: int = 128,
        kernel_sizes: List[int] = None,
        dropout: float = 0.3,
        num_classes: int = 2,
        padding_idx: int = 0,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [2, 3, 4, 5]

        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k)
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len)
        Returns:
            logits: (batch, num_classes)
        """
        # (batch, seq_len, embed_dim)
        x = self.embedding(input_ids)
        # (batch, embed_dim, seq_len) for Conv1d
        x = x.transpose(1, 2)

        conv_outs = []
        for conv in self.convs:
            # (batch, num_filters, L')
            c = F.relu(conv(x))
            # (batch, num_filters)
            c = F.max_pool1d(c, c.size(2)).squeeze(2)
            conv_outs.append(c)

        # (batch, num_filters * len(kernel_sizes))
        out = torch.cat(conv_outs, dim=1)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits
