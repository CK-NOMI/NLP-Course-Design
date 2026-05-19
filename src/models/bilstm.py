"""BiLSTM 模型"""
import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    """双向 LSTM 文本分类模型。

    Embedding → BiLSTM → 取最后时间步前向+后向隐状态拼接 → Dropout → FC
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2,
        padding_idx: int = 0,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len)
        Returns:
            logits: (batch, num_classes)
        """
        # (batch, seq_len, embed_dim)
        x = self.embedding(input_ids)

        # output: (batch, seq_len, hidden_dim * num_directions)
        # h_n: (num_layers * num_directions, batch, hidden_dim)
        _, (h_n, _) = self.lstm(x)

        # 取最后一层的前向和后向隐状态拼接
        if self.bidirectional:
            # h_n[-2]: 最后一层前向, h_n[-1]: 最后一层后向
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            hidden = h_n[-1]

        # (batch, hidden_dim * num_directions)
        out = self.dropout(hidden)
        logits = self.fc(out)
        return logits
