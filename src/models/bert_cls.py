"""BERT/MacBERT 分类模型"""
import torch.nn as nn
from transformers import AutoModel


class BertClassifier(nn.Module):
    """基于预训练模型的文本分类器。

    AutoModel + [CLS] hidden state + Dropout + Linear
    适用于 BERT、MacBERT 等 HuggingFace 预训练模型。
    """

    def __init__(
        self,
        pretrained_model_name: str,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len), optional
        Returns:
            logits: (batch, num_classes)
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # 取 [CLS] token 的隐状态
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits
