"""数据集模块：字级输入 Dataset（供 TextCNN / BiLSTM 使用）"""
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.vocab import CharVocab


class CharDataset(Dataset):
    """字级分类数据集。

    读取 CSV (id, text, label)，输出 input_ids / label / sample_id / text。
    """

    def __init__(self, csv_file: str, vocab: CharVocab, max_len: int = 256, max_samples: int = None):
        self.df = pd.read_csv(csv_file)
        if max_samples is not None:
            self.df = self.df.head(max_samples).reset_index(drop=True)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["text"]) if pd.notna(row["text"]) else ""
        label = int(row["label"])
        sample_id = row["id"]  # 支持 int 和字符串 id

        input_ids = self.vocab.text_to_ids(text, self.max_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": sample_id,
            "text": text,
        }


class BertDataset(Dataset):
    """BERT/MacBERT 分类数据集。

    读取 CSV (id, text, label)，使用 HuggingFace tokenizer 编码。
    """

    def __init__(self, csv_file: str, tokenizer, max_len: int = 128, max_samples: int = None):
        self.df = pd.read_csv(csv_file)
        if max_samples is not None:
            self.df = self.df.head(max_samples).reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["text"]) if pd.notna(row["text"]) else ""
        label = int(row["label"])
        sample_id = row["id"]  # 支持 int 和字符串 id

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": sample_id,
            "text": text,
        }
