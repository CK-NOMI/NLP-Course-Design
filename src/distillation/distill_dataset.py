"""蒸馏训练数据集：合并训练文本和教师 logits"""
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.vocab import CharVocab


class DistillCharDataset(Dataset):
    """字级蒸馏数据集（供 TextCNN / BiLSTM 学生模型使用）。

    合并 train CSV 和 teacher logits CSV，按 id 对齐。
    """

    def __init__(
        self,
        train_file: str,
        teacher_logits_file: str,
        vocab: CharVocab,
        max_len: int = 256,
        max_samples: int = None,
    ):
        train_df = pd.read_csv(train_file)
        logits_df = pd.read_csv(teacher_logits_file)

        if max_samples is not None:
            train_df = train_df.head(max_samples).reset_index(drop=True)

        # 按 id 合并
        # 确保 id 类型一致
        train_df["id"] = train_df["id"].astype(str)
        logits_df["id"] = logits_df["id"].astype(str)

        merged = train_df.merge(logits_df[["id", "logit_0", "logit_1"]], on="id", how="inner")

        if len(merged) == 0:
            raise ValueError(
                f"合并后样本数为 0。请检查 train_file 和 teacher_logits_file 的 id 是否匹配。"
                f"\ntrain_file ids 示例: {train_df['id'].head(5).tolist()}"
                f"\nlogits_file ids 示例: {logits_df['id'].head(5).tolist()}"
            )

        if len(merged) < len(train_df):
            missing = len(train_df) - len(merged)
            print(f"[警告] {missing} 条训练样本缺少教师 logits，已跳过。")

        self.df = merged.reset_index(drop=True)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["text"]) if pd.notna(row["text"]) else ""
        label = int(row["label"])
        sample_id = str(row["id"])

        input_ids = self.vocab.text_to_ids(text, self.max_len)

        teacher_logits = torch.tensor(
            [float(row["logit_0"]), float(row["logit_1"])],
            dtype=torch.float,
        )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "teacher_logits": teacher_logits,
            "sample_id": sample_id,
            "text": text,
        }
