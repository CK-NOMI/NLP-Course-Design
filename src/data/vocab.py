"""字级词表构建与管理"""
import json
from collections import Counter
from pathlib import Path
from typing import List, Optional

import pandas as pd

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
PAD_IDX = 0
UNK_IDX = 1


class CharVocab:
    """字级词表：从训练集文本构建 char2id 映射。"""

    def __init__(self):
        self.char2id = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.id2char = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN}

    @classmethod
    def build_from_texts(
        cls,
        texts: List[str],
        min_freq: int = 2,
        max_size: Optional[int] = None,
    ) -> "CharVocab":
        """从文本列表构建词表。"""
        vocab = cls()
        counter = Counter()
        for text in texts:
            counter.update(list(text))

        # 按频率排序
        sorted_chars = sorted(counter.items(), key=lambda x: -x[1])

        for char, freq in sorted_chars:
            if freq < min_freq:
                continue
            if max_size and len(vocab.char2id) >= max_size:
                break
            if char not in vocab.char2id:
                idx = len(vocab.char2id)
                vocab.char2id[char] = idx
                vocab.id2char[idx] = char

        return vocab

    @classmethod
    def build_from_csv(
        cls,
        csv_path: str,
        text_col: str = "text",
        min_freq: int = 2,
        max_size: Optional[int] = None,
    ) -> "CharVocab":
        """从 CSV 文件的 text 列构建词表。"""
        df = pd.read_csv(csv_path)
        texts = df[text_col].dropna().tolist()
        return cls.build_from_texts(texts, min_freq=min_freq, max_size=max_size)

    def text_to_ids(self, text: str, max_len: int) -> List[int]:
        """将文本转为 id 序列，padding/truncation 到 max_len。"""
        ids = [self.char2id.get(ch, UNK_IDX) for ch in text]
        # 截断
        if len(ids) > max_len:
            ids = ids[:max_len]
        # padding
        ids = ids + [PAD_IDX] * (max_len - len(ids))
        return ids

    def save(self, path: str):
        """保存词表为 JSON。"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.char2id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharVocab":
        """从 JSON 加载词表。"""
        vocab = cls()
        with open(path, "r", encoding="utf-8") as f:
            vocab.char2id = json.load(f)
        vocab.id2char = {v: k for k, v in vocab.char2id.items()}
        return vocab

    def __len__(self):
        return len(self.char2id)
