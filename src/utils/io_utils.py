"""文件读写工具"""
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd


def ensure_dir(path: str):
    """确保目录存在，不存在则创建。"""
    Path(path).mkdir(parents=True, exist_ok=True)


def read_csv(path: str) -> pd.DataFrame:
    """读取 CSV 文件。"""
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: str, index: bool = False):
    """保存 DataFrame 为 CSV。"""
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=index)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """逐行读取 JSONL 文件。"""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(items: List[Dict[str, Any]], path: str):
    """逐行保存为 JSONL 文件。"""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
