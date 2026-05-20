#!/usr/bin/env python
"""数据预处理：清洗 + 统一格式输出到 data/processed/"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.utils.io_utils import read_csv, save_csv, ensure_dir
from src.utils.logger import get_logger

logger = get_logger("preprocess")


def clean_text(text) -> str:
    """轻量清洗：去除首尾空白和多余空格，保留标点和口语表达。"""
    if not isinstance(text, str):
        return ""
    # 去除首尾空白
    text = text.strip()
    # 将连续空白压缩为单个空格
    text = " ".join(text.split())
    return text


def preprocess_split(input_path: str, output_path: str, split_name: str):
    """对单个划分进行清洗。"""
    df = read_csv(input_path)
    n_before = len(df)
    logger.info(f"[{split_name}] 原始样本数: {n_before}")

    # 清洗 text
    df["text"] = df["text"].apply(clean_text)

    # 删除空文本
    df = df[df["text"].str.len() > 0].copy()
    n_after_empty = len(df)
    if n_before - n_after_empty > 0:
        logger.info(f"  删除空文本: {n_before - n_after_empty} 条")

    # 删除重复文本
    df = df.drop_duplicates(subset=["text"], keep="first").copy()
    n_after_dedup = len(df)
    if n_after_empty - n_after_dedup > 0:
        logger.info(f"  删除重复文本: {n_after_empty - n_after_dedup} 条")

    # 检查非法标签
    valid_labels = {0, 1}
    invalid_mask = ~df["label"].isin(valid_labels)
    if invalid_mask.any():
        n_invalid = int(invalid_mask.sum())
        logger.warning(f"  发现 {n_invalid} 条非法标签，已删除")
        df = df[~invalid_mask].copy()

    # 重置 id
    df = df.reset_index(drop=True)
    df.insert(0, "id", range(len(df)))

    # 只保留需要的列
    df = df[["id", "text", "label"]]

    # 统计
    n_final = len(df)
    n_pos = int((df["label"] == 1).sum())
    n_neg = int((df["label"] == 0).sum())
    logger.info(
        f"  清洗后: {n_final} 条 (正向={n_pos}, 负向={n_neg})"
    )

    # 保存
    save_csv(df, output_path)
    logger.info(f"  已保存到 {output_path}")


def main():
    raw_dir = ROOT / "data" / "raw"
    processed_dir = ROOT / "data" / "processed"
    ensure_dir(str(processed_dir))

    splits = ["train", "dev", "test"]

    for split in splits:
        input_path = raw_dir / f"{split}.csv"
        if not input_path.exists():
            logger.warning(f"{input_path} 不存在，跳过")
            continue
        output_path = processed_dir / f"{split}.csv"
        preprocess_split(str(input_path), str(output_path), split)

    logger.info("预处理完成！")


if __name__ == "__main__":
    main()
