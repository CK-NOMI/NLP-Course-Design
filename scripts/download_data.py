#!/usr/bin/env python
"""下载 ChnSentiCorp 数据集并保存到 data/raw/"""
import sys
from pathlib import Path

# 项目根目录加入 path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets import load_dataset
import pandas as pd

from src.utils.io_utils import ensure_dir, save_csv
from src.utils.logger import get_logger

logger = get_logger("download_data")


def main():
    raw_dir = ROOT / "data" / "raw"
    ensure_dir(str(raw_dir))

    logger.info("正在从 HuggingFace 下载 ChnSentiCorp ...")
    ds = load_dataset("lansinuote/ChnSentiCorp")

    # 兼容不同字段名: train / validation / test
    split_map = {
        "train": "train",
        "validation": "dev",
        "test": "test",
    }

    for src_split, dst_name in split_map.items():
        if src_split not in ds:
            logger.warning(f"数据集中不存在 {src_split} 划分，跳过")
            continue

        split_data = ds[src_split]
        df = split_data.to_pandas()

        # 确保字段名统一
        if "text" not in df.columns and "review" in df.columns:
            df = df.rename(columns={"review": "text"})

        # 只保留 text 和 label
        df = df[["text", "label"]].copy()
        df = df.reset_index(drop=True)

        out_path = raw_dir / f"{dst_name}.csv"
        save_csv(df, str(out_path))

        # 统计
        n_total = len(df)
        n_pos = int((df["label"] == 1).sum())
        n_neg = int((df["label"] == 0).sum())
        logger.info(
            f"  {dst_name}: {n_total} 条 (正向={n_pos}, 负向={n_neg})"
        )

    logger.info(f"原始数据已保存到 {raw_dir}")


if __name__ == "__main__":
    main()
