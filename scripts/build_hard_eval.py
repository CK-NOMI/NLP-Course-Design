#!/usr/bin/env python
"""困难评估集初筛脚本：从 dev/test 中用规则筛选困难样本"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.utils.logger import get_logger
from src.utils.io_utils import ensure_dir
from src.evaluation.error_analysis import classify_hard_type

logger = get_logger("build_hard_eval")

# 短文本中命中这些强规则词时，即使长度 < min_text_len 也保留
STRONG_RULE_WORDS = ["踩雷", "避雷", "无语", "回购", "推荐", "浪费", "值得", "离谱"]


def main():
    parser = argparse.ArgumentParser(description="构造困难样本评估集")
    parser.add_argument("--input_file", type=str, required=True, help="输入 CSV 文件路径")
    parser.add_argument("--output_file", type=str, required=True, help="输出 CSV 文件路径")
    parser.add_argument("--include_other", action="store_true", default=False,
                        help="是否保留 other 类型（默认不保留）")
    parser.add_argument("--max_per_type", type=int, default=80,
                        help="每类最多保留多少条（默认 80）")
    parser.add_argument("--max_total", type=int, default=300,
                        help="总数最多保留多少条（默认 300）")
    parser.add_argument("--min_text_len", type=int, default=6,
                        help="最短文本长度（默认 6，过短文本除非命中强规则否则排除）")
    args = parser.parse_args()

    # 检查输入文件
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"输入文件不存在: {args.input_file}")
        sys.exit(1)

    # 警告：不应从训练集构造困难评估集
    if "train" in input_path.name.lower():
        logger.warning("警告：输入文件看起来是训练集。困难评估集应从 dev/test 中筛选，不应从训练集构造。")

    logger.info(f"读取输入文件: {args.input_file}")
    df = pd.read_csv(args.input_file)

    # 校验字段
    required_cols = ["id", "text", "label"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"输入文件缺少字段: {missing}")
        sys.exit(1)

    logger.info(f"输入样本数: {len(df)}")

    # 对每条样本进行困难类型判断
    hard_types = []
    rule_hits_list = []
    for _, row in df.iterrows():
        text = str(row["text"]) if pd.notna(row["text"]) else ""
        result = classify_hard_type(text)
        hard_types.append(result["hard_type"])
        rule_hits_list.append("|".join(result["rule_hits"]))

    df["hard_type"] = hard_types
    df["rule_hits"] = rule_hits_list

    # 筛选：排除 other（除非 include_other）
    if args.include_other:
        hard_df = df.copy()
    else:
        hard_df = df[df["hard_type"] != "other"].copy()

    # 过滤过短文本（除非命中强规则词）
    def _keep_row(row):
        text = str(row["text"]) if pd.notna(row["text"]) else ""
        if len(text) >= args.min_text_len:
            return True
        # 短文本：检查是否命中强规则词
        for w in STRONG_RULE_WORDS:
            if w in text:
                return True
        return False

    hard_df = hard_df[hard_df.apply(_keep_row, axis=1)].copy()

    # 每类最多保留 max_per_type 条
    if args.max_per_type is not None:
        parts = []
        for _, group in hard_df.groupby("hard_type"):
            parts.append(group.head(args.max_per_type))
        hard_df = pd.concat(parts, ignore_index=True)

    # 总数最多 max_total 条（按类型比例均匀采样）
    if args.max_total is not None and len(hard_df) > args.max_total:
        # 按类型比例分配名额
        type_counts_current = hard_df["hard_type"].value_counts()
        total_current = len(hard_df)
        parts = []
        remaining = args.max_total
        types_sorted = type_counts_current.index.tolist()

        for i, t in enumerate(types_sorted):
            if i == len(types_sorted) - 1:
                quota = remaining
            else:
                quota = max(1, int(args.max_total * type_counts_current[t] / total_current))
                quota = min(quota, remaining)
            type_subset = hard_df[hard_df["hard_type"] == t].head(quota)
            parts.append(type_subset)
            remaining -= len(type_subset)

        hard_df = pd.concat(parts, ignore_index=True)

    # 只保留需要的字段
    hard_df = hard_df[["id", "text", "label", "hard_type", "rule_hits"]].copy()
    hard_df = hard_df.reset_index(drop=True)

    # 输出 hard_test.csv
    output_path = Path(args.output_file)
    ensure_dir(str(output_path.parent))
    hard_df.to_csv(output_path, index=False)

    # 输出 type stats
    type_counts = hard_df["hard_type"].value_counts()
    type_stats = pd.DataFrame({
        "hard_type": type_counts.index,
        "count": type_counts.values,
        "ratio": (type_counts.values / max(len(hard_df), 1)).round(4),
    })
    stats_path = output_path.parent / (output_path.stem + "_stats.csv")
    type_stats.to_csv(stats_path, index=False)

    # 日志
    logger.info(f"困难样本总数: {len(hard_df)}")
    logger.info("各类型数量:")
    for _, row in type_stats.iterrows():
        logger.info(f"  {row['hard_type']}: {row['count']}")
    logger.info(f"输出文件: {args.output_file}")
    logger.info(f"统计文件: {stats_path}")
    logger.info("困难评估集构造完成。")


if __name__ == "__main__":
    main()
