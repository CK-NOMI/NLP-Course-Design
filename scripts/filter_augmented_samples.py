#!/usr/bin/env python
"""增强样本质量过滤脚本"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.utils.logger import get_logger
from src.utils.io_utils import ensure_dir
from src.augmentation.quality_filter import filter_samples

logger = get_logger("filter_augmented")


def main():
    parser = argparse.ArgumentParser(description="增强样本质量过滤")
    parser.add_argument("--input_file", type=str, required=True, help="增强候选样本文件")
    parser.add_argument("--output_file", type=str, required=True, help="过滤后输出文件")
    parser.add_argument("--min_text_len", type=int, default=4, help="最短文本长度")
    parser.add_argument("--max_text_len", type=int, default=256, help="最长文本长度")
    args = parser.parse_args()

    # 检查输入文件
    if not Path(args.input_file).exists():
        logger.error(f"输入文件不存在: {args.input_file}")
        sys.exit(1)

    logger.info(f"读取增强候选样本: {args.input_file}")
    df = pd.read_csv(args.input_file)
    logger.info(f"候选样本数: {len(df)}")

    # 转为字典列表
    samples = df.to_dict("records")

    # 过滤
    filtered = filter_samples(
        samples,
        min_text_len=args.min_text_len,
        max_text_len=args.max_text_len,
    )

    # 统计
    passed = [s for s in filtered if s.get("quality_pass")]
    failed = [s for s in filtered if not s.get("quality_pass")]
    logger.info(f"通过过滤: {len(passed)} 条")
    logger.info(f"未通过: {len(failed)} 条")

    if failed:
        reasons = {}
        for s in failed:
            r = s.get("filter_reason", "unknown")
            reasons[r] = reasons.get(r, 0) + 1
        logger.info(f"过滤原因: {reasons}")

    # 保存（包含所有样本，带 quality_pass 和 filter_reason 字段）
    ensure_dir(str(Path(args.output_file).parent))
    out_df = pd.DataFrame(filtered)
    out_df.to_csv(args.output_file, index=False)
    logger.info(f"输出文件: {args.output_file}")
    logger.info("质量过滤完成。")


if __name__ == "__main__":
    main()
