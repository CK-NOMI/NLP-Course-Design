#!/usr/bin/env python
"""构造增强训练集：合并原始训练集和过滤后的增强样本"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger
from src.augmentation.build_augmented_dataset import build_augmented_train

logger = get_logger("build_augmented_train")


def main():
    parser = argparse.ArgumentParser(description="构造增强训练集")
    parser.add_argument("--train_file", type=str, required=True, help="原始训练集")
    parser.add_argument("--aug_file", type=str, required=True, help="过滤后增强样本文件")
    parser.add_argument("--output_file", type=str, required=True, help="输出增强训练集")
    args = parser.parse_args()

    # 检查文件
    if not Path(args.train_file).exists():
        logger.error(f"训练集不存在: {args.train_file}")
        sys.exit(1)
    if not Path(args.aug_file).exists():
        logger.error(f"增强样本文件不存在: {args.aug_file}")
        sys.exit(1)

    logger.info(f"原始训练集: {args.train_file}")
    logger.info(f"增强样本: {args.aug_file}")
    logger.info(f"输出文件: {args.output_file}")

    result = build_augmented_train(
        train_file=args.train_file,
        aug_file=args.aug_file,
        output_file=args.output_file,
    )

    logger.info(f"原始训练集: {result['original_count']} 条")
    logger.info(f"增强样本: {result['augmented_count']} 条")
    logger.info(f"合并后总数: {result['total_count']} 条")
    logger.info(f"输出文件: {result['output_file']}")
    logger.info("增强训练集构造完成。")


if __name__ == "__main__":
    main()
