#!/usr/bin/env python
"""困难样本增强生成脚本（模板方式，不调用大模型 API）"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from src.utils.logger import get_logger
from src.utils.io_utils import ensure_dir
from src.augmentation.template_generator import generate_batch

logger = get_logger("generate_augmented")


def main():
    parser = argparse.ArgumentParser(description="生成困难样本增强候选")
    parser.add_argument("--config", type=str, required=True, help="增强配置文件")
    parser.add_argument("--input_file", type=str, required=True, help="困难样本输入文件")
    parser.add_argument("--output_file", type=str, required=True, help="增强样本输出文件")
    parser.add_argument("--target_per_type", type=int, default=None, help="每类生成数量")
    args = parser.parse_args()

    # 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    target_per_type = args.target_per_type or config.get("target_per_type", 20)
    seed = config.get("random_seed", 42)

    # 检查输入文件
    if not Path(args.input_file).exists():
        logger.error(f"输入文件不存在: {args.input_file}")
        sys.exit(1)

    logger.info(f"读取困难样本: {args.input_file}")
    df = pd.read_csv(args.input_file)
    logger.info(f"输入样本数: {len(df)}")

    # 转为字典列表
    samples = []
    for _, row in df.iterrows():
        samples.append({
            "id": row["id"],
            "text": str(row["text"]) if pd.notna(row["text"]) else "",
            "label": int(row["label"]),
            "hard_type": row.get("hard_type", "other"),
        })

    # 生成增强样本
    logger.info(f"每类目标生成数量: {target_per_type}")
    aug_samples = generate_batch(samples, target_per_type=target_per_type, seed=seed)
    logger.info(f"生成增强候选样本: {len(aug_samples)} 条")

    # 统计各类型
    type_counts = {}
    for s in aug_samples:
        ht = s["hard_type"]
        type_counts[ht] = type_counts.get(ht, 0) + 1
    logger.info(f"各类型数量: {type_counts}")

    # 保存
    ensure_dir(str(Path(args.output_file).parent))
    out_df = pd.DataFrame(aug_samples)
    out_df.to_csv(args.output_file, index=False)
    logger.info(f"输出文件: {args.output_file}")
    logger.info("增强样本生成完成。")


if __name__ == "__main__":
    main()
