#!/usr/bin/env python
"""误判分析脚本：从预测文件中提取误判样本并按困难类型分类"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger
from src.evaluation.error_analysis import analyze_prediction_file

logger = get_logger("error_analysis")


def main():
    parser = argparse.ArgumentParser(description="误判分析")
    parser.add_argument("--pred_file", type=str, required=True, help="预测文件路径")
    parser.add_argument("--model_name", type=str, required=True, help="模型名称")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    args = parser.parse_args()

    # 检查文件是否存在
    if not Path(args.pred_file).exists():
        logger.error(f"预测文件不存在: {args.pred_file}")
        logger.error("请先运行模型评估生成预测文件。")
        sys.exit(1)

    logger.info(f"开始误判分析: {args.pred_file}")
    logger.info(f"模型名称: {args.model_name}")
    logger.info(f"输出目录: {args.output_dir}")

    try:
        result = analyze_prediction_file(
            pred_file=args.pred_file,
            output_dir=args.output_dir,
            model_name=args.model_name,
        )
    except Exception as e:
        logger.error(f"分析失败: {e}")
        sys.exit(1)

    summary = result["summary"]
    logger.info(f"总样本数: {summary['total_samples']}")
    logger.info(f"误判样本数: {summary['error_samples']}")
    logger.info(f"准确率: {summary['accuracy']}")
    logger.info(f"误判率: {summary['error_rate']}")
    logger.info(f"类型分布: {summary['type_distribution']}")
    logger.info(f"误判明细: {result['error_samples_path']}")
    logger.info(f"类型统计: {result['type_stats_path']}")
    logger.info(f"Summary: {result['summary_path']}")
    logger.info("误判分析完成。")


if __name__ == "__main__":
    main()
