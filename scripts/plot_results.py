#!/usr/bin/env python
"""生成实验结果图表"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.logger import get_logger
from src.utils.io_utils import ensure_dir

logger = get_logger("plot_results")


def read_csv_simple(filepath):
    """简单读取 CSV（不依赖 pandas）"""
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        for line in f:
            values = line.strip().split(",")
            if len(values) == len(header):
                rows.append(dict(zip(header, values)))
    return rows


def plot_bar(models, values, title, ylabel, output_path, color="steelblue"):
    """绘制柱状图"""
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, values, color=color, width=0.5)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Model", fontsize=12)

    # 在柱子上方标注数值
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"图表已保存: {output_path}")


def plot_grouped_bar(models, test_values, hard_values, title, output_path):
    """绘制分组柱状图：test vs hard_test"""
    import numpy as np
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, test_values, width, label="test", color="steelblue")
    bars2 = ax.bar(x + width / 2, hard_values, width, label="hard_test", color="coral")

    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Macro-F1", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.0)
    ax.legend()

    for bar, val in zip(bars1, test_values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars2, hard_values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"图表已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="生成实验结果图表")
    parser.add_argument("--input_file", type=str, required=True, help="results_summary.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="图表输出目录")
    args = parser.parse_args()

    if not Path(args.input_file).exists():
        logger.error(f"输入文件不存在: {args.input_file}")
        sys.exit(1)

    ensure_dir(args.output_dir)
    rows = read_csv_simple(args.input_file)

    if not rows:
        logger.warning("结果文件为空，无法生成图表。")
        return

    # 分离 test 和 hard_test 结果
    test_results = {}
    hard_results = {}
    for r in rows:
        model = r["model"]
        f1 = float(r["macro_f1"]) if r["macro_f1"] else 0
        if r["dataset"] == "test":
            test_results[model] = f1
        elif r["dataset"] == "hard_test":
            hard_results[model] = f1

    # 图1：标准 test Macro-F1
    if test_results:
        models = list(test_results.keys())
        values = list(test_results.values())
        plot_bar(
            models, values,
            "Macro-F1 on Standard Test Set",
            "Macro-F1",
            str(Path(args.output_dir) / "model_test_macro_f1.png"),
        )
    else:
        logger.info("无标准 test 结果，跳过 model_test_macro_f1.png")

    # 图2：hard_test Macro-F1
    if hard_results:
        models = list(hard_results.keys())
        values = list(hard_results.values())
        plot_bar(
            models, values,
            "Macro-F1 on Hard Test Set",
            "Macro-F1",
            str(Path(args.output_dir) / "model_hard_test_macro_f1.png"),
            color="coral",
        )
    else:
        logger.info("无 hard_test 结果，跳过 model_hard_test_macro_f1.png")

    # 图3：test vs hard_test 对比
    all_models = sorted(set(list(test_results.keys()) + list(hard_results.keys())))
    if all_models and test_results and hard_results:
        test_vals = [test_results.get(m, 0) for m in all_models]
        hard_vals = [hard_results.get(m, 0) for m in all_models]
        plot_grouped_bar(
            all_models, test_vals, hard_vals,
            "Test vs Hard Test: Macro-F1 Comparison",
            str(Path(args.output_dir) / "test_vs_hard_macro_f1.png"),
        )
    else:
        logger.info("数据不足，跳过 test_vs_hard_macro_f1.png")

    logger.info("图表生成完成。")


if __name__ == "__main__":
    main()
