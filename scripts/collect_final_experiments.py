#!/usr/bin/env python
"""汇总最终实验结果：四模型基线 + TextCNN 消融实验"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.logger import get_logger
from src.utils.io_utils import ensure_dir

logger = get_logger("collect_final")

METRICS_DIR = Path("outputs/metrics")

# 四模型基线文件映射
BASELINE_FILES = {
    "textcnn": ("textcnn_test_metrics.json", "textcnn_hard_test_metrics.json"),
    "bilstm": ("bilstm_test_metrics.json", "bilstm_hard_test_metrics.json"),
    "bert": ("bert_formal_test_metrics.json", "bert_formal_hard_test_metrics.json"),
    "macbert": ("macbert_formal_test_metrics.json", "macbert_formal_hard_test_metrics.json"),
}

# TextCNN 消融实验文件映射
ABLATION_FILES = {
    "baseline": ("textcnn_test_metrics.json", "textcnn_hard_test_metrics.json"),
    "template_aug": ("textcnn_template_aug_test_metrics.json", "textcnn_template_aug_hard_test_metrics.json"),
    "distill_formal": ("textcnn_distill_macbert_formal_test_metrics.json", "textcnn_distill_macbert_formal_hard_test_metrics.json"),
    "template_aug+distill": ("textcnn_template_aug_distill_macbert_formal_test_metrics.json", "textcnn_template_aug_distill_macbert_formal_hard_test_metrics.json"),
    "llm_real_aug": ("textcnn_llm_real_aug_test_metrics.json", "textcnn_llm_real_aug_hard_test_metrics.json"),
    "llm_real_aug+distill": ("textcnn_llm_real_aug_distill_macbert_formal_test_metrics.json", "textcnn_llm_real_aug_distill_macbert_formal_hard_test_metrics.json"),
}


def load_metrics(filepath):
    if not filepath.exists():
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    output_dir = Path("outputs/reports")
    figures_dir = Path("outputs/figures")
    ensure_dir(str(output_dir))
    ensure_dir(str(figures_dir))

    missing = []

    # === 四模型基线汇总 ===
    baseline_rows = []
    for model, (test_file, hard_file) in BASELINE_FILES.items():
        test_path = METRICS_DIR / test_file
        hard_path = METRICS_DIR / hard_file
        test_data = load_metrics(test_path)
        hard_data = load_metrics(hard_path)

        if test_data is None:
            missing.append(test_file)
        if hard_data is None:
            missing.append(hard_file)

        baseline_rows.append({
            "model": model,
            "test_accuracy": test_data["accuracy"] if test_data else "",
            "test_macro_f1": test_data["macro_f1"] if test_data else "",
            "hard_accuracy": hard_data["accuracy"] if hard_data else "",
            "hard_macro_f1": hard_data["macro_f1"] if hard_data else "",
        })

    # 输出 baseline CSV
    csv_path = output_dir / "final_baseline_summary.csv"
    lines = ["model,test_accuracy,test_macro_f1,hard_accuracy,hard_macro_f1"]
    for r in baseline_rows:
        lines.append(f"{r['model']},{r['test_accuracy']},{r['test_macro_f1']},{r['hard_accuracy']},{r['hard_macro_f1']}")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Baseline CSV: {csv_path}")

    # 输出 baseline Markdown
    md_path = output_dir / "final_baseline_summary.md"
    md_lines = [
        "# Final Baseline Results (4 Models)\n",
        "| Model | Test Acc | Test F1 | Hard Acc | Hard F1 |",
        "|---|---|---|---|---|",
    ]
    for r in baseline_rows:
        md_lines.append(f"| {r['model']} | {r['test_accuracy']} | {r['test_macro_f1']} | {r['hard_accuracy']} | {r['hard_macro_f1']} |")
    md_lines.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Baseline MD: {md_path}")

    # === TextCNN 消融实验汇总 ===
    ablation_rows = []
    baseline_hard_f1 = None
    for setting, (test_file, hard_file) in ABLATION_FILES.items():
        test_path = METRICS_DIR / test_file
        hard_path = METRICS_DIR / hard_file
        test_data = load_metrics(test_path)
        hard_data = load_metrics(hard_path)

        if test_data is None:
            missing.append(test_file)
        if hard_data is None:
            missing.append(hard_file)

        test_f1 = test_data["macro_f1"] if test_data else 0
        hard_f1 = hard_data["macro_f1"] if hard_data else 0

        if setting == "baseline":
            baseline_hard_f1 = hard_f1

        improvement = round(hard_f1 - baseline_hard_f1, 4) if baseline_hard_f1 is not None else ""

        ablation_rows.append({
            "setting": setting,
            "test_macro_f1": test_f1,
            "hard_macro_f1": hard_f1,
            "hard_improvement_vs_baseline": improvement,
        })

    # 输出 ablation CSV
    csv_path = output_dir / "final_textcnn_ablation_summary.csv"
    lines = ["setting,test_macro_f1,hard_macro_f1,hard_improvement_vs_baseline"]
    for r in ablation_rows:
        lines.append(f"{r['setting']},{r['test_macro_f1']},{r['hard_macro_f1']},{r['hard_improvement_vs_baseline']}")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Ablation CSV: {csv_path}")

    # 输出 ablation Markdown
    md_path = output_dir / "final_textcnn_ablation_summary.md"
    md_lines = [
        "# TextCNN Ablation Results\n",
        "| Setting | Test F1 | Hard F1 | Hard Improvement |",
        "|---|---|---|---|",
    ]
    for r in ablation_rows:
        imp = r['hard_improvement_vs_baseline']
        imp_str = f"+{imp}" if isinstance(imp, float) and imp > 0 else str(imp)
        md_lines.append(f"| {r['setting']} | {r['test_macro_f1']} | {r['hard_macro_f1']} | {imp_str} |")
    md_lines.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Ablation MD: {md_path}")

    # === 图1：四模型 test / hard_test Macro-F1 对比 ===
    models = [r["model"] for r in baseline_rows]
    test_f1s = [float(r["test_macro_f1"]) if r["test_macro_f1"] else 0 for r in baseline_rows]
    hard_f1s = [float(r["hard_macro_f1"]) if r["hard_macro_f1"] else 0 for r in baseline_rows]

    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, test_f1s, width, label="Test", color="steelblue")
    bars2 = ax.bar(x + width/2, hard_f1s, width, label="Hard Test", color="coral")
    ax.set_title("Baseline Models: Test vs Hard Test Macro-F1", fontsize=14)
    ax.set_ylabel("Macro-F1", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.8, 0.95)
    ax.legend()
    for bar, val in zip(bars1, test_f1s):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars2, hard_f1s):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig_path = figures_dir / "final_baseline_macro_f1.png"
    plt.savefig(str(fig_path), dpi=150)
    plt.close()
    logger.info(f"Figure: {fig_path}")

    # === 图2：TextCNN 消融 hard_test Macro-F1 ===
    settings = [r["setting"] for r in ablation_rows]
    hard_vals = [float(r["hard_macro_f1"]) if r["hard_macro_f1"] else 0 for r in ablation_rows]

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["steelblue", "seagreen", "darkorange", "crimson", "mediumpurple", "deeppink"]
    bars = ax.bar(settings, hard_vals, color=colors[:len(settings)], width=0.5)
    ax.set_title("TextCNN Hard-Test Macro-F1 under Augmentation and Distillation Settings", fontsize=12)
    ax.set_ylabel("Macro-F1", fontsize=12)
    ax.set_xlabel("Setting", fontsize=12)
    ax.set_ylim(0.88, 0.94)
    plt.xticks(rotation=15, ha="right")
    for bar, val in zip(bars, hard_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, f"{val:.4f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    fig_path = figures_dir / "final_textcnn_ablation_hard_f1.png"
    plt.savefig(str(fig_path), dpi=150)
    plt.close()
    logger.info(f"Figure: {fig_path}")

    # 报告缺失
    if missing:
        logger.warning(f"缺失文件: {missing}")
    else:
        logger.info("所有 metrics 文件齐全。")

    logger.info("最终实验结果汇总完成。")


if __name__ == "__main__":
    main()
