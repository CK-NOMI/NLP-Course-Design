#!/usr/bin/env python
"""汇总实验结果：扫描 metrics JSON 生成结果表"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger
from src.utils.io_utils import ensure_dir

logger = get_logger("collect_results")

# 正式结果文件映射：(显示名, dataset, 文件名)
# BERT/MacBERT 优先使用 formal 文件
RESULT_FILES = [
    ("textcnn", "test", "textcnn_test_metrics.json"),
    ("textcnn", "hard_test", "textcnn_hard_test_metrics.json"),
    ("bilstm", "test", "bilstm_test_metrics.json"),
    ("bilstm", "hard_test", "bilstm_hard_test_metrics.json"),
    ("bert", "test", "bert_formal_test_metrics.json"),
    ("bert", "hard_test", "bert_formal_hard_test_metrics.json"),
    ("macbert", "test", "macbert_formal_test_metrics.json"),
    ("macbert", "hard_test", "macbert_formal_hard_test_metrics.json"),
]

# 模型列表（用于完整性检查）
MODELS = ["textcnn", "bilstm", "bert", "macbert"]


def main():
    parser = argparse.ArgumentParser(description="汇总实验结果")
    parser.add_argument("--metrics_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--allow_incomplete", action="store_true", default=False,
                        help="允许汇总只有 test 或只有 hard_test 的模型")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    ensure_dir(args.output_dir)

    # 检查每个模型的文件存在情况
    model_files = {}  # model -> {"test": path_or_None, "hard_test": path_or_None}
    for model, dataset, filename in RESULT_FILES:
        if model not in model_files:
            model_files[model] = {"test": None, "hard_test": None}
        filepath = metrics_dir / filename
        if filepath.exists():
            model_files[model][dataset] = filepath

    # 决定哪些模型纳入汇总
    included_models = []
    skipped_models = []
    missing_info = []

    for model in MODELS:
        files = model_files.get(model, {"test": None, "hard_test": None})
        has_test = files["test"] is not None
        has_hard = files["hard_test"] is not None

        if has_test and has_hard:
            included_models.append(model)
        elif args.allow_incomplete and (has_test or has_hard):
            included_models.append(model)
            if not has_test:
                missing_info.append(f"- {model}: missing test metrics")
            if not has_hard:
                missing_info.append(f"- {model}: missing hard_test metrics")
        else:
            skipped_models.append(model)
            reasons = []
            if not has_test:
                reasons.append("missing test")
            if not has_hard:
                reasons.append("missing hard_test")
            missing_info.append(f"- {model}: SKIPPED ({', '.join(reasons)})")

    # 读取数据
    rows = []
    for model in included_models:
        files = model_files[model]
        for dataset in ["test", "hard_test"]:
            filepath = files[dataset]
            if filepath is None:
                continue
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            rows.append({
                "model": model,
                "dataset": dataset,
                "accuracy": data.get("accuracy", ""),
                "macro_f1": data.get("macro_f1", ""),
                "precision": data.get("precision", ""),
                "recall": data.get("recall", ""),
                "metrics_file": filepath.name,
            })

    # 日志
    logger.info(f"纳入汇总的模型: {included_models}")
    if skipped_models:
        logger.info(f"跳过的模型（结果不完整）: {skipped_models}")

    # 输出 CSV
    csv_path = Path(args.output_dir) / "results_summary.csv"
    header = "model,dataset,accuracy,macro_f1,precision,recall,metrics_file"
    lines = [header]
    for r in rows:
        lines.append(f"{r['model']},{r['dataset']},{r['accuracy']},{r['macro_f1']},{r['precision']},{r['recall']},{r['metrics_file']}")

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"CSV 已保存: {csv_path}")

    # 输出 Markdown
    md_path = Path(args.output_dir) / "results_summary.md"
    md_lines = [
        "# Experiment Results Summary\n",
        "| Model | Dataset | Accuracy | Macro-F1 | Precision | Recall |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['model']} | {r['dataset']} | {r['accuracy']} | {r['macro_f1']} | {r['precision']} | {r['recall']} |"
        )
    md_lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Markdown 已保存: {md_path}")
    logger.info(f"共汇总 {len(rows)} 条结果。")

    # 输出 missing_results.md
    missing_path = Path(args.output_dir) / "missing_results.md"
    missing_lines = [
        "# Missing Results Report\n",
        f"Included models: {', '.join(included_models) if included_models else 'none'}\n",
        f"Skipped models: {', '.join(skipped_models) if skipped_models else 'none'}\n",
        "## Details\n",
    ]
    if missing_info:
        for line in missing_info:
            missing_lines.append(line)
    else:
        missing_lines.append("All models have complete results.")
    missing_lines.append("")

    with open(missing_path, "w", encoding="utf-8") as f:
        f.write("\n".join(missing_lines))
    logger.info(f"Missing report 已保存: {missing_path}")


if __name__ == "__main__":
    main()
