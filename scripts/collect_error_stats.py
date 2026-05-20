#!/usr/bin/env python
"""汇总误判分析结果"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger
from src.utils.io_utils import ensure_dir

logger = get_logger("collect_error_stats")

# 预期的 summary 文件
EXPECTED_SUMMARIES = [
    ("textcnn", "test", "textcnn_test_summary.json"),
    ("textcnn", "hard_test", "textcnn_hard_test_summary.json"),
    ("bilstm", "test", "bilstm_test_summary.json"),
    ("bilstm", "hard_test", "bilstm_hard_test_summary.json"),
    ("bert", "test", "bert_test_summary.json"),
    ("bert", "hard_test", "bert_hard_test_summary.json"),
    ("macbert", "test", "macbert_test_summary.json"),
    ("macbert", "hard_test", "macbert_hard_test_summary.json"),
]


def main():
    parser = argparse.ArgumentParser(description="汇总误判分析结果")
    parser.add_argument("--error_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    error_dir = Path(args.error_dir)
    ensure_dir(args.output_dir)

    rows = []
    for model, dataset, filename in EXPECTED_SUMMARIES:
        filepath = error_dir / filename
        if not filepath.exists():
            logger.info(f"跳过（文件不存在）: {filename}")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        total = data.get("total_samples", 0)
        errors = data.get("error_samples", 0)
        error_rate = data.get("error_rate", 0)
        type_dist = data.get("type_distribution", {})

        for hard_type, count in type_dist.items():
            ratio = round(count / max(errors, 1), 4)
            rows.append({
                "model": model,
                "dataset": dataset,
                "total_samples": total,
                "error_samples": errors,
                "error_rate": error_rate,
                "hard_type": hard_type,
                "count": count,
                "ratio": ratio,
            })

    if not rows:
        logger.warning("没有找到任何误判分析文件。")
        return

    # 输出 CSV
    csv_path = Path(args.output_dir) / "error_stats_summary.csv"
    header = "model,dataset,total_samples,error_samples,error_rate,hard_type,count,ratio"
    lines = [header]
    for r in rows:
        lines.append(
            f"{r['model']},{r['dataset']},{r['total_samples']},{r['error_samples']},"
            f"{r['error_rate']},{r['hard_type']},{r['count']},{r['ratio']}"
        )

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"CSV 已保存: {csv_path}")

    # 输出 Markdown
    md_path = Path(args.output_dir) / "error_stats_summary.md"
    md_lines = [
        "# Error Analysis Summary\n",
        "| Model | Dataset | Total | Errors | Error Rate | Hard Type | Count | Ratio |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['model']} | {r['dataset']} | {r['total_samples']} | {r['error_samples']} | "
            f"{r['error_rate']} | {r['hard_type']} | {r['count']} | {r['ratio']} |"
        )
    md_lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Markdown 已保存: {md_path}")
    logger.info(f"共汇总 {len(rows)} 条记录。")


if __name__ == "__main__":
    main()
