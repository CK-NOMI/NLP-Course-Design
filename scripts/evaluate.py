#!/usr/bin/env python
"""统一评估入口"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import pandas as pd
import yaml
from torch.utils.data import DataLoader

from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.utils.io_utils import ensure_dir
from src.data.vocab import CharVocab
from src.data.dataset import CharDataset
from src.models.textcnn import TextCNN
from src.evaluation.metrics import compute_metrics

logger = get_logger("evaluate")


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="评估模型")
    parser.add_argument("--model", type=str, required=True, choices=["textcnn"])
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    data_cfg = config["data"]
    model_cfg = config["model"]

    # === Vocab ===
    vocab = CharVocab.load(data_cfg["vocab_path"])
    logger.info(f"Vocab 大小: {len(vocab)}")

    # === Dataset ===
    max_len = data_cfg["max_len"]
    test_ds = CharDataset(args.test_file, vocab, max_len)
    test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"], shuffle=False)
    logger.info(f"测试集: {len(test_ds)} 条")

    # === Model ===
    if args.model == "textcnn":
        model = TextCNN(
            vocab_size=len(vocab),
            embed_dim=model_cfg["embed_dim"],
            num_filters=model_cfg["num_filters"],
            kernel_sizes=model_cfg["kernel_sizes"],
            dropout=model_cfg["dropout"],
            num_classes=model_cfg["num_classes"],
        )
    else:
        raise ValueError(f"不支持的模型: {args.model}")

    # 加载 checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()
    logger.info(f"已加载 checkpoint: {args.checkpoint}")

    # === 预测 ===
    all_ids = []
    all_texts = []
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            logits = model(input_ids)
            probs = torch.softmax(logits, dim=-1).cpu()
            preds = logits.argmax(dim=-1).cpu().tolist()

            all_ids.extend([int(x) for x in batch["sample_id"]])
            all_texts.extend(batch["text"])
            all_labels.extend(batch["label"].tolist())
            all_preds.extend(preds)
            all_probs.extend(probs.tolist())

    # === 指标 ===
    metrics = compute_metrics(all_labels, all_preds)
    logger.info(f"Test Accuracy: {metrics['accuracy']}")
    logger.info(f"Test Macro-F1: {metrics['macro_f1']}")
    logger.info(f"Test Precision: {metrics['precision']}")
    logger.info(f"Test Recall: {metrics['recall']}")

    # 保存 metrics
    metrics_dir = config["output"]["metrics_dir"]
    ensure_dir(metrics_dir)
    metrics_path = Path(metrics_dir) / f"{args.model}_test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"指标已保存到 {metrics_path}")

    # === 预测文件 ===
    ensure_dir(str(Path(args.output).parent))
    result_df = pd.DataFrame({
        "id": all_ids,
        "text": all_texts,
        "label": all_labels,
        "pred": all_preds,
        "prob_neg": [p[0] for p in all_probs],
        "prob_pos": [p[1] for p in all_probs],
        "is_correct": [l == p for l, p in zip(all_labels, all_preds)],
    })
    result_df.to_csv(args.output, index=False)
    logger.info(f"预测文件已保存到 {args.output}")


if __name__ == "__main__":
    main()
