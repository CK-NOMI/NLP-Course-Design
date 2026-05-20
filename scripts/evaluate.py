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
from src.data.dataset import CharDataset, BertDataset
from src.models.textcnn import TextCNN
from src.models.bilstm import BiLSTM
from src.models.bert_cls import BertClassifier
from src.evaluation.metrics import compute_metrics

logger = get_logger("evaluate")


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="评估模型")
    parser.add_argument("--model", type=str, required=True, choices=["textcnn", "bilstm", "bert", "macbert"])
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_test_samples", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    data_cfg = config["data"]
    model_cfg = config["model"]
    max_len = data_cfg["max_len"]
    batch_size = config["training"]["batch_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 根据模型类型选择数据集和模型 ===
    if args.model in ("textcnn", "bilstm"):
        # 字级模型
        vocab = CharVocab.load(data_cfg["vocab_path"])
        logger.info(f"Vocab 大小: {len(vocab)}")

        test_ds = CharDataset(args.test_file, vocab, max_len, max_samples=args.max_test_samples)

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
            model = BiLSTM(
                vocab_size=len(vocab),
                embed_dim=model_cfg["embed_dim"],
                hidden_dim=model_cfg["hidden_dim"],
                num_layers=model_cfg["num_layers"],
                dropout=model_cfg["dropout"],
                num_classes=model_cfg["num_classes"],
                bidirectional=model_cfg.get("bidirectional", True),
            )

    elif args.model in ("bert", "macbert"):
        # 预训练模型
        from transformers import AutoTokenizer

        pretrained_name = model_cfg["pretrained_model_name"]
        logger.info(f"加载 tokenizer: {pretrained_name}")
        tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

        test_ds = BertDataset(args.test_file, tokenizer, max_len, max_samples=args.max_test_samples)

        model = BertClassifier(
            pretrained_model_name=pretrained_name,
            num_classes=model_cfg["num_classes"],
            dropout=model_cfg["dropout"],
        )

    else:
        raise ValueError(f"不支持的模型: {args.model}")

    # 加载 checkpoint
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()
    logger.info(f"已加载 checkpoint: {args.checkpoint}")

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    logger.info(f"测试集: {len(test_ds)} 条")

    # === 预测 ===
    all_ids = []
    all_texts = []
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)

            if "attention_mask" in batch:
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids, attention_mask=attention_mask)
            else:
                logits = model(input_ids)

            probs = torch.softmax(logits, dim=-1).cpu()
            preds = logits.argmax(dim=-1).cpu().tolist()

            all_ids.extend([str(x) for x in batch["sample_id"]])
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

    # 保存 metrics（文件名根据 --output 的 prediction 文件名自动生成）
    metrics_dir = config["output"]["metrics_dir"]
    ensure_dir(metrics_dir)
    output_stem = Path(args.output).stem  # e.g. textcnn_test_predictions
    metrics_stem = output_stem.replace("_predictions", "_metrics")
    metrics_path = Path(metrics_dir) / f"{metrics_stem}.json"
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
