#!/usr/bin/env python
"""生成教师模型 logits"""
import argparse
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
from src.data.dataset import BertDataset
from src.models.bert_cls import BertClassifier

logger = get_logger("generate_teacher_logits")


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="生成教师模型 logits")
    parser.add_argument("--teacher_model", type=str, required=True, choices=["bert", "macbert"])
    parser.add_argument("--config", type=str, required=True, help="教师模型配置文件")
    parser.add_argument("--checkpoint", type=str, required=True, help="教师模型 checkpoint")
    parser.add_argument("--input_file", type=str, required=True, help="输入训练集 CSV")
    parser.add_argument("--output_file", type=str, required=True, help="输出 logits CSV")
    parser.add_argument("--max_samples", type=int, default=None, help="最大样本数")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    model_cfg = config["model"]
    data_cfg = config["data"]
    max_len = data_cfg["max_len"]
    batch_size = config["training"]["batch_size"]

    # 检查文件
    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint 不存在: {args.checkpoint}")
        sys.exit(1)
    if not Path(args.input_file).exists():
        logger.error(f"输入文件不存在: {args.input_file}")
        sys.exit(1)

    # 加载 tokenizer
    from transformers import AutoTokenizer
    pretrained_name = model_cfg["pretrained_model_name"]
    logger.info(f"加载 tokenizer: {pretrained_name}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

    # 数据集
    dataset = BertDataset(args.input_file, tokenizer, max_len, max_samples=args.max_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    logger.info(f"输入样本数: {len(dataset)}")

    # 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertClassifier(
        pretrained_model_name=pretrained_name,
        num_classes=model_cfg["num_classes"],
        dropout=model_cfg["dropout"],
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()
    logger.info(f"已加载教师模型: {args.checkpoint} | device={device}")

    # 生成 logits
    all_ids = []
    all_texts = []
    all_labels = []
    all_logits_0 = []
    all_logits_1 = []
    all_prob_0 = []
    all_prob_1 = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=-1).cpu()
            logits_cpu = logits.cpu()

            all_ids.extend([str(x) for x in batch["sample_id"]])
            all_texts.extend(batch["text"])
            all_labels.extend(batch["label"].tolist())
            all_logits_0.extend(logits_cpu[:, 0].tolist())
            all_logits_1.extend(logits_cpu[:, 1].tolist())
            all_prob_0.extend(probs[:, 0].tolist())
            all_prob_1.extend(probs[:, 1].tolist())

    # 保存
    ensure_dir(str(Path(args.output_file).parent))
    out_df = pd.DataFrame({
        "id": all_ids,
        "text": all_texts,
        "label": all_labels,
        "logit_0": all_logits_0,
        "logit_1": all_logits_1,
        "prob_0": all_prob_0,
        "prob_1": all_prob_1,
    })
    out_df.to_csv(args.output_file, index=False)
    logger.info(f"教师 logits 已保存: {args.output_file} ({len(out_df)} 条)")


if __name__ == "__main__":
    main()
