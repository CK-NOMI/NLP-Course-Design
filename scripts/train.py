#!/usr/bin/env python
"""统一训练入口"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
from torch.utils.data import DataLoader

from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.data.vocab import CharVocab
from src.data.dataset import CharDataset, BertDataset
from src.models.textcnn import TextCNN
from src.models.bilstm import BiLSTM
from src.models.bert_cls import BertClassifier
from src.training.trainer import Trainer

logger = get_logger("train")


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="训练模型")
    parser.add_argument("--model", type=str, required=True, choices=["textcnn", "bilstm", "bert", "macbert"])
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_dev_samples", type=int, default=None)
    parser.add_argument("--override_epochs", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    # 覆盖 epochs
    if args.override_epochs is not None:
        config["training"]["epochs"] = args.override_epochs

    data_cfg = config["data"]
    model_cfg = config["model"]
    max_len = data_cfg["max_len"]
    batch_size = config["training"]["batch_size"]

    # === 根据模型类型选择数据集和模型 ===
    if args.model in ("textcnn", "bilstm"):
        # 字级模型：使用 CharVocab + CharDataset
        vocab_path = data_cfg["vocab_path"]
        if Path(vocab_path).exists():
            logger.info(f"加载已有 vocab: {vocab_path}")
            vocab = CharVocab.load(vocab_path)
        else:
            logger.info(f"构建 vocab (min_freq={data_cfg['vocab_min_freq']}) ...")
            vocab = CharVocab.build_from_csv(
                data_cfg["train_file"],
                min_freq=data_cfg["vocab_min_freq"],
            )
            vocab.save(vocab_path)
            logger.info(f"Vocab 大小: {len(vocab)} | 已保存到 {vocab_path}")

        train_ds = CharDataset(data_cfg["train_file"], vocab, max_len, max_samples=args.max_train_samples)
        dev_ds = CharDataset(data_cfg["dev_file"], vocab, max_len, max_samples=args.max_dev_samples)

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
        # 预训练模型：使用 AutoTokenizer + BertDataset
        from transformers import AutoTokenizer

        pretrained_name = model_cfg["pretrained_model_name"]
        logger.info(f"加载 tokenizer: {pretrained_name}")
        tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

        train_ds = BertDataset(data_cfg["train_file"], tokenizer, max_len, max_samples=args.max_train_samples)
        dev_ds = BertDataset(data_cfg["dev_file"], tokenizer, max_len, max_samples=args.max_dev_samples)

        model = BertClassifier(
            pretrained_model_name=pretrained_name,
            num_classes=model_cfg["num_classes"],
            dropout=model_cfg["dropout"],
        )

    else:
        raise ValueError(f"不支持的模型: {args.model}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)

    logger.info(f"Train: {len(train_ds)} 条 | Dev: {len(dev_ds)} 条")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型: {args.model} | 参数量: {n_params:,}")

    # === Training ===
    trainer = Trainer(model, train_loader, dev_loader, config, args.exp_name)
    best_ckpt = trainer.train()
    logger.info(f"训练完成！Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
