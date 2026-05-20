#!/usr/bin/env python
"""蒸馏训练入口"""
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
from src.data.dataset import CharDataset
from src.distillation.distill_dataset import DistillCharDataset
from src.distillation.distill_trainer import DistillTrainer
from src.models.textcnn import TextCNN
from src.models.bilstm import BiLSTM

logger = get_logger("train_distilled")


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="蒸馏训练")
    parser.add_argument("--config", type=str, required=True, help="蒸馏配置文件")
    parser.add_argument("--student_model", type=str, required=True, choices=["textcnn", "bilstm"])
    parser.add_argument("--teacher_logits_file", type=str, required=True, help="教师 logits 文件")
    parser.add_argument("--exp_name", type=str, required=True, help="实验名称")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_dev_samples", type=int, default=None)
    parser.add_argument("--override_epochs", type=int, default=None)
    args = parser.parse_args()

    # 加载蒸馏配置
    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    if args.override_epochs is not None:
        config["training"]["epochs"] = args.override_epochs

    # 加载学生模型配置
    student_cfg_path = config["student"]["config_file"]
    student_config = load_config(student_cfg_path)
    student_model_cfg = student_config["model"]
    student_data_cfg = student_config["data"]

    # Vocab
    vocab_path = student_data_cfg["vocab_path"]
    if not Path(vocab_path).exists():
        logger.error(f"Vocab 文件不存在: {vocab_path}")
        logger.error("请先运行 TextCNN/BiLSTM 基线训练生成 vocab。")
        sys.exit(1)

    vocab = CharVocab.load(vocab_path)
    logger.info(f"Vocab 大小: {len(vocab)}")

    # 数据
    train_file = config["data"]["train_file"]
    dev_file = config["data"]["dev_file"]
    max_len = student_data_cfg["max_len"]
    batch_size = config["training"]["batch_size"]

    # 蒸馏训练集
    if not Path(args.teacher_logits_file).exists():
        logger.error(f"教师 logits 文件不存在: {args.teacher_logits_file}")
        sys.exit(1)

    train_ds = DistillCharDataset(
        train_file=train_file,
        teacher_logits_file=args.teacher_logits_file,
        vocab=vocab,
        max_len=max_len,
        max_samples=args.max_train_samples,
    )

    # Dev 集（普通 CharDataset，不需要 teacher logits）
    dev_ds = CharDataset(dev_file, vocab, max_len, max_samples=args.max_dev_samples)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)

    logger.info(f"蒸馏训练集: {len(train_ds)} 条 | Dev: {len(dev_ds)} 条")

    # 学生模型
    if args.student_model == "textcnn":
        model = TextCNN(
            vocab_size=len(vocab),
            embed_dim=student_model_cfg["embed_dim"],
            num_filters=student_model_cfg["num_filters"],
            kernel_sizes=student_model_cfg["kernel_sizes"],
            dropout=student_model_cfg["dropout"],
            num_classes=student_model_cfg["num_classes"],
        )
    elif args.student_model == "bilstm":
        model = BiLSTM(
            vocab_size=len(vocab),
            embed_dim=student_model_cfg["embed_dim"],
            hidden_dim=student_model_cfg["hidden_dim"],
            num_layers=student_model_cfg["num_layers"],
            dropout=student_model_cfg["dropout"],
            num_classes=student_model_cfg["num_classes"],
            bidirectional=student_model_cfg.get("bidirectional", True),
        )
    else:
        raise ValueError(f"不支持的学生模型: {args.student_model}")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"学生模型: {args.student_model} | 参数量: {n_params:,}")

    # 蒸馏训练
    trainer = DistillTrainer(model, train_loader, dev_loader, config, args.exp_name)
    best_ckpt = trainer.train()
    logger.info(f"蒸馏训练完成！Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
