"""通用分类训练器"""
import json
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from src.evaluation.metrics import compute_metrics
from src.utils.io_utils import ensure_dir
from src.utils.logger import get_logger


class Trainer:
    """通用分类训练器，支持 early stopping 和 best checkpoint 保存。"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        config: Dict,
        exp_name: str,
    ):
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.config = config
        self.exp_name = exp_name

        # 训练参数
        train_cfg = config["training"]
        self.epochs = train_cfg["epochs"]
        self.lr = train_cfg["lr"]
        self.weight_decay = train_cfg.get("weight_decay", 0.0)
        self.patience = train_cfg.get("patience", 3)

        # 输出路径
        out_cfg = config["output"]
        self.ckpt_dir = Path(out_cfg["checkpoint_dir"]) / exp_name
        self.log_dir = Path(out_cfg["log_dir"]) / exp_name
        ensure_dir(str(self.ckpt_dir))
        ensure_dir(str(self.log_dir))

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 优化器和损失
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

        # 日志
        self.logger = get_logger(
            f"trainer.{exp_name}",
            log_file=str(self.log_dir / "training.log"),
        )

        # 保存 config 快照
        with open(self.log_dir / "config.yaml", "w") as f:
            yaml.dump(config, f, allow_unicode=True)

    def train(self) -> str:
        """执行训练，返回 best checkpoint 路径。"""
        best_f1 = -1.0
        patience_counter = 0
        log_file = self.log_dir / "train_log.jsonl"
        best_ckpt_path = str(self.ckpt_dir / "best.pt")

        self.logger.info(f"开始训练 | device={self.device} | epochs={self.epochs}")
        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            # === Train ===
            train_loss = self._train_epoch()

            # === Eval ===
            dev_loss, dev_metrics = self._eval_epoch()

            dev_acc = dev_metrics["accuracy"]
            dev_f1 = dev_metrics["macro_f1"]

            # 记录日志
            log_entry = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "dev_loss": round(dev_loss, 4),
                "dev_acc": dev_acc,
                "dev_f1": dev_f1,
            }
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

            self.logger.info(
                f"Epoch {epoch}/{self.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"dev_loss={dev_loss:.4f} | "
                f"dev_acc={dev_acc:.4f} | "
                f"dev_f1={dev_f1:.4f}"
            )

            # === Early stopping / Save best ===
            if dev_f1 > best_f1:
                best_f1 = dev_f1
                patience_counter = 0
                torch.save(self.model.state_dict(), best_ckpt_path)
                self.logger.info(f"  -> 保存 best checkpoint (f1={best_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break

        elapsed = time.time() - start_time
        self.logger.info(f"训练完成 | 耗时={elapsed:.1f}s | best_dev_f1={best_f1:.4f}")
        self.logger.info(f"Best checkpoint: {best_ckpt_path}")

        return best_ckpt_path

    def _forward_batch(self, batch):
        """统一前向传播，兼容有无 attention_mask 的模型。"""
        input_ids = batch["input_ids"].to(self.device)
        if "attention_mask" in batch:
            attention_mask = batch["attention_mask"].to(self.device)
            logits = self.model(input_ids, attention_mask=attention_mask)
        else:
            logits = self.model(input_ids)
        return logits

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in self.train_loader:
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            logits = self._forward_batch(batch)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _eval_epoch(self):
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.dev_loader:
                labels = batch["label"].to(self.device)

                logits = self._forward_batch(batch)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                n_batches += 1

                preds = logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / max(n_batches, 1)
        metrics = compute_metrics(all_labels, all_preds)
        return avg_loss, metrics
