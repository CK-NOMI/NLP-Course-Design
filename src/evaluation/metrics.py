"""统一评价指标模块"""
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def compute_metrics(
    y_true,
    y_pred,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """计算二分类评价指标。

    Args:
        y_true: 真实标签 (list 或 ndarray)
        y_pred: 预测标签 (list 或 ndarray)
        y_prob: 可选，预测概率 shape=(n, 2)

    Returns:
        dict: accuracy, precision, recall, macro_f1, confusion_matrix
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    results = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "macro_f1": round(f1, 4),
        "confusion_matrix": cm,
    }
    return results
