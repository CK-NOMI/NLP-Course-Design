"""增强样本质量过滤模块"""
from typing import List, Dict


def filter_samples(
    samples: List[Dict],
    min_text_len: int = 4,
    max_text_len: int = 256,
    valid_labels: set = None,
) -> List[Dict]:
    """对增强样本进行质量过滤。

    过滤规则：
    1. 去除空文本
    2. 去除重复 aug_text
    3. 过滤长度过短或过长
    4. 检查 label 只能是 0 或 1
    5. 过滤和原文完全相同的样本

    Args:
        samples: 增强样本列表
        min_text_len: 最短文本长度
        max_text_len: 最长文本长度
        valid_labels: 合法标签集合

    Returns:
        过滤后的样本列表（每条增加 quality_pass 和 filter_reason 字段）
    """
    if valid_labels is None:
        valid_labels = {0, 1}

    seen_texts = set()
    results = []

    for sample in samples:
        aug_text = sample.get("aug_text", "")
        source_text = sample.get("source_text", "")
        label = sample.get("label")

        # 检查空文本
        if not aug_text or not aug_text.strip():
            sample["quality_pass"] = False
            sample["filter_reason"] = "empty_text"
            results.append(sample)
            continue

        # 检查长度
        if len(aug_text.strip()) < min_text_len:
            sample["quality_pass"] = False
            sample["filter_reason"] = "too_short"
            results.append(sample)
            continue

        if len(aug_text.strip()) > max_text_len:
            sample["quality_pass"] = False
            sample["filter_reason"] = "too_long"
            results.append(sample)
            continue

        # 检查标签
        if label not in valid_labels:
            sample["quality_pass"] = False
            sample["filter_reason"] = "invalid_label"
            results.append(sample)
            continue

        # 检查和原文完全相同
        if aug_text.strip() == source_text.strip():
            sample["quality_pass"] = False
            sample["filter_reason"] = "same_as_source"
            results.append(sample)
            continue

        # 检查重复
        if aug_text.strip() in seen_texts:
            sample["quality_pass"] = False
            sample["filter_reason"] = "duplicate"
            results.append(sample)
            continue

        # 通过所有检查
        seen_texts.add(aug_text.strip())
        sample["quality_pass"] = True
        sample["filter_reason"] = ""
        results.append(sample)

    return results
