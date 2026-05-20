"""合并原始训练集和增强样本，构造增强训练集"""
import pandas as pd
from pathlib import Path

from src.utils.io_utils import ensure_dir


def build_augmented_train(train_file: str, aug_file: str, output_file: str):
    """合并原始训练集和过滤后的增强样本。

    Args:
        train_file: 原始训练集路径 (id, text, label)
        aug_file: 过滤后增强样本路径 (含 aug_text, label, quality_pass 等)
        output_file: 输出路径 (id, text, label)

    Returns:
        合并后的 DataFrame
    """
    # 读取原始训练集
    train_df = pd.read_csv(train_file)
    original_count = len(train_df)

    # 读取增强样本（只保留通过质量过滤的）
    aug_df = pd.read_csv(aug_file)
    if "quality_pass" in aug_df.columns:
        aug_df = aug_df[aug_df["quality_pass"] == True].copy()

    # 构造增强样本的标准格式
    aug_records = []
    for i, row in aug_df.iterrows():
        aug_records.append({
            "id": f"aug_{i}",
            "text": row["aug_text"],
            "label": int(row["label"]),
        })

    aug_standard_df = pd.DataFrame(aug_records)
    aug_count = len(aug_standard_df)

    # 合并
    combined_df = pd.concat([train_df[["id", "text", "label"]], aug_standard_df], ignore_index=True)

    # 保存
    ensure_dir(str(Path(output_file).parent))
    combined_df.to_csv(output_file, index=False)

    return {
        "original_count": original_count,
        "augmented_count": aug_count,
        "total_count": len(combined_df),
        "output_file": output_file,
    }
