"""误判分析与困难样本类型识别模块"""
import re
from typing import Dict, List

import json
import pandas as pd
from pathlib import Path

from src.utils.io_utils import ensure_dir


# === 困难类型关键词规则（收紧版） ===

# 否定类：多字否定短语优先，单字"不"需要后接特定词
NEGATION_PHRASES = [
    "没有", "不是", "并不", "不太", "不怎么", "算不上", "谈不上",
    "不满意", "不推荐", "不值得", "不好", "不行", "不舒服", "不喜欢",
    "不会再", "再也不", "没意思", "没感觉", "没达到", "没想到",
    "不想再",
]

# 转折类：明确转折词和短语
TRANSITION_PHRASES = [
    "但是", "不过", "然而", "可是", "只是", "虽然", "尽管",
    "倒是", "问题是", "遗憾的是", "美中不足",
]
# 转折类正则：单字"但"后接明显转折结构，"却"作为独立转折
TRANSITION_PATTERNS = [
    re.compile(r"但[，,\s]"),
    re.compile(r"但还是"),
    re.compile(r"但总体"),
    re.compile(r"但价格"),
    re.compile(r"但服务"),
    re.compile(r"但质量"),
    re.compile(r"但环境"),
    re.compile(r"却"),
]

# 程度变化类：较强程度词和"程度词+负面词"组合
DEGREE_PHRASES = [
    "非常", "特别", "极其", "十分", "相当", "超级",
    "巨", "贼", "很不", "太不",
    "有点失望", "有点差", "有点贵", "有点慢",
    "稍微差", "略微失望",
]
# 单独"太"需要后接形容词才算，用正则
DEGREE_PATTERNS = [
    re.compile(r"太[好差贵慢脏吵小远近冷热难]"),
    re.compile(r"太不"),
]

# 口语扰动类
COLLOQUIAL_PHRASES = [
    "哈哈", "呵呵", "emmm", "无语", "踩雷", "避雷",
    "一般般", "不咋样", "还行吧", "就那样", "离谱",
]
COLLOQUIAL_PATTERNS = [
    re.compile(r"[a-zA-Z]{3,}"),       # 3个以上连续英文
    re.compile(r"哈哈+"),              # 重复哈
    re.compile(r"。。。+"),            # 多个句号
    re.compile(r"！！+"),              # 多个感叹号
    re.compile(r"666"),
    re.compile(r"yyds"),
    re.compile(r"啥"),
    re.compile(r"咋"),
]

# 隐含主观类
IMPLICIT_PHRASES = [
    "再也不会", "下次不会", "以后不会", "值得", "浪费", "白来",
    "回购", "推荐", "避雷", "踩雷", "物有所值", "性价比", "名不副实",
]


def _match_phrases(text: str, phrases: List[str]) -> List[str]:
    """返回文本中命中的短语列表。"""
    hits = []
    for phrase in phrases:
        if phrase in text:
            hits.append(phrase)
    return hits


def _match_patterns(text: str, patterns: list) -> List[str]:
    """返回文本中命中的正则模式。"""
    hits = []
    for pat in patterns:
        match = pat.search(text)
        if match:
            hits.append(match.group())
    return hits


def classify_hard_type(text: str) -> Dict:
    """对一条中文文本进行困难类型判断（收紧版规则）。

    优先级：transition > negation > degree > colloquial > implicit > other

    Returns:
        {"hard_type": str, "rule_hits": list}
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {"hard_type": "other", "rule_hits": []}

    # 转折类
    transition_hits = _match_phrases(text, TRANSITION_PHRASES)
    transition_pattern_hits = _match_patterns(text, TRANSITION_PATTERNS)
    all_transition = transition_hits + transition_pattern_hits
    if all_transition:
        return {"hard_type": "transition", "rule_hits": all_transition}

    # 否定类
    negation_hits = _match_phrases(text, NEGATION_PHRASES)
    if negation_hits:
        return {"hard_type": "negation", "rule_hits": negation_hits}

    # 程度变化类
    degree_hits = _match_phrases(text, DEGREE_PHRASES)
    degree_pattern_hits = _match_patterns(text, DEGREE_PATTERNS)
    all_degree = degree_hits + degree_pattern_hits
    if all_degree:
        return {"hard_type": "degree", "rule_hits": all_degree}

    # 口语扰动类
    colloquial_hits = _match_phrases(text, COLLOQUIAL_PHRASES)
    colloquial_pattern_hits = _match_patterns(text, COLLOQUIAL_PATTERNS)
    all_colloquial = colloquial_hits + colloquial_pattern_hits
    if all_colloquial:
        return {"hard_type": "colloquial", "rule_hits": all_colloquial}

    # 隐含主观类
    implicit_hits = _match_phrases(text, IMPLICIT_PHRASES)
    if implicit_hits:
        return {"hard_type": "implicit", "rule_hits": implicit_hits}

    return {"hard_type": "other", "rule_hits": []}


def analyze_prediction_file(
    pred_file: str,
    output_dir: str,
    model_name: str = None,
):
    """分析预测文件中的误判样本，输出误判明细和类型统计。

    Args:
        pred_file: 预测 CSV 文件路径
        output_dir: 输出目录
        model_name: 模型名称（用于输出文件命名）
    """
    pred_path = Path(pred_file)
    if not pred_path.exists():
        raise FileNotFoundError(f"预测文件不存在: {pred_file}")

    if model_name is None:
        model_name = pred_path.stem

    df = pd.read_csv(pred_file)

    # 校验字段
    required_cols = ["id", "text", "label", "pred", "prob_neg", "prob_pos", "is_correct"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"预测文件缺少字段: {missing}")

    total_samples = len(df)
    error_df = df[df["is_correct"] == False].copy()
    error_count = len(error_df)
    accuracy = round(1 - error_count / max(total_samples, 1), 4)
    error_rate = round(error_count / max(total_samples, 1), 4)

    # 对每条误判样本分类
    hard_types = []
    rule_hits_list = []
    for _, row in error_df.iterrows():
        result = classify_hard_type(str(row["text"]) if pd.notna(row["text"]) else "")
        hard_types.append(result["hard_type"])
        rule_hits_list.append("|".join(result["rule_hits"]))

    error_df["hard_type"] = hard_types
    error_df["rule_hits"] = rule_hits_list

    # 类型统计
    type_counts = error_df["hard_type"].value_counts()
    type_stats = pd.DataFrame({
        "hard_type": type_counts.index,
        "count": type_counts.values,
        "ratio": (type_counts.values / max(error_count, 1)).round(4),
    })

    # 输出
    ensure_dir(output_dir)

    error_samples_path = Path(output_dir) / f"{model_name}_error_samples.csv"
    error_df.to_csv(error_samples_path, index=False)

    type_stats_path = Path(output_dir) / f"{model_name}_error_type_stats.csv"
    type_stats.to_csv(type_stats_path, index=False)

    # summary JSON
    type_distribution = {row["hard_type"]: int(row["count"]) for _, row in type_stats.iterrows()}
    summary = {
        "model_name": model_name,
        "total_samples": total_samples,
        "error_samples": error_count,
        "accuracy": accuracy,
        "error_rate": error_rate,
        "type_distribution": type_distribution,
    }
    summary_path = Path(output_dir) / f"{model_name}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "error_samples_path": str(error_samples_path),
        "type_stats_path": str(type_stats_path),
        "summary_path": str(summary_path),
        "summary": summary,
    }
