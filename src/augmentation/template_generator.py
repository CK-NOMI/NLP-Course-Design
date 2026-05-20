"""基于规则模板的困难样本生成器（不调用大模型 API，仅用于代码闭环测试）"""
import random
from typing import List, Dict


# 各类型的改写模板规则
NEGATION_PREFIXES = ["并不觉得", "不太认为", "没感觉到"]
NEGATION_SUFFIXES = ["，不是很满意", "，没有想象中好", "，不太推荐"]

TRANSITION_INSERTS = [
    "虽然有些地方不错，但是",
    "环境还行，不过",
    "整体一般，然而",
]

DEGREE_MODIFIERS = ["非常", "特别", "有点", "稍微", "极其"]

COLLOQUIAL_SUFFIXES = ["哈哈", "emmm", "吧", "啊", "呢", "无语"]

IMPLICIT_REWRITES = [
    "就那样吧，没什么好说的",
    "也不是不行，就是一般",
    "怎么说呢，看个人吧",
]


def generate_by_template(text: str, label: int, hard_type: str, seed: int = None) -> str:
    """用规则模板对文本进行简单改写。

    Args:
        text: 原始文本
        label: 情感标签
        hard_type: 困难类型
        seed: 随机种子

    Returns:
        改写后的文本
    """
    if seed is not None:
        random.seed(seed)

    if hard_type == "negation":
        choice = random.choice(["prefix", "suffix"])
        if choice == "prefix":
            prefix = random.choice(NEGATION_PREFIXES)
            return prefix + text[:20] if len(text) > 20 else prefix + text
        else:
            suffix = random.choice(NEGATION_SUFFIXES)
            return text + suffix

    elif hard_type == "transition":
        insert = random.choice(TRANSITION_INSERTS)
        # 在文本前加转折
        return insert + text

    elif hard_type == "degree":
        modifier = random.choice(DEGREE_MODIFIERS)
        # 在文本开头加程度词
        return modifier + text

    elif hard_type == "colloquial":
        suffix = random.choice(COLLOQUIAL_SUFFIXES)
        return text + suffix

    elif hard_type == "implicit":
        # 用隐含表达替换或追加
        rewrite = random.choice(IMPLICIT_REWRITES)
        if len(text) > 30:
            return text[:15] + "，" + rewrite
        else:
            return rewrite

    else:
        # other 类型：简单复制加后缀
        return text + "，总体来说一般"


def generate_batch(
    samples: List[Dict],
    target_per_type: int = 20,
    seed: int = 42,
) -> List[Dict]:
    """批量生成增强样本。

    Args:
        samples: 输入样本列表，每条包含 id, text, label, hard_type
        target_per_type: 每种类型目标生成数量
        seed: 随机种子

    Returns:
        增强样本列表
    """
    random.seed(seed)

    # 按类型分组
    type_groups = {}
    for s in samples:
        ht = s.get("hard_type", "other")
        if ht not in type_groups:
            type_groups[ht] = []
        type_groups[ht].append(s)

    results = []
    for hard_type, group in type_groups.items():
        count = 0
        for i, sample in enumerate(group):
            if count >= target_per_type:
                break
            aug_text = generate_by_template(
                text=sample["text"],
                label=sample["label"],
                hard_type=hard_type,
                seed=seed + i,
            )
            results.append({
                "source_id": sample["id"],
                "source_text": sample["text"],
                "aug_text": aug_text,
                "label": sample["label"],
                "hard_type": hard_type,
                "gen_method": "template",
            })
            count += 1

    return results
