# -*- coding: utf-8 -*-
"""困难样本增强 Prompt 构造模块（强化 JSON 输出格式要求）"""


LABEL_MAP = {0: "负向", 1: "正向"}

JSON_FORMAT_INSTRUCTION = (
    '\n\n严格要求：\n'
    '- 只输出一个JSON对象，不要输出任何其他内容\n'
    '- 不要输出解释、不要输出markdown代码块\n'
    '- 格式必须是: {"aug_text": "改写后的中文文本"}\n'
    '- aug_text必须是自然流畅的中文'
)


def _negation_prompt(text, label_text):
    return (
        '请对以下中文评论进行改写，加入否定表达使语义更复杂。\n'
        '要求：改写后情感极性保持为【' + label_text + '】，长度不超过原文1.5倍，保持自然中文。\n\n'
        '原文：' + text + '\n'
        '情感标签：' + label_text
        + JSON_FORMAT_INSTRUCTION
    )


def _transition_prompt(text, label_text):
    return (
        '请对以下中文评论进行改写，加入转折结构使前后形成对比。\n'
        '要求：改写后最终情感极性保持为【' + label_text + '】，转折后体现最终情感，保持自然中文。\n\n'
        '原文：' + text + '\n'
        '情感标签：' + label_text
        + JSON_FORMAT_INSTRUCTION
    )


def _degree_prompt(text, label_text):
    return (
        '请对以下中文评论进行改写，调整程度表达使判断难度增加。\n'
        '要求：改写后情感极性保持为【' + label_text + '】，通过程度词变化增加模糊性，保持自然中文。\n\n'
        '原文：' + text + '\n'
        '情感标签：' + label_text
        + JSON_FORMAT_INSTRUCTION
    )


def _implicit_prompt(text, label_text):
    return (
        '请对以下中文评论进行改写，去掉明显情感词，用隐晦含蓄方式表达相同态度。\n'
        '要求：改写后情感极性保持为【' + label_text + '】，可用反问、省略、暗示等手法，保持自然中文。\n\n'
        '原文：' + text + '\n'
        '情感标签：' + label_text
        + JSON_FORMAT_INSTRUCTION
    )


def _colloquial_prompt(text, label_text):
    return (
        '请对以下中文评论进行改写，改为更口语化、网络化的表达风格。\n'
        '要求：改写后情感极性保持为【' + label_text + '】，可用语气词、网络用语，保持自然中文。\n\n'
        '原文：' + text + '\n'
        '情感标签：' + label_text
        + JSON_FORMAT_INSTRUCTION
    )


PROMPT_BUILDERS = {
    "negation": _negation_prompt,
    "transition": _transition_prompt,
    "degree": _degree_prompt,
    "implicit": _implicit_prompt,
    "colloquial": _colloquial_prompt,
}


def build_prompt(text, label, hard_type):
    """根据困难类型构造增强 prompt。

    Args:
        text: 原始文本
        label: 情感标签 (0 或 1)
        hard_type: 困难类型

    Returns:
        prompt 字符串
    """
    label_text = LABEL_MAP.get(label, "未知")
    builder = PROMPT_BUILDERS.get(hard_type, _negation_prompt)
    return builder(text, label_text)
