# -*- coding: utf-8 -*-
"""大模型辅助困难样本生成器（增强解析容错）"""
import json
import re
from typing import List, Dict, Tuple

from src.augmentation.prompt_builder import build_prompt
from src.augmentation.llm_client import LLMClient


def parse_llm_response(response: str) -> str:
    """解析 LLM 返回的 JSON 或半结构化文本，提取 aug_text。

    支持：
    - 标准 JSON: {"aug_text": "..."}
    - 带 ```json 代码块
    - JSON 数组: [{"aug_text": "..."}]
    - 纯文本（去除引号后返回）

    Returns:
        提取的增强文本，解析失败返回 None
    """
    if not response or not response.strip():
        return None

    text = response.strip()

    # 去除 markdown 代码块标记
    if "```json" in text:
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    elif "```" in text:
        match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    # 尝试直接解析 JSON object
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "aug_text" in data:
            return data["aug_text"]
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict) and "aug_text" in data[0]:
                return data[0]["aug_text"]
    except json.JSONDecodeError:
        pass

    # 尝试从文本中提取 JSON 片段（第一个 { 到最后一个 }）
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end])
            if isinstance(data, dict) and "aug_text" in data:
                return data["aug_text"]
        except json.JSONDecodeError:
            pass

    # 尝试从 [ 到 ] 提取 JSON 数组
    arr_start = text.find("[")
    arr_end = text.rfind("]") + 1
    if arr_start >= 0 and arr_end > arr_start:
        try:
            data = json.loads(text[arr_start:arr_end])
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], dict) and "aug_text" in data[0]:
                    return data[0]["aug_text"]
        except json.JSONDecodeError:
            pass

    # 如果不是 JSON，直接返回整段文本（去除引号，长度合理时）
    cleaned = text.strip().strip('"').strip("'")
    if 4 < len(cleaned) < 500:
        return cleaned

    return None


def generate_with_llm(
    samples: List[Dict],
    llm_client: LLMClient,
    target_per_type: int = 20,
) -> Tuple[List[Dict], List[Dict]]:
    """使用大模型生成增强样本。

    Args:
        samples: 输入样本列表，每条包含 id, text, label, hard_type
        llm_client: LLM 客户端实例
        target_per_type: 每种类型目标生成数量

    Returns:
        (成功样本列表, 失败样本列表)
    """
    # 按类型分组
    type_groups = {}
    for s in samples:
        ht = s.get("hard_type", "other")
        if ht not in type_groups:
            type_groups[ht] = []
        type_groups[ht].append(s)

    results = []
    failures = []
    gen_method = "llm_dry_run" if llm_client.dry_run else "llm_api"

    for hard_type, group in type_groups.items():
        count = 0
        for sample in group:
            if count >= target_per_type:
                break

            # 构造 prompt
            prompt = build_prompt(
                text=sample["text"],
                label=sample["label"],
                hard_type=hard_type,
            )

            # 调用 LLM
            raw_response = ""
            try:
                raw_response = llm_client.generate(prompt)
            except Exception as e:
                failures.append({
                    "source_id": sample["id"],
                    "label": sample["label"],
                    "hard_type": hard_type,
                    "error_type": type(e).__name__,
                    "error": str(e)[:500],
                    "raw_response_preview": "",
                    "prompt_preview": prompt[:1000],
                })
                continue

            # 解析响应
            aug_text = parse_llm_response(raw_response)
            if aug_text is None:
                failures.append({
                    "source_id": sample["id"],
                    "label": sample["label"],
                    "hard_type": hard_type,
                    "error_type": "ParseError",
                    "error": "Failed to parse aug_text from response",
                    "raw_response_preview": raw_response[:2000],
                    "prompt_preview": prompt[:1000],
                })
                continue

            results.append({
                "source_id": sample["id"],
                "source_text": sample["text"],
                "aug_text": aug_text,
                "label": sample["label"],
                "hard_type": hard_type,
                "gen_method": gen_method,
            })
            count += 1

    return results, failures
