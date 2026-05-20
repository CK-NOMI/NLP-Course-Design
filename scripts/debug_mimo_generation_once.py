#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单次 API 调用诊断：检查 mimo 模型返回内容"""
import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from openai import OpenAI
from src.augmentation.prompt_builder import build_prompt
from src.augmentation.llm_generator import parse_llm_response
from src.utils.io_utils import ensure_dir


def main():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "")

    if not api_key:
        # 尝试从 API_key.txt 读取
        key_file = Path("API_key.txt")
        if key_file.exists():
            api_key = key_file.read_text(encoding="utf-8").strip()
        else:
            print("ERROR: OPENAI_API_KEY not set and API_key.txt not found", flush=True)
            sys.exit(1)

    if not base_url:
        base_url = "https://token-plan-cn.xiaomimimo.com/v1"

    model = "mimo-v2.5-pro"
    print(f"Base URL: {base_url}", flush=True)
    print(f"Model: {model}", flush=True)
    print(f"API key loaded: {bool(api_key)}", flush=True)

    # 读取第一条困难样本
    hard_file = Path("data/hard_eval/hard_test.csv")
    df = pd.read_csv(hard_file)
    sample = df.iloc[0]
    text = str(sample["text"])
    label = int(sample["label"])
    hard_type = sample["hard_type"]

    print(f"\nSample: id={sample['id']}, hard_type={hard_type}, label={label}", flush=True)
    print(f"Text: {text[:100]}...", flush=True)

    # 构造 prompt
    prompt = build_prompt(text, label, hard_type)
    print(f"\nPrompt length: {len(prompt)}", flush=True)
    print(f"Prompt preview:\n{prompt[:500]}", flush=True)

    # 调用 API
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0, max_retries=2)

    print(f"\nCalling API...", flush=True)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=512,
            timeout=60.0,
        )
    except Exception as e:
        print(f"API call failed: {type(e).__name__}: {e}", flush=True)
        sys.exit(1)

    # 保存完整 response
    ensure_dir("data/augmented")
    debug_path = Path("data/augmented/llm_real_debug_response.json")
    try:
        raw_json = response.model_dump_json(indent=2)
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(raw_json)
        print(f"\nFull response saved to: {debug_path}", flush=True)
    except Exception as e:
        print(f"Failed to save response: {e}", flush=True)

    # 分析
    choice = response.choices[0] if response.choices else None
    if choice is None:
        print("ERROR: No choices in response", flush=True)
        sys.exit(1)

    content = choice.message.content
    finish_reason = choice.finish_reason

    print(f"\nfinish_reason: {finish_reason}", flush=True)
    print(f"content is None: {content is None}", flush=True)
    print(f"content is empty: {content == '' if content is not None else 'N/A'}", flush=True)

    if content:
        print(f"content length: {len(content)}", flush=True)
        print(f"content preview (first 500 chars):\n{content[:500]}", flush=True)

        # 尝试解析
        aug_text = parse_llm_response(content)
        print(f"\nparse_llm_response result: {aug_text}", flush=True)
        print(f"Parse success: {aug_text is not None}", flush=True)
    else:
        print("ERROR: content is None or empty!", flush=True)
        print(f"Full response object: {str(response)[:1000]}", flush=True)


if __name__ == "__main__":
    main()
