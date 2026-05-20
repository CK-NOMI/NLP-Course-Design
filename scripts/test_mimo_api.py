#!/usr/bin/env python
"""MiMo API 连通性测试"""
import os
from openai import OpenAI

api_key = os.environ.get("OPENAI_API_KEY")
base_url = os.environ.get("OPENAI_BASE_URL")

if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")
if not base_url:
    raise RuntimeError("OPENAI_BASE_URL is not set")

client = OpenAI(api_key=api_key, base_url=base_url)

print("Base URL:", base_url)
print("API key loaded:", bool(api_key))

# 尝试 models.list()
try:
    models = client.models.list()
    print("models.list() success")
    for m in models.data:
        print("  model:", m.id)
except Exception as e:
    print("models.list() failed, will try chat completion")
    print("  error type:", type(e).__name__)
    print("  error:", str(e)[:300])

# 尝试 chat completion
candidates = ["mimo-v2.5-pro", "mimo-v2.5"]
last_error = None

for model in candidates:
    try:
        print(f"\nTrying model: {model}")
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "请用一句话回复：API连通成功。"}],
            temperature=0.2,
            max_tokens=128,
        )
        print("chat.completions.create() success")
        print("used model:", model)
        print("reply:", resp.choices[0].message.content)
        break
    except Exception as e:
        last_error = e
        print(f"  failed model: {model}")
        print(f"  error type: {type(e).__name__}")
        print(f"  error: {str(e)[:300]}")
else:
    raise RuntimeError(f"All candidate models failed: {last_error}")
