# -*- coding: utf-8 -*-
"""大模型 API 客户端（支持 dry-run 模式，增加 timeout 和重试）"""
import os
import json


class LLMClient:
    """OpenAI-compatible Chat Completions 客户端。

    dry_run=True 时不调用真实 API，返回模拟结果。
    """

    def __init__(
        self,
        model: str = "mimo-v2.5-pro",
        api_key_env: str = "OPENAI_API_KEY",
        base_url_env: str = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        dry_run: bool = True,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.dry_run = dry_run

        # 从环境变量读取 API Key（不打印）
        self.api_key = os.environ.get(api_key_env, "")
        self.base_url = None
        if base_url_env:
            self.base_url = os.environ.get(base_url_env, None)

        if not self.dry_run and not self.api_key:
            raise ValueError(
                f"未设置环境变量 {api_key_env}。"
                f"请设置 API Key 或使用 dry_run=True 模式。"
            )

    def generate(self, prompt: str) -> str:
        """调用大模型生成文本。

        Args:
            prompt: 输入 prompt

        Returns:
            模型返回的文本内容

        Raises:
            ValueError: 如果 API 返回空内容
        """
        if self.dry_run:
            return self._dry_run_response(prompt)

        return self._call_api(prompt)

    def _dry_run_response(self, prompt: str) -> str:
        """dry-run 模式：返回模拟 JSON 响应。"""
        if "原文：" in prompt:
            start = prompt.index("原文：") + 3
            end_pos = prompt.find("\n", start)
            if end_pos == -1:
                end_pos = start + 50
            original_snippet = prompt[start:end_pos].strip()[:20]
        else:
            original_snippet = "模拟文本"

        mock_text = f"[dry-run] 改写：{original_snippet}...这是模拟增强样本"
        return json.dumps({"aug_text": mock_text}, ensure_ascii=False)

    def _call_api(self, prompt: str) -> str:
        """真实调用 OpenAI-compatible API（带 timeout 和重试）。"""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装 openai 包：pip install openai>=1.0.0")

        client_kwargs = {
            "api_key": self.api_key,
            "timeout": 60.0,
            "max_retries": 2,
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        client = OpenAI(**client_kwargs)

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=60.0,
        )

        # 提取 content
        choice = response.choices[0] if response.choices else None
        if choice is None:
            raw_preview = str(response)[:2000]
            raise ValueError(
                f"API returned no choices. raw_response_preview: {raw_preview}"
            )

        content = choice.message.content
        finish_reason = choice.finish_reason

        if not content or not content.strip():
            raw_preview = str(response)[:2000]
            raise ValueError(
                f"API returned empty content. "
                f"finish_reason={finish_reason}. "
                f"raw_response_preview: {raw_preview}"
            )

        return content
