#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""大模型辅助困难样本增强脚本（带进度日志和失败记录）"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from src.utils.io_utils import ensure_dir
from src.augmentation.llm_client import LLMClient
from src.augmentation.llm_generator import generate_with_llm
from src.augmentation.prompt_builder import build_prompt


def log(msg):
    """带 flush 的日志输出"""
    print(f"[generate] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="大模型辅助困难样本增强")
    parser.add_argument("--config", type=str, required=True, help="增强配置文件")
    parser.add_argument("--input_file", type=str, required=True, help="困难样本输入文件")
    parser.add_argument("--output_file", type=str, required=True, help="增强样本输出文件")
    parser.add_argument("--target_per_type", type=int, default=None, help="每类生成数量")
    parser.add_argument("--dry_run", action="store_true", default=False,
                        help="dry-run 模式（不调用真实 API）")
    args = parser.parse_args()

    # 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    llm_cfg = config.get("llm", {})
    target_per_type = args.target_per_type or config.get("target_per_type", 20)

    # dry_run 优先级：命令行 > 配置文件
    dry_run = args.dry_run or llm_cfg.get("dry_run", True)

    # 检查输入文件
    if not Path(args.input_file).exists():
        log(f"ERROR: 输入文件不存在: {args.input_file}")
        sys.exit(1)

    log(f"读取困难样本: {args.input_file}")
    df = pd.read_csv(args.input_file)
    log(f"输入样本数: {len(df)}")
    log(f"dry_run: {dry_run}")
    log(f"model: {llm_cfg.get('model', 'unknown')}")
    log(f"每类目标生成数量: {target_per_type}")

    # 转为字典列表
    samples = []
    for _, row in df.iterrows():
        samples.append({
            "id": row["id"],
            "text": str(row["text"]) if pd.notna(row["text"]) else "",
            "label": int(row["label"]),
            "hard_type": row.get("hard_type", "other"),
        })

    # 创建 LLM 客户端
    llm_client = LLMClient(
        model=llm_cfg.get("model", "mimo-v2.5-pro"),
        api_key_env=llm_cfg.get("api_key_env", "OPENAI_API_KEY"),
        base_url_env=llm_cfg.get("base_url_env", None),
        temperature=llm_cfg.get("temperature", 0.7),
        max_tokens=llm_cfg.get("max_tokens", 512),
        dry_run=dry_run,
    )

    # 按类型分组生成（带进度日志）
    type_groups = {}
    for s in samples:
        ht = s.get("hard_type", "other")
        if ht not in type_groups:
            type_groups[ht] = []
        type_groups[ht].append(s)

    all_results = []
    all_failures = []
    gen_method = "llm_dry_run" if dry_run else "llm_api"

    for hard_type, group in type_groups.items():
        log(f"--- 开始生成 hard_type={hard_type} (目标={target_per_type}, 可用={len(group)}) ---")
        count = 0
        for i, sample in enumerate(group):
            if count >= target_per_type:
                break

            log(f"  [{hard_type}] {count+1}/{target_per_type} 调用 API...")

            prompt = build_prompt(
                text=sample["text"],
                label=sample["label"],
                hard_type=hard_type,
            )

            raw_response = ""
            try:
                raw_response = llm_client.generate(prompt)
                log(f"  [{hard_type}] {count+1}/{target_per_type} API 返回 {len(raw_response)} 字符")
            except Exception as e:
                log(f"  [{hard_type}] {count+1}/{target_per_type} API 失败: {type(e).__name__}: {str(e)[:200]}")
                all_failures.append({
                    "source_id": sample["id"],
                    "label": sample["label"],
                    "hard_type": hard_type,
                    "error_type": type(e).__name__,
                    "error": str(e)[:500],
                    "raw_response_preview": "",
                    "prompt_preview": prompt[:1000],
                })
                continue

            # 解析
            from src.augmentation.llm_generator import parse_llm_response
            aug_text = parse_llm_response(raw_response)

            if aug_text is None:
                log(f"  [{hard_type}] {count+1}/{target_per_type} 解析失败, response[:100]={raw_response[:100]}")
                all_failures.append({
                    "source_id": sample["id"],
                    "label": sample["label"],
                    "hard_type": hard_type,
                    "error_type": "ParseError",
                    "error": "Failed to parse aug_text",
                    "raw_response_preview": raw_response[:2000],
                    "prompt_preview": prompt[:1000],
                })
                continue

            all_results.append({
                "source_id": sample["id"],
                "source_text": sample["text"],
                "aug_text": aug_text,
                "label": sample["label"],
                "hard_type": hard_type,
                "gen_method": gen_method,
            })
            count += 1
            log(f"  [{hard_type}] {count}/{target_per_type} 成功")

        log(f"--- {hard_type} 完成: 成功={count} ---")

    log(f"生成成功: {len(all_results)} 条")
    log(f"生成失败: {len(all_failures)} 条")

    # 统计各类型
    type_counts = {}
    for s in all_results:
        ht = s["hard_type"]
        type_counts[ht] = type_counts.get(ht, 0) + 1
    log(f"各类型数量: {type_counts}")

    # 保存成功样本
    ensure_dir(str(Path(args.output_file).parent))
    if all_results:
        out_df = pd.DataFrame(all_results)
        out_df.to_csv(args.output_file, index=False)
        log(f"输出文件: {args.output_file}")
    else:
        log("WARNING: 没有生成任何成功样本，未写入输出文件。")

    # 保存失败样本
    if all_failures:
        failed_path = Path(args.output_file).parent / "llm_real_failed_requests.csv"
        failed_df = pd.DataFrame(all_failures)
        failed_df.to_csv(failed_path, index=False)
        log(f"失败记录: {failed_path}")

    log("LLM 增强样本生成完成。")


if __name__ == "__main__":
    main()
