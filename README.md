# 基于大模型辅助困难样本增强与轻量蒸馏的中文短文本情感分析

## 项目简介

本项目围绕中文短文本情感分析任务展开，使用 ChnSentiCorp 数据集完成情感二分类（正向/负向）。项目核心目标是通过大模型辅助的困难样本增强与轻量蒸馏策略，提升模型在否定、转折、程度变化等复杂文本上的分类鲁棒性。

详细技术方案请参阅 [docs/实现方案.md](docs/实现方案.md)。

## 环境安装

```bash
# 推荐 Python 3.10
pip install -r requirements.txt
```

## 第一阶段：数据准备

### 下载原始数据

```bash
python scripts/download_data.py
```

运行后生成：
- `data/raw/train.csv`
- `data/raw/dev.csv`
- `data/raw/test.csv`

### 数据预处理

```bash
python scripts/preprocess.py
```

运行后生成：
- `data/processed/train.csv`
- `data/processed/dev.csv`
- `data/processed/test.csv`

每个文件包含字段：`id, text, label`

> **数据规模说明：** ChnSentiCorp 官方原始划分约为 train=9600、dev=1200、test=1200。本项目预处理阶段会进行空文本清理和重复文本删除，因此当前清洗后的实际样本数为：train=8249，dev=1178，test=1178。后续所有基线训练、增强训练和蒸馏实验均基于 `data/processed/` 下的清洗版本进行。

## 当前已完成模块

| 模块 | 路径 | 说明 |
|---|---|---|
| 目录结构 | 全项目 | 按实现方案创建完整目录 |
| 随机种子 | `src/utils/seed.py` | 固定 random/numpy/torch 种子 |
| IO 工具 | `src/utils/io_utils.py` | CSV/JSONL 读写 + 目录创建 |
| 日志工具 | `src/utils/logger.py` | 控制台+文件双输出 |
| 评价指标 | `src/evaluation/metrics.py` | Acc/P/R/F1/混淆矩阵 |
| 数据下载 | `scripts/download_data.py` | 从 HuggingFace 下载 ChnSentiCorp |
| 数据预处理 | `scripts/preprocess.py` | 清洗、去重、格式统一 |
| 基础配置 | `configs/base.yaml` | seed/路径/超参数默认值 |

## 项目目录

```
NLP-Course-Design/
├── configs/          # 配置文件
├── data/             # 数据（不提交到 git）
├── scripts/          # 可执行脚本
├── src/              # 核心源码
├── outputs/          # 实验输出
├── docs/             # 文档
├── requirements.txt
└── README.md
```
