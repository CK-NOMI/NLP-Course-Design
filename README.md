# 基于大模型辅助困难样本增强与轻量蒸馏的中文短文本情感分析

## 项目简介

本项目围绕中文短文本情感分析任务展开，使用 ChnSentiCorp 数据集完成情感二分类（正向/负向）。项目核心目标是通过大模型辅助的困难样本增强与轻量蒸馏策略，提升模型在否定、转折、程度变化等复杂文本上的分类鲁棒性。

详细技术方案请参阅 [docs/实现方案.md](docs/实现方案.md)。

## 核心方法

1. **四模型基线**：TextCNN / BiLSTM / BERT / MacBERT 在统一数据划分下训练和评估
2. **困难样本构建**：从 test 集中用规则初筛 300 条困难样本（否定/转折/程度/口语/隐含主观）
3. **模板增强**：本地规则模板生成 24 条增强样本（代码闭环验证）
4. **真实 LLM API 增强**：调用小米 MiMo API（mimo-v2.5-pro）生成 59 条高质量增强样本
5. **知识蒸馏**：MacBERT 作为教师模型，TextCNN 作为学生模型，CE + KL 联合损失训练
6. **组合实验**：真实 LLM 增强 + MacBERT 蒸馏联合训练

## 最终核心结果

### 四模型正式基线

| Model | Test Acc | Test F1 | Hard Acc | Hard F1 |
|---|---|---|---|---|
| TextCNN | 0.916 | 0.916 | 0.9 | 0.8988 |
| BiLSTM | 0.8888 | 0.8887 | 0.8833 | 0.8813 |
| BERT | 0.9015 | 0.9015 | 0.8867 | 0.8848 |
| MacBERT | 0.8922 | 0.8921 | 0.9 | 0.8992 |

### TextCNN 增强与蒸馏消融实验

| Setting | Test F1 | Hard F1 | Hard Improvement |
|---|---|---|---|
| baseline | 0.916 | 0.8988 | — |
| template_aug | 0.9168 | 0.9089 | +1.01pp |
| distill_formal | 0.915 | 0.9083 | +0.95pp |
| template_aug+distill | 0.9057 | 0.9114 | +1.26pp |
| llm_real_aug | **0.9244** | 0.9155 | +1.67pp |
| llm_real_aug+distill | 0.9143 | **0.9259** | **+2.71pp** |

**核心结论：**
- `llm_real_aug` 在标准 test 上最优，Macro-F1 = 0.9244
- `llm_real_aug+distill` 在 hard_test 上最优，Macro-F1 = 0.9259
- 相比 TextCNN baseline，困难样本集提升 **+2.71pp**
- 真实大模型增强 + 知识蒸馏存在叠加收益

## 复现方式

### 环境要求

- Python 3.9
- conda 环境名：CK
- PyTorch 2.6.0+cu118
- transformers 4.57.6
- CUDA 可用（NVIDIA GPU）

### 查看已有结果

```powershell
# 查看最终汇总表
cat outputs/reports/final_baseline_summary.md
cat outputs/reports/final_textcnn_ablation_summary.md

# 查看图表
outputs/figures/final_baseline_macro_f1.png
outputs/figures/final_textcnn_ablation_hard_f1.png
```

### 重新生成汇总表和图表

```powershell
conda run -n CK python scripts/collect_final_experiments.py
```

### 运行单个评估（以 TextCNN baseline 为例）

```powershell
conda run -n CK python scripts/evaluate.py --model textcnn --config configs/textcnn.yaml --checkpoint outputs/checkpoints/textcnn_baseline/best.pt --test_file data/processed/test.csv --output outputs/predictions/textcnn_test_predictions.csv
```

### 运行误判分析

```powershell
conda run -n CK python scripts/error_analysis.py --pred_file outputs/predictions/textcnn_test_predictions.csv --model_name textcnn_test --output_dir outputs/error_analysis
```

## API 安全说明

- API key **不在仓库中**
- `API_key.txt` 和 `.env` 已加入 `.gitignore`
- 真实 API 生成的增强样本已保存在 `data/augmented/` 下
- 默认不需要再次调用 API，所有增强样本已持久化
- 如需重新生成，请在本地创建 `API_key.txt` 并设置环境变量

## 项目目录

```
NLP-Course-Design/
├── configs/          # 配置文件（基线/增强/蒸馏）
├── data/
│   ├── processed/    # 清洗后数据（train/dev/test）
│   ├── hard_eval/    # 困难样本评估集
│   ├── augmented/    # 增强样本和增强训练集
│   └── distillation/ # 教师 logits
├── scripts/          # 可执行脚本
├── src/              # 核心源码
│   ├── models/       # TextCNN/BiLSTM/BertClassifier
│   ├── data/         # Dataset 类
│   ├── training/     # Trainer
│   ├── distillation/ # 蒸馏损失和训练器
│   ├── augmentation/ # 增强模块（模板/LLM/过滤）
│   ├── evaluation/   # 指标和误判分析
│   └── utils/        # 工具模块
├── outputs/
│   ├── checkpoints/  # 模型权重
│   ├── metrics/      # 指标 JSON
│   ├── reports/      # 汇总表
│   └── figures/      # 图表
├── docs/             # 文档
├── requirements.txt
└── README.md
```

## 数据规模

| 划分 | 样本数 |
|---|---|
| train | 8249 |
| dev | 1178 |
| test | 1178 |
| hard_test | 300 |
| LLM 增强样本 | 59 |
