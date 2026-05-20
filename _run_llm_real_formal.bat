@echo off
set /p OPENAI_API_KEY=<API_key.txt
set OPENAI_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
conda run -n CK python scripts/generate_llm_augmented_samples.py --config configs/augmentation_mimo_real.yaml --input_file data/hard_eval/hard_test.csv --output_file data/augmented/llm_real_augmented_samples.csv --target_per_type 20
