@echo off
set /p OPENAI_API_KEY=<API_key.txt
set OPENAI_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
conda run -n CK python scripts/test_mimo_api.py
