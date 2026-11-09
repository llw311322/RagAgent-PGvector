@echo off
chcp 65001 >nul
title RAG 本地问答机器人
cd /d %~dp0

rem 解决中文 Windows 下 Python 的编码兼容问题
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo ============================================
echo   本地 RAG 文档问答机器人 启动脚本
echo ============================================
echo.

rem 1. 检查虚拟环境，没有则创建
if not exist venv\Scripts\python.exe (
    echo [1/3] 首次运行，正在创建虚拟环境...
    python -m venv venv
    if errorlevel 1 (
        echo [错误] 创建虚拟环境失败，请确认已安装 Python 3.9+
        pause
        exit /b 1
    )
    echo [2/3] 正在安装依赖（首次约需 3~10 分钟）...
    venv\Scripts\pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    if errorlevel 1 (
        echo [错误] 依赖安装失败，请检查网络后重试
        pause
        exit /b 1
    )
) else (
    echo [1/3] 虚拟环境已存在，跳过创建
)

echo [2/3] 检查 ollama 服务...
ollama list >nul 2>&1
if errorlevel 1 (
    echo [警告] 未检测到 ollama 服务，请先启动 ollama（托盘图标运行中）
)

echo [3/3] 正在启动问答机器人...
echo 启动后请用浏览器打开 http://localhost:8501
echo.
venv\Scripts\streamlit run app.py
pause
