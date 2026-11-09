# -*- coding: utf-8 -*-
"""一次性 UI 启动验证脚本：启动 Streamlit -> 检查页面 -> 自动关闭
用法: venv\Scripts\python.exe verify_ui.py
"""
import os
import subprocess
import sys
import time
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(BASE, "app.py")
PORT = 8501

cmd = [
    sys.executable, "-m", "streamlit", "run", APP,
    "--server.headless=true",
    f"--server.port={PORT}",
    "--browser.gatherUsageStats=false",
]

print("启动 Streamlit 中...")
proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    time.sleep(15)
    with urllib.request.urlopen(f"http://localhost:{PORT}", timeout=10) as r:
        body = r.read().decode("utf-8", "ignore")
        ok = r.status == 200 and len(body) > 0
        print(f"HTTP {r.status} | 页面字节数 {len(body)}")
        if ok:
            print("✅ Streamlit 界面启动成功，浏览器访问 http://localhost:8501 即可使用")
        else:
            print("❌ 页面响应异常")
except Exception as e:
    print("❌ 启动验证失败:", e)
finally:
    proc.terminate()
    time.sleep(2)
    print("验证完成，已关闭临时服务进程（不影响你之后用 run.bat 启动）")
