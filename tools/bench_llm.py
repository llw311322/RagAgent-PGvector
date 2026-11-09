"""LLM 端到端测速：首字延迟 + 总耗时 + 生成字数（用真实检索上下文）"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chat
import config
import retriever
from llm import get_llm

Q = "防火规范第5.2.2条是什么"


def run(provider, key=None):
    scored = retriever.retrieve(Q)
    ctx = chat.build_context(scored)
    prompt = chat.SYSTEM_PROMPT.format(context=ctx)
    llm = get_llm(provider, key)
    t0 = time.perf_counter()
    first = None
    buf = []
    for d in llm.stream(prompt, Q):
        if first is None:
            first = time.perf_counter() - t0
        buf.append(d)
    total = time.perf_counter() - t0
    txt = "".join(buf)
    n = len(txt)
    return first, total, n, txt[:60].replace("\n", " ")


if __name__ == "__main__":
    keys = {}
    p = os.path.join(config.BASE_DIR, "data", "api_keys.json")
    if os.path.exists(p):
        keys = json.load(open(p, encoding="utf-8"))

    targets = sys.argv[1:] or ["ollama", "deepseek", "zhipu", "dashscope"]
    for pid in targets:
        try:
            f, t, n, head = run(pid, keys.get(pid))
            print(f"{pid:10s} 首字 {f:5.2f}s | 总 {t:6.2f}s | {n:4d}字 | {n/t:5.1f} 字/s")
            print(f"           {head}")
        except Exception as e:
            print(f"{pid:10s} 失败: {type(e).__name__}: {str(e)[:120]}")
