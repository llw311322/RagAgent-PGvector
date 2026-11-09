"""端到端自检：维度 / 库存 / 检索命中 / 拒答 / 延迟

用法：python tools/check_lc.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lc import bm25, embeddings, retriever, store  # noqa: E402

HIT_CASES = [
    ("防火规范第5.2.2条是什么", ["5.2.2"]),
    ("消防车道的净宽度和净空高度是多少", ["4.0", "4m"]),
    ("住宅建筑层高有什么规定", ["层高", "净高"]),
    ("疏散楼梯的最小净宽度", ["净宽"]),
    ("地下汽车库的防火分区最大允许建筑面积", ["防火分区"]),
]
REJECT_CASES = ["今天天气怎么样", "帮我写一份简历", "红烧肉怎么做"]


def main() -> int:
    print("=" * 62)
    st = embeddings.status()
    print(f"向量模型 : {st.get('model')} | 维度 {st.get('dim')} | ok={st.get('ok')}")
    s = store.stats()
    print(f"向量库   : PGVector / {s['collection']} | 文档 {s['docs']} 篇 | 块 {s['chunks']}")
    idx = bm25.get_index()
    print(f"BM25     : {idx.n_docs if idx else 0} 块")

    ok = 0
    lat = []
    print("-" * 62)
    for q, expect in HIT_CASES:
        t = time.time()
        docs = retriever.retrieve(q)
        dt = time.time() - t
        lat.append(dt)
        blob = " ".join((d.metadata.get("clause_no") or "") + d.page_content[:200] for d in docs)
        hit = any(e in blob for e in expect)
        ok += hit
        top = docs[0] if docs else None
        head = ""
        if top:
            m = top.metadata
            head = f"{m.get('standard_code') or '—'} {m.get('clause_no') or ''} rerank={m.get('rerank')}"
        print(f"{'✅' if hit else '❌'} {q[:26]:<28} {len(docs)} 条 {dt:5.2f}s  {head[:52]}")

    print("-" * 62)
    rejected = 0
    for q in REJECT_CASES:
        t = time.time()
        docs = retriever.retrieve(q)
        dt = time.time() - t
        rejected += (len(docs) == 0)
        print(f"{'✅' if not docs else '❌'} 拒答 {q[:20]:<22} 命中 {len(docs)} 条 {dt:5.2f}s")

    print("-" * 62)
    print(f"命中 {ok}/{len(HIT_CASES)} | 拒答 {rejected}/{len(REJECT_CASES)} | "
          f"检索均值 {sum(lat)/len(lat)*1000:.0f}ms")
    return 0 if ok == len(HIT_CASES) and rejected == len(REJECT_CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
