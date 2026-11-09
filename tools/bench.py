"""性能诊断：分别测量 向量化 / 召回 / 重排 / LLM 首字延迟 各环节耗时"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import retriever
from vector_store import get_embeddings, get_vectorstore

QS = [
    "防火规范第5.2.2条是什么",
    "工业建筑的耐火等级不应低于二级的情形",
    "电动汽车充电站与其他建筑之间的防火间距",
    "高层民用建筑的消防车道宽度要求",
    "今天天气怎么样",
]


def t(fn, *a, **k):
    s = time.perf_counter()
    r = fn(*a, **k)
    return r, (time.perf_counter() - s) * 1000


def main():
    st = get_vectorstore()._collection.count()
    print(f"collection chunks = {st}")

    emb_fn = get_embeddings()
    _, ms = t(emb_fn.embed_query, "预热")
    print(f"[embed_query 预热后] {ms:.0f} ms")

    # 单次向量化耗时
    for q in QS:
        emb, ms = t(emb_fn.embed_query, q)
        print(f"  embed '{q[:14]}...' = {ms:.0f} ms")

    print("\n--- 全链路检索 ---")
    tot = 0
    for q in QS:
        res, ms = t(retriever.retrieve, q)
        tot += ms
        top = res[0] if res else None
        print(f"{ms:7.0f} ms | {len(res)} 命中 | {q}")
        if top:
            print(f"         top1={top[1]:.3f} {top[0].metadata.get('source','')[:30]} | {top[0].metadata.get('article','')}")
        else:
            print("         (无命中)")

    print(f"\n平均检索 {tot/len(QS):.0f} ms")

    # 各子步骤拆解（拿最慢的那条）
    q = QS[0]
    emb, ms1 = t(emb_fn.embed_query, q)
    cand, ms2 = t(retriever._query_collection, emb, 15)
    print(f"\n拆解 '{q}': embed {ms1:.0f} ms + 向量召回 {ms2:.0f} ms")


if __name__ == "__main__":
    main()
