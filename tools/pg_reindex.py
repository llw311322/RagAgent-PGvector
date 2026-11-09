"""全量重建：清空 PG 集合 → 按 docs/ 重新分块打标 → bge-m3 向量化 → 重建 BM25

用法：
    python tools/pg_reindex.py            # 全量重建
    python tools/pg_reindex.py --scan     # 只做增量（mtime 变化的文件）
    python tools/pg_reindex.py --stats    # 只看库内统计
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from lc import bm25, ingest, store


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true", help="增量入库（默认全量重建）")
    ap.add_argument("--stats", action="store_true", help="只输出统计")
    args = ap.parse_args()

    t0 = time.time()
    if args.stats:
        s = store.stats()
        _log(f"文档 {s['docs']} 篇 / 块 {s['chunks']}")
        for f in ("std_class", "domain", "buildings"):
            _log(f"{f}: {store.distribution(f, 8)}")
        return 0

    last = {"t": 0.0}

    def cb(done_f: int, total_f: int, name: str, done_c: int, total_c: int):
        now = time.time()
        if now - last["t"] < 5 and done_c != total_c:
            return
        last["t"] = now
        _log(f"({done_f}/{total_f}) {name[:46]} 向量化 {done_c}/{total_c}")

    if args.scan:
        n, details = ingest.scan_docs_dir(cb)
        _log(f"增量完成：新增 {n} 块，耗时 {time.time()-t0:.0f}s")
    else:
        _log("清空集合并全量重建…")
        n, details = ingest.rebuild_all(cb)
        _log(f"全量完成：{n} 块，耗时 {(time.time()-t0)/60:.1f} 分钟")

    for d in details[:10]:
        _log(d)
    errs = [d for d in details if d.startswith("❌")]
    if errs:
        _log(f"失败 {len(errs)} 个文件：")
        for e in errs[:20]:
            _log(e)

    s = store.stats()
    _log(f"库内：文档 {s['docs']} 篇 / 块 {s['chunks']}")
    _log("重建 BM25…")
    tb = time.time()
    idx = bm25.get_index(force=True)
    _log(f"BM25 就绪：{idx.n_docs if idx else 0} 块，{time.time()-tb:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
