"""重建知识库：清空后按新分块规则全量灌库（幂等、可续跑、带进度）

用法：
  python tools/reindex.py --reset                 # 清空并从零全量灌库
  python tools/reindex.py                         # 续跑：只灌未入库/已修改的文件
  python tools/reindex.py --pattern 防火          # 只灌文件名含「防火」的文档
  python tools/reindex.py --limit 5               # 只灌前 5 个文件（试跑）
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config                      # noqa: E402
import document_loader             # noqa: E402
import vector_store                # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="先清空向量库")
    ap.add_argument("--pattern", default="", help="只处理文件名包含该字符串的文档")
    ap.add_argument("--limit", type=int, default=0, help="最多处理多少个文件")
    args = ap.parse_args()

    if args.reset:
        print("清空向量库…", flush=True)
        vector_store.clear_all()
        vector_store.reset_vectorstore_singleton()

    files = [
        f for f in sorted(os.listdir(config.DOCS_DIR))
        if os.path.isfile(os.path.join(config.DOCS_DIR, f))
        and os.path.splitext(f)[1].lower() in document_loader.SUPPORTED_EXTS
    ]
    if args.pattern:
        files = [f for f in files if args.pattern in f]
    if args.limit:
        files = files[: args.limit]

    total_files = len(files)
    print(f"待处理文件 {total_files} 个（目录 {config.DOCS_DIR}）", flush=True)

    existing = vector_store.load_existing_by_source()
    print(f"库内已有 {len(existing)} 篇文档", flush=True)

    t0 = time.time()
    done_files = 0
    total_chunks = 0
    failed = []

    for idx, name in enumerate(files, 1):
        path = os.path.join(config.DOCS_DIR, name)
        mtime = os.path.getmtime(path)
        try:
            docs = document_loader.load_document(path)
            for d in docs:
                d.metadata["mtime"] = mtime

            def cb(done: int, tot: int, _n=name, _i=idx) -> None:
                print(f"  [{_i}/{total_files}] {_n[:34]} 向量化 {done}/{tot}", flush=True)

            n = vector_store.add_documents(docs, existing=existing, progress_cb=cb)
            if n:
                done_files += 1
                total_chunks += n
                el = time.time() - t0
                rate = total_chunks / el if el else 0
                rest_files = total_files - idx
                print(
                    f"[{idx}/{total_files}] ✅ {name[:40]} +{n} 块"
                    f"｜累计 {total_chunks} 块 {rate:.1f} 块/秒"
                    f"｜已用 {el/60:.1f} 分钟｜预计剩余 {rest_files * el / max(idx,1) / 60:.1f} 分钟",
                    flush=True,
                )
            else:
                print(f"[{idx}/{total_files}] 跳过（已入库且未修改）{name[:40]}", flush=True)
        except Exception as e:  # noqa: BLE001
            failed.append((name, str(e)))
            print(f"[{idx}/{total_files}] ❌ {name[:40]} -> {e}", flush=True)

        # 每 10 个文件打一次库内总量，便于观察
        if idx % 10 == 0:
            s = vector_store.get_stats()
            print(f"    —— 当前库内：{s['docs']} 篇 / {s['chunks']} 块", flush=True)

    s = vector_store.get_stats()
    print(
        f"\n完成：新灌 {done_files} 篇 / {total_chunks} 块，用时 {(time.time()-t0)/60:.1f} 分钟\n"
        f"库内合计：{s['docs']} 篇 / {s['chunks']} 块"
        f"{'，失败 ' + str(len(failed)) + ' 个：' + str([f[0] for f in failed]) if failed else ''}",
        flush=True,
    )

    # 预先构建并落盘标题索引（否则应用首次提问会卡十几秒）
    t = time.time()
    idx = vector_store.get_title_index()
    print(f"标题索引：{len(idx['items'])} 条，构建/落盘用时 {time.time()-t:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
