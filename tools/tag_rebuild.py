"""对库内已有数据全量重打标 + 清洗噪声块（不重新向量化，只改 metadata / 删块）

用法：
    python tools/tag_rebuild.py            # 预览：只统计，不写库
    python tools/tag_rebuild.py --apply    # 实际执行：更新 metadata、删除噪声块

打标字段：std_class / std_no / domain / buildings / quality / garble
清洗动作：quality=noise（图示扫描页乱码）与 thin（无条文号的碎片块）直接删除。
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import meta_tagger
import config
from vector_store import (
    _scan_all,
    get_vectorstore,
    invalidate_all_indexes,
)


def _source_statuses() -> dict:
    """按源文件正文首部判定文档有效性（valid / draft / abolished）。

    必须整篇判一次再传给每个块：草案身份的证据（「（征求意见稿）」）只在首页出现，
    逐块判定会让同一份草案的其余 700 多块仍被当成强制性规范。
    """
    out = {}
    try:
        names = os.listdir(config.DOCS_DIR)
    except OSError:
        return out
    for fn in names:
        if not fn.lower().endswith((".md", ".txt")):
            continue
        try:
            with open(os.path.join(config.DOCS_DIR, fn), encoding="utf-8", errors="ignore") as f:
                head = f.read(1200)
        except OSError:
            head = ""
        out[fn] = meta_tagger.doc_status(fn, head)
    return out


def main(apply: bool) -> None:
    vs = get_vectorstore()
    col = vs._collection
    total = col.count()
    print(f"库内块数: {total}")
    statuses = _source_statuses()
    print(f"文档有效性: 草案 {sum(1 for v in statuses.values() if v == 'draft')} 篇, "
          f"已废止 {sum(1 for v in statuses.values() if v == 'abolished')} 篇")

    upd_ids: list = []
    upd_metas: list = []
    del_ids: list = []
    stats = {
        "quality": Counter(),
        "std_class": Counter(),
        "domain": Counter(),
        "buildings": Counter(),
        "source_type": Counter(),
        "mandatory": Counter(),
        "valid": Counter(),
        "noise_by_source": Counter(),
    }
    done = 0

    def flush_update():
        if upd_ids:
            for i in range(0, len(upd_ids), 500):
                col.update(ids=upd_ids[i : i + 500], metadatas=upd_metas[i : i + 500])
            upd_ids.clear()
            upd_metas.clear()

    def flush_delete():
        if del_ids:
            for i in range(0, len(del_ids), 500):
                col.delete(ids=del_ids[i : i + 500])
            del_ids.clear()

    for res in _scan_all(col, include=("documents", "metadatas"), batch=2000):
        for cid, text, meta in zip(
            res.get("ids") or [], res.get("documents") or [], res.get("metadatas") or []
        ):
            done += 1
            meta = dict(meta or {})
            meta_tagger.tag_chunk(meta, text or "", statuses.get(meta.get("source", ""), ""))
            q = meta.get("quality")
            stats["quality"][q] += 1
            stats["std_class"][meta.get("std_class", "")] += 1
            stats["domain"][meta.get("domain", "")] += 1
            stats["source_type"][meta.get("source_type", "")] += 1
            stats["mandatory"]["强制性" if meta.get("is_mandatory") else "一般"] += 1
            stats["valid"]["现行" if meta.get("valid", True) else "作废/征求意见"] += 1
            for b in meta.get("buildings", []) or []:
                stats["buildings"][b] += 1
            if q in ("noise", "thin"):
                stats["noise_by_source"][meta.get("source", "?")] += 1
                del_ids.append(cid)
            else:
                upd_ids.append(cid)
                upd_metas.append(meta)
        if apply:
            flush_update()
            flush_delete()
        if done % 10000 == 0:
            print(f"  已处理 {done}/{total}")

    if apply:
        flush_update()
        flush_delete()

    print("\n===== 清洗与打标报告 =====")
    print(f"总块数 {done}")
    print(f"质量: {dict(stats['quality'])}")
    print(f"删除块数: {len(stats['noise_by_source']) and sum(stats['noise_by_source'].values())}")
    print("\n标准大类:", dict(stats["std_class"]))
    print("\n条文主题:", dict(stats["domain"]))
    print("\n建筑类型:", dict(stats["buildings"]))
    print("\n条文类型:", dict(stats["source_type"]))
    print("强制性:", dict(stats["mandatory"]), "| 效力:", dict(stats["valid"]))
    top_noise = stats["noise_by_source"].most_common(15)
    if top_noise:
        print("\n噪声最多的文件:")
        for s, n in top_noise:
            print(f"  {n:5d}  {s[:60]}")

    if apply:
        if del_ids or upd_ids:
            invalidate_all_indexes()
        print(f"\n完成。剩余块数: {col.count()}")
    else:
        print("\n（预览模式，未写库。加 --apply 执行）")


if __name__ == "__main__":
    main("--apply" in sys.argv)
