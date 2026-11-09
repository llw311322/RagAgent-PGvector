"""入库链路：文档 → 分块打标 → 批量向量化 → PGVector

保持与旧链路完全一致的分块与元数据（text_splitter + meta_tagger），
只把「写入目标」换成 PGVector、「向量模型」换成 bge-m3（1024 维）。
向量化文本仍是「标题路径 + 正文」（text_splitter.embed_text），
但落库的 document 字段是正文本身，保证引用时能对上原文行号。
"""
import os
import uuid
from typing import Callable, List, Optional, Tuple

import config
import document_loader
import text_splitter
from lc import bm25, embeddings, store

_NS = uuid.UUID("6f1a2b3c-4d5e-4f60-8a7b-9c0d1e2f3a4b")


def _chunk_id(source: str, idx: int) -> str:
    """确定性 id：同一文件同一块，重复灌库不会产生副本"""
    return str(uuid.uuid5(_NS, f"{source}#{idx}"))


def ingest_file(path: str, mtime: float, progress_cb: Optional[Callable] = None) -> int:
    source = os.path.basename(path)
    ext = os.path.splitext(source)[1].lstrip(".").lower()

    raw_docs = document_loader.load_document(path)
    chunks = []
    for d in raw_docs:
        if ext in {"txt", "md"}:
            chunks.extend(
                text_splitter.split_text_lines(d.page_content, source, ext, mtime=mtime)
            )
        else:
            chunks.extend(text_splitter.split_documents([d]))
    for d in chunks:                       # 兼容非 txt/md 分支（split_documents 不带 source）
        d.metadata.setdefault("source", source)
        d.metadata.setdefault("type", ext)
        d.metadata["mtime"] = mtime
    if not chunks:
        return 0

    # 覆盖式写入：先删该文件旧块，再写新块（条文修订后不会残留旧条文）
    store.delete_source(source)

    total = len(chunks)
    done = 0
    for i in range(0, total, config.EMBED_BATCH_PG):
        batch = chunks[i : i + config.EMBED_BATCH_PG]
        # 送进模型的文本带标题路径；入库的正文保持原文
        vecs = embeddings.embed_documents([text_splitter.embed_text(d) for d in batch])
        store.get_store().add_embeddings(
            texts=[d.page_content for d in batch],
            embeddings=vecs,
            metadatas=[d.metadata for d in batch],
            ids=[_chunk_id(source, d.metadata.get("chunk_index", i + j)) for j, d in enumerate(batch)],
        )
        done += len(batch)
        if progress_cb:
            progress_cb(done, total)
    return total


def scan_docs_dir(progress_cb: Optional[Callable] = None, force: bool = False) -> Tuple[int, List[str]]:
    """增量扫描：mtime 变化的文件才重灌。progress_cb(done_f, total_f, name, done_c, total_c)"""
    os.makedirs(config.DOCS_DIR, exist_ok=True)
    files = [
        f for f in sorted(os.listdir(config.DOCS_DIR))
        if os.path.isfile(os.path.join(config.DOCS_DIR, f))
        and os.path.splitext(f)[1].lower() in document_loader.SUPPORTED_EXTS
    ]
    existing = store.existing_sources()
    total = 0
    details: List[str] = []
    for idx, f in enumerate(files, 1):
        path = os.path.join(config.DOCS_DIR, f)
        mtime = os.path.getmtime(path)
        old = existing.get(f)
        if old and not force and abs(old[0] - mtime) < 1.0:
            continue
        try:
            def cb(done_c: int, total_c: int, _f=f, _i=idx, _n=len(files)):
                if progress_cb:
                    progress_cb(_i, _n, _f, done_c, total_c)

            n = ingest_file(path, mtime, cb)
            total += n
            if n:
                tag = "更新" if old else "新增"
                details.append(f"✅ {f}：{tag} {n} 块")
        except Exception as e:  # noqa: BLE001
            details.append(f"❌ {f}：{e}")
    if total:
        bm25.invalidate()
    return total, details


def rebuild_all(progress_cb: Optional[Callable] = None) -> Tuple[int, List[str]]:
    """全量重建：清空集合 → 重灌 docs（换向量模型/改分块策略后用）"""
    store.drop_collection()
    store.get_store(refresh=True)
    bm25.invalidate()
    return scan_docs_dir(progress_cb, force=True)
