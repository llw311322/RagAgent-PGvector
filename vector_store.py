"""Chroma 向量库模块：文本向量化、入库、持久化"""
import json
import os
import time
import uuid
from typing import Callable, Dict, List, Optional

import requests
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

import config
import text_splitter

# txt/md 走「按行分块 + 行号定位」；其余类型走通用递归切分
LINE_TRACKED_EXTS = {"txt", "md"}

# 向量化批量参数（走 /api/embed 批量端点，实测约为单条端点的 6 倍速度）
EMBED_BATCH_SIZE = 64
EMBED_TIMEOUT = 600  # 单批最长等待（秒），防止请求无限挂起
EMBED_MAX_RETRY = 3


_EMB_SINGLETON = None  # 进程级单例：OllamaEmbeddings 构造时会校验模型（实测约 1.7s/次），必须复用


def get_embeddings() -> OllamaEmbeddings:
    """创建向量模型实例（本地 ollama，用于查询时向量化）。进程内单例复用。"""
    global _EMB_SINGLETON
    if _EMB_SINGLETON is None:
        _EMB_SINGLETON = OllamaEmbeddings(
            model=config.EMBED_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )
    return _EMB_SINGLETON


_VS_SINGLETON = None  # 进程级单例：避免每次查询重复初始化 Chroma（实测约 1.9s/次）


def reset_vectorstore_singleton():
    """清空/删除 collection 后调用，强制下次重建连接"""
    global _VS_SINGLETON
    _VS_SINGLETON = None


def get_vectorstore() -> Chroma:
    """获取/创建 Chroma 向量库（持久化 + 余弦距离，保证相似度阈值有统一意义）。
    进程内单例，首次创建后复用。"""
    global _VS_SINGLETON
    if _VS_SINGLETON is not None:
        return _VS_SINGLETON
    os.makedirs(config.CHROMA_DIR, exist_ok=True)
    _VS_SINGLETON = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=config.CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"},  # 用余弦距离，便于阈值过滤
    )
    return _VS_SINGLETON


def _embed_texts(texts: List[str]) -> List[List[float]]:
    """批量向量化：走 Ollama /api/embed 批量端点，带超时与重试，绝不无限挂起"""
    url = config.OLLAMA_BASE_URL.rstrip("/") + "/api/embed"
    embeddings: List[List[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        last_err: Optional[Exception] = None
        for attempt in range(1, EMBED_MAX_RETRY + 1):
            try:
                r = requests.post(
                    url,
                    json={"model": config.EMBED_MODEL, "input": batch},
                    timeout=EMBED_TIMEOUT,
                )
                r.raise_for_status()
                data = r.json().get("embeddings")
                if not data or len(data) != len(batch):
                    raise RuntimeError(f"返回向量数不符: {len(data) if data else 0}/{len(batch)}")
                embeddings.extend(data)
                last_err = None
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(1.5 * attempt)  # 退避后重试
        if last_err is not None:
            raise RuntimeError(f"向量化失败（已重试 {EMBED_MAX_RETRY} 次）: {last_err}")
    return embeddings


def _registry_path() -> str:
    return os.path.join(os.path.dirname(config.CHROMA_DIR), "kb_registry.json")


def _read_registry() -> Optional[dict]:
    try:
        with open(_registry_path(), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_registry(reg: dict) -> None:
    try:
        with open(_registry_path(), "w", encoding="utf-8") as f:
            json.dump(reg, f, ensure_ascii=False)
    except Exception:
        pass


# 块数缓存。Chroma 的 count() 在库大、且有删除残留时需要 2 秒以上（实测 4.9 万块 / 删过 2852 块
# → 每次 2.1 秒），而检索每次都要用它做「索引是否过期」的版本判断。不缓存会让检索从 75ms
# 变成 4~6 秒，这是接精排时暴露出来的最大性能坑。
_COUNT = {"n": -1}


def get_count(vs: Optional[Chroma] = None) -> int:
    """库内块数（进程内缓存，写库/删库后由 invalidate_all_indexes 清空）"""
    if _COUNT["n"] >= 0:
        return _COUNT["n"]
    vs = vs or get_vectorstore()
    _COUNT["n"] = int(vs._collection.count())
    return _COUNT["n"]


def _registry(vs: Optional[Chroma] = None) -> Optional[dict]:
    """读取注册表；块数与库内实际不一致时视为失效（返回 None，由调用方全量兜底重建）"""
    reg = _read_registry()
    if not reg:
        return None
    vs = vs or get_vectorstore()
    return reg if reg.get("count") == get_count(vs) else None


def _scan_all(col, include=("metadatas",), batch: int = 2000):
    """分页读取整库。Chroma 底层是 SQLite，一次 get 全部行会报
    「too many SQL variables」（库大时必然触发），必须分批。"""
    offset = 0
    while True:
        res = col.get(include=list(include), limit=batch, offset=offset)
        ids = res.get("ids") or []
        if not ids:
            return
        yield res
        offset += len(ids)
        if len(ids) < batch:
            return


def _rebuild_registry(vs: Optional[Chroma] = None) -> dict:
    """全量扫描 metadata 重建注册表（注册表缺失/失效时的兜底）"""
    vs = vs or get_vectorstore()
    col = vs._collection
    reg = {"count": 0, "sources": {}}
    for res in _scan_all(col):
        reg["count"] += len(res.get("ids") or [])
        for m in res.get("metadatas") or []:
            if not m or not m.get("source"):
                continue
            e = reg["sources"].setdefault(m["source"], {"mtime": None, "chunks": 0})
            e["chunks"] += 1
            if m.get("mtime") is not None:
                e["mtime"] = str(m["mtime"])
    return reg


def load_existing_by_source(vs: Optional[Chroma] = None) -> Dict[str, dict]:
    """库内现有块索引：{source: {"ids": [...], "mtimes": {...}}}。
    优先读注册表（O(1)），失效时才全量扫描 metadata。"""
    vs = vs or get_vectorstore()
    reg = _registry(vs)
    if reg is not None:
        return {
            s: {"ids": [], "mtimes": ({v["mtime"]} if v.get("mtime") is not None else set())}
            for s, v in reg.get("sources", {}).items()
        }

    result: Dict[str, dict] = {}
    for res in _scan_all(vs._collection):
        for cid, m in zip(res.get("ids") or [], res.get("metadatas") or []):
            if not m or not m.get("source"):
                continue
            entry = result.setdefault(m["source"], {"ids": [], "mtimes": set()})
            entry["ids"].append(cid)
            if m.get("mtime") is not None:
                entry["mtimes"].add(str(m["mtime"]))
    return result


def add_documents(
    docs: List[Document],
    existing: Optional[Dict[str, dict]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> int:
    """
    文档入库：
    - txt/md：按行分块，metadata 记录原文起止行号（用于引用溯源定位原文）；
    - 按 source（文件名）去重；文件内容变化（mtime 不同）时自动删除旧块重灌；
    - existing: 预加载的库内索引（多次调用时传入同一份，避免重复全量拉取）；
    - progress_cb(done, total): 分块向量化进度回调；
    - 返回新入库的块数（0 表示无变化）。
    """
    if not docs:
        return 0
    vs = get_vectorstore()
    if existing is None:
        existing = load_existing_by_source(vs)
    reg = _registry(vs)   # 注册表须在删块之前读取（删块后块数变化会被判为失效）

    new_docs: List[Document] = []
    for d in docs:
        src = d.metadata.get("source")
        if src and src in existing:
            old_mtimes = existing[src]["mtimes"]
            new_mtime = str(d.metadata.get("mtime")) if d.metadata.get("mtime") is not None else None
            if new_mtime is not None and new_mtime in old_mtimes:
                continue  # 同一文件且未修改，跳过
            if new_mtime is not None or not old_mtimes:
                # 文件已修改（或旧数据无 mtime 标记）→ 删旧块重灌（按 source 过滤，不依赖 id 列表）
                vs.delete(where={"source": src})
                del existing[src]
                if reg is not None:
                    reg["sources"].pop(src, None)
            else:
                continue  # 无法判断新旧（均无 mtime），保守跳过
        new_docs.append(d)

    if not new_docs:
        return 0

    # 分块：txt/md 用行号分块，其余用递归切分
    split_docs: List[Document] = []
    for d in new_docs:
        ext = (d.metadata.get("type") or "").lower()
        if ext in LINE_TRACKED_EXTS:
            split_docs += text_splitter.split_text_lines(
                d.page_content,
                d.metadata.get("source", "未知来源"),
                ext,
                mtime=d.metadata.get("mtime"),
            )
        else:
            split_docs += text_splitter.split_documents([d])

    if not split_docs:
        return 0

    # 批量向量化（带进度回调），再直写底层 collection
    # 注意：向量化用「标题路径 + 正文」（text_splitter.embed_text），文档仍存正文本身，
    # 这样既能带上条文标题上下文提升召回，又不破坏行号与原文的一致性。
    texts = [text_splitter.embed_text(d) for d in split_docs]
    bodies = [d.page_content for d in split_docs]
    embeddings: List[List[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE * 4):
        seg = texts[i : i + EMBED_BATCH_SIZE * 4]
        embeddings.extend(_embed_texts(seg))
        if progress_cb:
            progress_cb(min(i + len(seg), len(texts)), len(texts))

    col = vs._collection
    col.upsert(
        ids=[str(uuid.uuid4()) for _ in split_docs],
        embeddings=embeddings,
        documents=bodies,
        metadatas=[d.metadata for d in split_docs],
    )
    invalidate_all_indexes()

    # 更新注册表（统计与去重都靠它，避免每次全量拉 metadata）
    if reg is None:
        reg = _rebuild_registry(vs)
    else:
        for d in split_docs:
            src = d.metadata.get("source")
            if not src:
                continue
            e = reg["sources"].setdefault(src, {"mtime": None, "chunks": 0})
            e["chunks"] += 1
            if d.metadata.get("mtime") is not None:
                e["mtime"] = str(d.metadata.get("mtime"))
    _COUNT["n"] = int(col.count())
    reg["count"] = _COUNT["n"]
    _write_registry(reg)
    return len(split_docs)


# ---------- 标题倒排索引（用于「释义式提问」的召回）----------
# 场景：用户按条文标题的意思提问（「工业建筑的耐火等级不应低于二级的情形」），
# 向量相似度只有 0.70，进不了 top15；但该问题与条文标题词面高度重合。
# 这里为每个片段的标题建一个二元组倒排索引，查询时按标题重合度补召，命中率显著提升。
_TITLE_IDX = {"count": -1, "items": [], "postings": {}}


def invalidate_title_index() -> None:
    _TITLE_IDX["count"] = -1


def invalidate_all_indexes() -> None:
    """库内容变化后，块数缓存、标题索引与 BM25 词面索引必须同时失效，否则检索会用到旧数据"""
    _COUNT["n"] = -1
    invalidate_title_index()
    try:
        import bm25_index

        bm25_index.invalidate()
    except Exception:
        pass


def _cjk_bigrams(text: str) -> List[str]:
    cjk = "".join(ch if "\u4e00" <= ch <= "\u9fff" else " " for ch in text)
    grams = []
    for run in cjk.split():
        grams += [run[i : i + 2] for i in range(len(run) - 1)]
    return grams


def get_title_index() -> dict:
    """返回 {"items": [(cid, title)], "postings": {bigram: [item_idx]}}，按库内块数缓存。

    大库（几万块）全量读 metadata 要十几秒，因此构建后写一份磁盘缓存，
    缓存以块数为版本号，库内容变化（块数变化）时自动重建。
    """
    vs = get_vectorstore()
    col = vs._collection
    n = get_count(vs)
    if _TITLE_IDX["count"] == n and _TITLE_IDX["items"]:
        return _TITLE_IDX

    cache = os.path.join(os.path.dirname(config.CHROMA_DIR), "title_index.json")
    try:
        with open(cache, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if obj.get("count") == n and obj.get("items"):
            _TITLE_IDX.update(
                count=n,
                items=[tuple(x) for x in obj["items"]],
                postings=obj["postings"],
            )
            return _TITLE_IDX
    except Exception:
        pass

    data_rows = _scan_all(col, include=("metadatas",))
    items: List[tuple] = []
    postings: Dict[str, List[int]] = {}
    for res in data_rows:
        for cid, m in zip(res.get("ids") or [], res.get("metadatas") or []):
            h = (m or {}).get("heading")
            if not h:
                continue
            title = h.split(">")[-1].strip()
            if not title:
                continue
            gi = len(items)
            items.append((cid, title))
            for bg in set(_cjk_bigrams(title)):
                postings.setdefault(bg, []).append(gi)
    _TITLE_IDX.update(count=n, items=items, postings=postings)

    try:
        with open(cache, "w", encoding="utf-8") as f:
            json.dump({"count": n, "items": items, "postings": postings}, f, ensure_ascii=False)
    except Exception:
        pass
    return _TITLE_IDX


def get_stats() -> dict:
    """返回知识库统计信息（优先读注册表，O(1)）"""
    vs = get_vectorstore()
    reg = _registry(vs)
    if reg is None:
        reg = _rebuild_registry(vs)
        _write_registry(reg)
    sources = reg.get("sources", {})
    return {
        "chunks": reg.get("count", 0),
        "docs": len(sources),
        "sources": sorted(sources.keys()),
    }


def delete_source(source: str) -> int:
    """按文档名删除其全部向量块，返回删除的块数"""
    vs = get_vectorstore()
    reg = _registry(vs)
    n = (reg or {}).get("sources", {}).get(source, {}).get("chunks", 0)
    vs.delete(where={"source": source})
    invalidate_all_indexes()
    if reg is not None:
        reg["sources"].pop(source, None)
        _COUNT["n"] = int(vs._collection.count())
        reg["count"] = _COUNT["n"]
        _write_registry(reg)
    return n


def clear_all() -> None:
    """清空整个知识库"""
    vs = get_vectorstore()
    client = vs._client
    client.delete_collection(config.COLLECTION_NAME)
    reset_vectorstore_singleton()
    invalidate_all_indexes()
    try:
        os.remove(_registry_path())
    except OSError:
        pass
