"""BM25 词面索引（数据源改为 PGVector 表）

为什么不用 langchain_community 的 BM25Retriever：
- 它是纯内存的，每次进程启动都要把 4.6 万块重新分词建索引（约 60~100s），没有持久化；
- 已有的 BM25Index 用 numpy 倒排 + pickle 缓存，启动 8s、查询 0.4~16ms，
  且中文二元组分词是为规范文本调过的。
所以这里只换「数据来源」（Chroma → langchain_pg_embedding），检索算法与缓存机制不变，
再通过 LangChain 的 BaseRetriever 接口参与 EnsembleRetriever 融合。
"""
import os
import pickle
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import psycopg

import config
from bm25_index import MAX_POSTING_DOCS, BM25Index, tokenize
from lc import store

_CACHE = os.path.join(config.BASE_DIR, "data", "bm25_pg.pkl")
_STATE: dict = {"stamp": None, "index": None}


def _iter_rows() -> Iterator[Tuple[str, str, dict]]:
    cid = store._collection_id()
    if not cid:
        return
    with psycopg.connect(config.PG_DSN_RAW) as conn:
        with conn.cursor(name="bm25_scan") as cur:      # 服务端游标，避免 4.6 万行全塞内存
            cur.itersize = 5000
            cur.execute(
                "SELECT id::text, document, cmetadata FROM langchain_pg_embedding "
                "WHERE collection_id = %s",
                (cid,),
            )
            for row in cur:
                yield row[0], (row[1] or ""), (row[2] or {})


def _stamp() -> str:
    """版本指纹：块数 + 最新 mtime。入库/删文都会变，避免误用旧缓存"""
    cid = store._collection_id()
    if not cid:
        return "empty"
    rows = store._sql(
        "SELECT count(*), coalesce(max((cmetadata->>'mtime')::float8), 0) "
        "FROM langchain_pg_embedding WHERE collection_id = %s",
        (cid,),
    )
    n, mt = rows[0] if rows else (0, 0)
    return f"{n}:{mt:.0f}"


def _build() -> BM25Index:
    postings: Dict[str, Tuple[List[int], List[int]]] = {}
    doc_len: List[int] = []
    ids: List[str] = []
    docs: List[str] = []
    metas: List[dict] = []
    n = 0
    for cid, text, meta in _iter_rows():
        heading = (meta or {}).get("heading") or ""
        full = f"{heading}\n{text}" if heading else text
        toks = tokenize(full)
        doc_len.append(len(toks))
        ids.append(cid)
        docs.append(text)
        metas.append(meta)
        counts: Dict[str, int] = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
        for t, c in counts.items():
            e = postings.get(t)
            if e is None:
                postings[t] = ([n], [c])
            else:
                e[0].append(n)
                e[1].append(c)
        n += 1

    packed = {
        t: (np.asarray(i, dtype=np.int32), np.asarray(c, dtype=np.float32))
        for t, (i, c) in postings.items()
        if len(i) <= MAX_POSTING_DOCS
    }
    dl = np.asarray(doc_len, dtype=np.float32)
    return BM25Index(
        packed, dl, float(dl.mean()) if len(dl) else 0.0, n, ids, docs, metas
    )


def get_index(force: bool = False) -> Optional[BM25Index]:
    stamp = _stamp()
    if stamp == "empty":
        return None
    if not force and _STATE["stamp"] == stamp and _STATE["index"] is not None:
        return _STATE["index"]

    if not force:
        try:
            with open(_CACHE, "rb") as f:
                obj = pickle.load(f)
            if obj.get("stamp") == stamp:
                idx = BM25Index(
                    obj["postings"], obj["doc_len"], obj["avgdl"], obj["n_docs"],
                    obj.get("ids"), obj.get("docs"), obj.get("metas"),
                )
                _STATE.update(stamp=stamp, index=idx)
                return idx
        except Exception:
            pass

    idx = _build()
    _STATE.update(stamp=stamp, index=idx)
    try:
        with open(_CACHE, "wb") as f:
            pickle.dump(
                {
                    "stamp": stamp,
                    "postings": idx.postings,
                    "doc_len": idx.doc_len,
                    "avgdl": idx.avgdl,
                    "n_docs": idx.n_docs,
                    "ids": idx.ids,
                    "docs": idx.docs,
                    "metas": idx.metas,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    except Exception:
        pass
    return idx


def invalidate() -> None:
    _STATE.update(stamp=None, index=None)
    try:
        os.remove(_CACHE)
    except OSError:
        pass
