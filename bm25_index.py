"""BM25 词面检索索引（中文按字符二元组切分，无需分词库）

为什么需要它：
1. 向量模型 nomic-embed-text 是英文模型，对中文几乎没区分度——无关问题也能拿到 0.78
   的余弦相似度，而真正相关的条文正文只有 0.70~0.75。只靠向量必然「该召回的召不回、
   不该召回的乱进候选」。
2. 原先用 Chroma 的 where_document $contains 做词面补召，实测每次约 160ms 且只能匹配
   单个词组。BM25 一次查询约 20~40ms，还能按词频/罕见度加权，召回质量高得多。

设计：
- 只建一份内存倒排（term -> 文档下标 + 词频），永久缓存在 data/bm25.pkl，
  以库内块数为版本号，库内容变化时自动重建；
- 查询用 numpy bincount 累加分数，几万块规模下 20ms 内完成。
"""
import os
import pickle
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

import config

K1, B = 1.5, 0.75
# 极常见二元组（"建筑""规范"）在几万块里都出现，做召回没有区分度，直接跳过
MAX_POSTING_DOCS = 40000

_CACHE_NAME = "bm25.pkl"
_STATE = {"count": -1, "index": None}

_ASCII_RE = re.compile(r"[A-Za-z][A-Za-z0-9+\-]*")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> List[str]:
    """CJK 二元组 + 英文/数字词。中文不依赖分词库，二元组在规范文本上足够稳。"""
    out: List[str] = []
    for m in _ASCII_RE.finditer(text):
        w = m.group().lower()
        if len(w) >= 2:
            out.append(w)
    for m in _CJK_RE.finditer(text):
        run = m.group()
        if len(run) == 1:
            out.append(run)
            continue
        for i in range(len(run) - 1):
            out.append(run[i : i + 2])
    return out


def _cache_path() -> str:
    return os.path.join(os.path.dirname(config.CHROMA_DIR), _CACHE_NAME)


class BM25Index:
    def __init__(
        self,
        postings: Dict[str, Tuple[np.ndarray, np.ndarray]],
        doc_len,
        avgdl,
        n_docs,
        ids: List[str] = None,
        docs: List[str] = None,
        metas: List[dict] = None,
    ):
        self.postings = postings
        self.doc_len = doc_len
        self.avgdl = max(1.0, float(avgdl))
        self.n_docs = n_docs
        # 顺带缓存全文与元数据：词面召回命中后直接取用，省掉回查 Chroma（约 160ms/次）
        self.ids = ids or []
        self.docs = docs or []
        self.metas = metas or []
        self.id_to_pos = {c: i for i, c in enumerate(self.ids)} if self.ids else {}

    def score(self, query: str, top_n: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (文档下标, 归一化分数)，按分数降序，最多 top_n 条。"""
        terms = list(dict.fromkeys(tokenize(query)))
        if not terms or not self.postings:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        acc = np.zeros(self.n_docs, dtype=np.float32)
        used = 0
        for t in terms:
            item = self.postings.get(t)
            if item is None:
                continue
            idx, tf = item
            idf = self._idf(len(idx))
            dl = self.doc_len[idx]
            denom = tf + K1 * (1.0 - B + B * dl / self.avgdl)
            w = idf * (tf * (K1 + 1.0)) / denom
            acc += np.bincount(idx, weights=w, minlength=self.n_docs)[: self.n_docs].astype(
                np.float32
            )
            used += 1
        if not used:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        k = min(top_n, self.n_docs)
        top = np.argpartition(-acc, k - 1)[:k]
        top = top[np.argsort(-acc[top])]
        return top.astype(np.int64), acc[top]

    def _idf(self, df: int) -> float:
        n = self.n_docs
        return float(np.log(1.0 + (n - df + 0.5) / (df + 0.5)))

    def score_at(self, query: str, positions) -> np.ndarray:
        """对指定文档下标批量算 BM25 分（用于重排阶段给候选打分）。
        与 score() 的区别是只算给定位置，几万块规模下约 2~5ms。"""
        pos = np.asarray(positions, dtype=np.int32)
        acc = np.zeros(len(pos), dtype=np.float32)
        valid = pos >= 0
        if not valid.any():
            return acc
        for t in dict.fromkeys(tokenize(query)):
            item = self.postings.get(t)
            if item is None:
                continue
            idx, tf = item
            k = np.searchsorted(idx, pos).clip(0, len(idx) - 1)
            hit = (idx[k] == pos) & valid
            if not hit.any():
                continue
            idf = self._idf(len(idx))
            tfs = tf[k]
            dl = self.doc_len[pos]
            denom = tfs + K1 * (1.0 - B + B * dl / self.avgdl) + 1e-6
            acc += np.where(hit, idf * (tfs * (K1 + 1.0)) / denom, 0.0).astype(np.float32)
        return acc


def _build(col) -> BM25Index:
    """全量扫描建索引（只在首次或库内容变化时执行，约 60~100s）"""
    from vector_store import _scan_all

    postings: Dict[str, Tuple[List[int], List[int]]] = {}
    doc_len = []
    all_ids: List[str] = []
    all_docs: List[str] = []
    all_metas: List[dict] = []
    n = 0
    for res in _scan_all(col, include=("documents", "metadatas"), batch=2000):
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
        ids = res.get("ids") or []
        for cid, text, meta in zip(ids, docs, metas):
            heading = (meta or {}).get("heading") or ""
            full = f"{heading}\n{text}" if heading else (text or "")
            toks = tokenize(full)
            doc_len.append(len(toks))
            all_ids.append(cid)
            all_docs.append(text or "")
            all_metas.append(meta or {})
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

    packed: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for t, (idxs, tfs) in postings.items():
        if len(idxs) > MAX_POSTING_DOCS:
            continue
        packed[t] = (
            np.asarray(idxs, dtype=np.int32),
            np.asarray(tfs, dtype=np.float32),
        )
    dl = np.asarray(doc_len, dtype=np.float32)
    return BM25Index(
        packed, dl, float(dl.mean()) if len(dl) else 0.0, n, all_ids, all_docs, all_metas
    )


def get_index(col=None) -> Optional[BM25Index]:
    """获取 BM25 索引（进程内单例 + 磁盘缓存，按库内块数校验版本）"""
    from vector_store import get_count, get_vectorstore

    vs = get_vectorstore()
    col = col or vs._collection
    n = get_count(vs)
    if _STATE["count"] == n and _STATE["index"] is not None:
        return _STATE["index"]

    path = _cache_path()
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if obj.get("count") == n:
            idx = BM25Index(
                obj["postings"],
                obj["doc_len"],
                obj["avgdl"],
                obj["n_docs"],
                obj.get("ids"),
                obj.get("docs"),
                obj.get("metas"),
            )
            _STATE.update(count=n, index=idx)
            return idx
    except Exception:
        pass

    idx = _build(col)
    _STATE.update(count=n, index=idx)
    try:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "count": n,
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
    """入库/删除后调用：清内存并删磁盘缓存（块数可能与重建前相同，只靠 count 校验会误用旧索引）"""
    _STATE.update(count=-1, index=None)
    try:
        os.remove(_cache_path())
    except OSError:
        pass
