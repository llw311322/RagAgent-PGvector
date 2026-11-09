"""混合检索（LangChain 原生组件串联）

链路：
  PGVectorRetriever ─┐
                     ├─ EnsembleRetriever(加权 RRF) ─ RerankCompressor(ONNX 精排) ─ 结果
  PGKeywordRetriever ┘

设计取舍：
- 融合用 LangChain 的 EnsembleRetriever（加权 RRF），它只依赖各路排名，
  不需要两路分数量纲一致——向量余弦与 BM25 分本来就没有可比性。
- 融合后的绝对相关度由 bge-reranker 给出（0~1），据此设阈值实现「查不到就拒答」。
- 元数据过滤：标量字段（domain/std_class/valid）下推给 PG 用 SQL 过滤，
  数组字段（buildings）在 Python 侧后过滤——PGVector 的 filter 不支持 jsonb 数组包含。
"""
import re
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from pydantic import ConfigDict, Field

import config
import rerank
from bm25_index import tokenize
from lc import bm25, store


# ---------------- 元数据过滤 ----------------
def pg_filter(filters: Optional[dict]) -> Optional[dict]:
    """把界面筛选条件转成 PGVector 的 jsonb 过滤表达式"""
    if not filters:
        return None
    conds = {}
    if filters.get("domain"):
        conds["domain"] = {"$eq": filters["domain"]}
    if filters.get("std_class"):
        conds["std_class"] = {"$eq": filters["std_class"]}
    if filters.get("valid") is not None:
        conds["valid"] = {"$eq": bool(filters["valid"])}
    return conds or None


def meta_match(meta: dict, filters: Optional[dict]) -> bool:
    """数组字段（buildings）在 Python 侧后过滤；标量字段已下推 SQL，这里再兜一次底"""
    if not filters:
        return True
    b = filters.get("building")
    if b:
        vals = meta.get("buildings") or []
        if isinstance(vals, str):
            vals = [vals]
        if b not in vals and "通用" not in vals:
            return False
    for key in ("domain", "std_class"):
        v = filters.get(key)
        if v and (meta.get(key) or "") != v:
            return False
    if filters.get("valid") is not None:
        if bool(meta.get("valid", True)) != bool(filters["valid"]):
            return False
    return True


_GENERIC_WORDS = (
    "规范", "标准", "条文", "是什么", "什么", "要求", "规定", "多少", "怎么", "如何",
    "哪些", "请问", "一下", "分别", "具体", "内容", "第", "条", "的", "是", "有",
)


def _relevance_terms(text: str) -> set:
    """去掉「规范/标准/是什么」这类泛词后再取词：它们几乎命中所有条文，不构成相关性证据"""
    cleaned = text
    for w in _GENERIC_WORDS:
        cleaned = cleaned.replace(w, " ")
    terms = set(tokenize(cleaned))
    return terms or set(tokenize(text))


def _kw_overlap(query: str, text: str) -> int:
    """查询词命中数：用来拦截「向量很像但完全没提这个词」的片段"""
    qt = _relevance_terms(query)
    if not qt:
        return 0
    tt = set(tokenize(text))
    return len(qt & tt)


# ---------------- 向量召回 ----------------
class PGVectorRetriever(BaseRetriever):
    """PGVector 语义召回；相似度写回 metadata['sim'] 供展示与兜底判断"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    store: Any = None
    k: int = Field(default=config.LC_VECTOR_K)
    search_filter: Optional[dict] = None
    filters: Optional[dict] = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        st = self.store or store.get_store()
        hits = st.similarity_search_with_score(
            query, k=self.k, filter=pg_filter(self.filters)
        )
        out: List[Document] = []
        for doc, dist in hits:
            if not meta_match(doc.metadata, self.filters):
                continue
            doc.metadata["sim"] = round(max(0.0, 1.0 - float(dist)), 4)
            doc.metadata["retriever"] = "vector"
            out.append(doc)
        return out


# ---------------- 词面召回 ----------------
class PGKeywordRetriever(BaseRetriever):
    """BM25 词面召回：条文号、术语、数字（如 5.2.2、3.00h）靠它才能精准命中"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    k: int = Field(default=config.LC_BM25_K)
    filters: Optional[dict] = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        idx = bm25.get_index()
        if idx is None:
            return []
        # 多取一些再过滤，避免筛选条件把候选全刷掉
        positions, scores = idx.score(query, top_n=min(self.k * 4, 200))
        if len(positions) == 0:
            return []
        top = float(scores[0]) or 1.0
        out: List[Document] = []
        for pos, sc in zip(positions, scores):
            meta = idx.metas[int(pos)] or {}
            if not meta_match(meta, self.filters):
                continue
            meta = dict(meta)
            meta["bm25"] = round(float(sc) / top, 4)
            meta["retriever"] = "bm25"
            out.append(Document(page_content=idx.docs[int(pos)], metadata=meta))
            if len(out) >= self.k:
                break
        return out


# ---------------- 精排压缩 ----------------
class RerankCompressor(BaseDocumentCompressor):
    """bge-reranker（ONNX）交叉编码精排，输出 0~1 相关性并设下限

    ContextualCompressionRetriever 会把融合后的候选交给它，
    返回的文档既是最终上下文，也是「回答依据」的来源。
    """

    top_n: int = Field(default=config.LC_RERANK_TOP_N)
    min_score: float = Field(default=config.LC_RERANK_MIN)
    batch_size: int = Field(default=8)
    # 只对融合后的头部候选精排：42 条全排要 6~7 秒，16 条约 1.3 秒，
    # 而 RRF 已经把明显不相关的排到后面，多排的部分对结果没有影响
    max_candidates: int = Field(default=config.LC_RERANK_MAX_CAND)
    query_for_gate: str = ""

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Any] = None,
    ) -> Sequence[Document]:
        if not documents:
            return []
        documents = list(documents)[: self.max_candidates]
        texts = []
        for d in documents:
            heading = (d.metadata or {}).get("heading") or ""
            texts.append(f"{heading}\n{d.page_content}" if heading else d.page_content)

        scores = rerank.score(query, texts, batch_size=self.batch_size)
        if scores is None or len(scores) != len(documents):
            # 精排不可用时退化为融合顺序，但不放行完全无词面交集的片段
            kept = [
                d for d in documents
                if _kw_overlap(query, d.page_content) > 0 or d.metadata.get("clause_no")
            ]
            return list(kept)[: self.top_n]

        ranked: List[tuple] = []
        for d, s, txt in zip(documents, scores, texts):
            d.metadata["rerank"] = round(float(s), 4)
            # 词面交集算在「标题路径 + 正文」上：条文号常常只在标题里
            d.metadata["kw_hit"] = _kw_overlap(query, txt)
            ranked.append((d, float(s)))
        ranked.sort(key=lambda x: -x[1])

        out: List[Document] = []
        for d, s in ranked:
            if s < self.min_score:
                continue
            # 双保险：精排分不高、又完全没有词面交集 → 判为无关
            # （0.85 是实测分界：相关问题普遍 0.95+，无关问题最高 0.61）
            if s < config.LC_KEYWORD_GATE and not d.metadata.get("kw_hit") and not d.metadata.get("exact"):
                continue
            out.append(d)
            if len(out) >= self.top_n:
                break
        return out


# ---------------- 条文号精确兜底 ----------------
_CLAUSE_RE = re.compile(r"(?<![\d.])(\d{1,2}(?:\.\d{1,3}){1,2})(?![\d.])")
_CODE_RE = re.compile(r"\b([A-Z]{2,3}\s?\d{3,5})\b")


def _clauses_in(query: str) -> List[str]:
    """从问题里抽取条文号。排除「2.00h」「1.5m」这类数值：段位为 00 或整体过短的不要"""
    out: List[str] = []
    for m in _CLAUSE_RE.findall(query):
        parts = m.split(".")
        if parts[-1] == "00":          # 2.00h 这类数值不是条文号
            continue
        if int(parts[0]) > 30:         # 规范条号首段不会太大
            continue
        if m not in out:
            out.append(m)
    return out


class ClauseAwareRetriever(BaseRetriever):
    """在混合检索结果上叠加「条文号精确召回」

    为什么需要：语义检索分不清「防火通用规范 5.2.2」和「汽车库规范 5.2.2」，
    而用户点名条文号时，最该给的恰恰是那一条。这里先按元数据精确捞，
    再与语义结果合并去重，精确命中排在前面（精排分记 0.95，不伪造 1.0）。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base: Any = None
    k: int = Field(default=config.LC_TOP_K)
    use_exact: bool = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        docs = list(self.base.invoke(query)) if self.base is not None else []
        if not self.use_exact:
            return docs[: self.k]

        clauses = _clauses_in(query)[:2]
        if not clauses:
            return docs[: self.k]
        codes = _CODE_RE.findall(query.upper())

        # 去掉条文号本身再算词面交集：「5.2.2」不能作为相关性的证据，
        # 否则入侵报警、无障碍、施工升降机里的 5.2.2 会和防火规范的 5.2.2 一样高分
        text_q = _CLAUSE_RE.sub(" ", query)
        q_terms = _relevance_terms(text_q)
        exact: List[Document] = []
        for c in clauses:
            cands = []
            for meta, body in store.find_clause(c, codes[0] if codes else "", limit=200):
                m = dict(meta or {})
                # 只要真正同名的那一条：库里的 ILIKE 是为了容错（如「5.2.2」写成「5.2.2 条」），
                # 但「6.6.3」这种近似号不该被当成 6.3.1 的依据
                if (m.get("clause_no") or "") != c and (m.get("article") or "") != c:
                    continue
                hay = " ".join(str(m.get(k) or "") for k in ("source", "standard_name", "heading")) + " " + (body or "")
                kw = len(q_terms & set(tokenize(hay)))
                # 排序以权威性为主、词面为辅：只看词频会让正文里恰好写了「防火规范」的
                # 汽车库条文压过强制性通用规范本身
                rank = 6 if "强制性通用规范" in (m.get("std_class") or "") else 0
                rank += 3 if m.get("is_mandatory") else 0
                rank += 2 if (m.get("source_type") or "条文正文") == "条文正文" else 0
                rank += kw
                cands.append((rank, kw, m, body))
            if not cands:
                continue
            cands.sort(key=lambda x: -x[0])
            best_kw = max(x[1] for x in cands)
            if best_kw > 0:
                # 先剔除「一个实词都没命中」的同名条文，再取前 3——
                # 顺序反了会把整批候选都跳掉（曾经踩过）
                cands = [x for x in cands if x[1] > 0]
            for rank, kw, m, body in cands[:3]:
                m["exact"] = True
                m["retriever"] = "clause"
                m["kw_hit"] = kw
                m["rerank"] = max(float(m.get("rerank") or 0), 0.9)
                exact.append(Document(page_content=body or "", metadata=m))

        merged: List[Document] = []
        seen = set()
        for d in exact + docs:
            key = (d.metadata.get("source"), d.metadata.get("start_line"), d.metadata.get("clause_no"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(d)
        return merged[: self.k]


# ---------------- 组装 ----------------
def build_retriever(filters: Optional[dict] = None, k: Optional[int] = None) -> ClauseAwareRetriever:
    vec = PGVectorRetriever(
        store=store.get_store(), k=config.LC_VECTOR_K, filters=filters
    )
    kw = PGKeywordRetriever(k=config.LC_BM25_K, filters=filters)
    ensemble = EnsembleRetriever(
        retrievers=[vec, kw],
        weights=list(config.LC_ENSEMBLE_WEIGHTS),
        c=60,   # RRF 常数，60 是经验值：压低单路第一名的绝对优势，让两路都能贡献
    )
    compressed = ContextualCompressionRetriever(
        base_compressor=RerankCompressor(top_n=config.LC_RERANK_TOP_N),
        base_retriever=ensemble,
    )
    return ClauseAwareRetriever(base=compressed, k=k or config.LC_TOP_K)


def retrieve(question: str, filters: Optional[dict] = None, k: Optional[int] = None) -> List[Document]:
    docs = build_retriever(filters, k).invoke(question)
    return list(docs)[: (k or config.LC_TOP_K)]
