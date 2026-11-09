"""语义检索模块：向量召回 + 条文号精确召回 + 词面重排（hybrid）+ 阈值过滤

设计要点（2026-11 复核优化）：
1. 速度：向量库与向量模型进程内单例（构造开销实测 1.7~1.9s/次，必须复用）；整条链路
   只做 1 次查询向量化，直接调 Chroma 底层 collection.query。热查询约 15ms。
2. 准确度（三重增强）：
   - 条文号精确召回：问题带条文号（如 "5.2.2"）时用 $contains 精确拉取，含条文标题行
     （某行以编号开头）的块加权置顶——条文编号对向量模型是噪音，纯向量检索会漏；
   - 词面重排：nomic 对长规范文本的向量区分度有限（实测无关问题相似度也高达 0.67+），
     用查询词的 CJK 二元组命中率重排候选，罕见关键词（如「电动汽车充电站」）强信号；
   - 阈值作用于融合分（0.55*向量相似度 + 0.45*词面命中率），实测能同时拦住无关问题
     （词面 0 命中）并保留相关问题。
3. 同文档行号重叠的片段去重，避免 top-k 被同一条文的相邻块占满。
"""
import math
import re
from typing import List, Tuple

from langchain_core.documents import Document

import bm25_index
import config
import rerank
from vector_store import get_embeddings, get_stats, get_title_index, get_vectorstore

# 条文号：如 5.2.2 / 8.1.10 / 12.3.4
ARTICLE_RE = re.compile(r"\d{1,2}\.\d{1,2}\.\d{1,3}")
# 融合权重：向量相似度 + BM25 词面分
# 向量权重压到 0.40 是因为 nomic 对中文的区分度太差（无关文本也有 0.75+ 相似度）
W_SIM, W_BM = 0.40, 0.60
# BM25 原始分达到该值即视为词面完全相关（实测相关问题 20~65；取 55 避免短块轻易饱和——
# 饱和后大量块同分封顶，标题命中的正确条文反而会被短句噪声压住）
BM25_FULL_SCORE = 55.0
# 查询与条文标题词面重合 ≥0.8 时给保底分：标题几乎照抄查询 = 该条文就是标准答案，
# 不能让「耐火等级不低于二级的灭火器材间」这种短句噪声压过「5.2.2 下列工业建筑的耐火等级不应低于二级」
TITLE_EXACT_FLOOR, TITLE_MIN_SCORE = 0.80, 0.85
W_KW, ARTICLE_MIN_SCORE = 0.45, 0.70
# 片段本身就是被问条文（metadata.article 相等）时的保底分
ARTICLE_BODY_SCORE = 0.72
# 「标准名 + 条文号」双重命中时的保底分（用户已把范围限定到某部规范，该块就是标准答案）
TARGET_MIN_SCORE = 0.78
# 标题匹配加分：问题常是条文标题的改写（「工业建筑的耐火等级不应低于二级的情形」
# 对应标题「5.2.2 下列工业建筑的耐火等级不应低于二级」），标题相似度高是强信号
W_TITLE, TITLE_FLOOR = 0.35, 0.45
# 文档名匹配加分（问题里提到的规范简称）与「同一文档最多命中几块」
W_SRC, MAX_PER_SOURCE = 0.30, 2
# BM25 召回规模与最低原始分（实测相关问题 20~65，无关问题 ≤12）
BM25_TOP_N, BM25_MIN_SCORE = 80, 6.0
# 规范权威性加分（通用规范 > 国标 > 行标；图示/解读/征求意见稿降权）
W_AUTH = 1.0
# OCR 乱码块降权
GARBLE_RATIO, GARBLE_PENALTY = 0.22, 0.15

# 工程口语简称 → 真正的规范名关键词。
# 全库 200 部规范里名字含「防火规范」的有一堆（汽车库、装修、农村…），
# 用户只说「防火规范」时指的是通用规范/建规，靠字面匹配一定会错。
ALIASES = {
    "防火规范": ("建筑防火通用规范", "建筑设计防火规范"),
    "建规": ("建筑设计防火规范",),
    "高规": ("高层民用建筑设计防火规范", "民用建筑通用规范"),
    "通则": ("民用建筑设计通则", "民用建筑通用规范"),
    "住宅规范": ("住宅建筑规范", "住宅设计规范"),
    "车库规范": ("汽车库、修车库、停车场设计防火规范",),
    "中小学规范": ("中小学校设计规范",),
    "无障碍规范": ("无障碍设计规范", "建筑与市政工程无障碍通用规范"),
}

# 非权威来源：图示集、条文说明、征求意见稿、解读——可作为补充，但不应压过正文规范
_NOISE_KEYS = ("图示", "图集", "条文说明", "征求意见", "局部修订", "解读", "指南", "培训", "强制性条文")


# 含这些字的二元组多是虚词/疑问词（"怎么""什么""哪些"），
# 参与词面匹配只会让「今天天气怎么样」这类无关问题蒙到分数，必须排除。
_STOP_CHARS = set("的了是和与及对在中为把被给让从到而并且或吗呢吧啊呀么怎啥这那")


def _query_terms(query: str) -> List[str]:
    """查询词项：连续字母数字串 + CJK 二元组（用于词面匹配与标题召回）"""
    q = query.strip()
    if not q:
        return []
    terms = set(re.findall(r"[A-Za-z0-9]+(?:\.\d+)*", q))
    cjk = re.sub(r"[^\u4e00-\u9fff]", " ", q)
    for run in cjk.split():
        for i in range(len(run) - 1):
            bg = run[i : i + 2]
            if bg[0] in _STOP_CHARS or bg[1] in _STOP_CHARS:
                continue
            terms.add(bg)
    return list(terms)


def _kw_ratio(query: str, text: str) -> float:
    """查询词的 CJK 二元组 + 连续字母数字串 在文本中的命中率（0~1）"""
    terms = _query_terms(query)
    if not terms:
        return 0.0
    hit = sum(1 for t in terms if t in text)
    return hit / len(terms)


def _cosine(a, b) -> float:
    """余弦相似度（不引 numpy，避免每组候选都做矩阵运算）"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


def _row_to_doc(doc_text: str, meta: dict) -> Document:
    d = Document(page_content=doc_text, metadata=meta or {})
    return d


def _search_text(d: Document) -> str:
    """参与词面匹配的文本：标题路径 + 正文（标题路径含章/节/条文名，是重要的词面信号）"""
    h = (d.metadata or {}).get("heading") or ""
    return f"{h}\n{d.page_content}" if h else d.page_content


def _rerank_text(d: Document) -> str:
    """送进 Cross-Encoder 的文本。条文号对精排是有效信号（问「5.2.2」时它直接决定命中），
    但要拼在标题路径后面，保证「5.2.2 下列工业建筑…」这种完整语义能被模型读到。"""
    m = d.metadata or {}
    head = m.get("heading") or ""
    art = m.get("clause_no") or m.get("article") or ""
    prefix = f"{art} " if art and not head.endswith(art) else ""
    body = d.page_content
    return f"{head}\n{prefix}{body}" if head else f"{prefix}{body}"


def _query_collection(
    query_embedding: List[float],
    n_results: int,
    where_document: dict = None,
    where: dict = None,
) -> List[Tuple[Document, float]]:
    """直接调 Chroma 底层 query，返回 [(Document, 余弦相似度)]"""
    vs = get_vectorstore()
    col = vs._collection
    kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=max(1, n_results),
        include=["documents", "metadatas", "distances"],
    )
    if where_document:
        kwargs["where_document"] = where_document
    if where:
        kwargs["where"] = where
    res = col.query(**kwargs)

    out: List[Tuple[Document, float]] = []
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    for cid, text, meta, dist in zip(ids, docs, metas, dists):
        sim = 1.0 - float(dist)
        d = _row_to_doc(text, meta)
        d.metadata["id"] = cid
        out.append((d, round(sim, 4)))
    return out


def _doc_from_bm25(bm, p: int) -> Document:
    """BM25 索引里缓存的全文/元数据直接拼成 Document，避免回查 Chroma"""
    d = Document(page_content=bm.docs[p], metadata=dict(bm.metas[p]))
    d.metadata["id"] = bm.ids[p]
    return d


def _embedding_map(ids: List[str]) -> dict:
    """批量取若干块的向量。BM25 命中的块没有现成相似度，需要补算，
    一次 get 80 条约 25ms，远低于逐个查询。"""
    if not ids:
        return {}
    res = get_vectorstore()._collection.get(ids=ids, include=["embeddings"])
    embs = res.get("embeddings")
    if embs is None:
        return {}
    return dict(zip(res.get("ids") or [], embs))


def _title_seg(meta: dict) -> str:
    """标题路径的最后一段（即该片段所属条文的标题本身）"""
    h = (meta or {}).get("heading") or ""
    return h.split(">")[-1].strip() if h else ""


def _is_article(d: Document, article_set: set) -> bool:
    """该片段是否就是某条文正文（结构化分块后 metadata.article 即条文号；正文/标题行首兜底）"""
    if not article_set:
        return False
    meta = d.metadata or {}
    if meta.get("article") in article_set:
        return True
    head = meta.get("heading") or ""
    for a in article_set:
        if re.search(rf"(?<![\d.]){re.escape(a)}(?![\d.])", head):
            return True
        for ln in d.page_content.splitlines():
            s = ln.lstrip()
            if s.startswith(a) and (len(s) == len(a) or not (s[len(a)].isdigit() or s[len(a)] == ".")):
                return True
    return False


def _is_exact_article(d: Document, article_set: set) -> bool:
    """片段本身就是该条文（metadata.article 精确相等，结构化分块保证一条一块）"""
    return bool(article_set) and (d.metadata or {}).get("article") in article_set


def _src_key(name: str) -> str:
    """文档名归一化（只留中文）：同一部规范的不同副本/OCR 版会被视为同一来源，
    避免「GB50067-2014：汽车库…」和「01-05-05汽车库…GB 50067-2014」同时占位。"""
    return re.sub(r"[^\u4e00-\u9fff]", "", name or "") or (name or "")


# 各标准大类的权威性权重（重打标后块上带 std_class 字段，直接查表）
STD_CLASS_AUTH = {
    "国家标准-强制性通用规范": 0.15,
    "国家标准": 0.05,
    "政策文件": 0.03,
    "行业标准": 0.02,
    "地方标准": 0.02,
    "其他": 0.0,
    "参考资料": -0.10,
}


def _authority(source: str, std_class: str = "") -> float:
    """规范权威性权重：强制性通用规范 > 国标 > 行标/地标；图示/解读/征求意见稿为负。"""
    if std_class:
        return STD_CLASS_AUTH.get(std_class, 0.0)
    s = source or ""
    w = 0.0
    if "通用规范" in s:
        w += 0.10
    if re.search(r"GB\s*/?T?\s*55\d{3}", s):
        w += 0.05
    elif re.search(r"GB\s*/?T?\s*50\d{3}", s):
        w += 0.03
    elif "GB" in s:
        w += 0.02
    elif re.search(r"JGJ|CJJ|DG|DB\d", s):
        w += 0.005
    if any(k in s for k in _NOISE_KEYS):
        w -= 0.10
    return w


def _garble_ratio(text: str) -> float:
    """乱码比例：OCR 扫出来的图示集正文充斥「l 蔡|瑞均町、」「N / -- ""」之类的碎字符，
    这类块即使向量分数高也毫无价值，必须降权。"""
    if not text:
        return 1.0
    good = 0
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff" or ch.isalnum():
            good += 1
        elif ch in "，。；：、（）《》%．,.;:()-—/[]":
            good += 1
    return 1.0 - good / len(text)


def _overlaps(m1: dict, m2: dict) -> bool:
    """同一文档的两个片段行号范围是否重叠"""
    if not (m1.get("start_line") and m2.get("start_line")):
        return False
    if m1.get("source") != m2.get("source"):
        return False
    return not (m1["end_line"] < m2["start_line"] or m2["end_line"] < m1["start_line"])


def _match_sources(query: str) -> List[str]:
    """问题里点名的规范（按文档名与问题的词面重合度 + 口语简称别名 + 权威性判断）。
    例：「防火规范第5.2.2条」→ 命中「建筑防火通用规范」「建筑设计防火规范」，
    而不是字面同样含「防火规范」的汽车库/装修规范。用于把条文号限定在用户真正想问的那部规范里。"""
    sources = get_stats().get("sources", [])
    if not sources:
        return []
    terms = _query_terms(query)
    if len(terms) < 3:
        return []
    # 口语简称命中时，目标文档直接强加权——字面相似度不再作为主信号，
    # 否则「防火规范」会把《汽车库、修车库、停车场设计防火规范》顶到第一位
    targets = [v for k, v in ALIASES.items() if k in query]
    scored = []
    for s in sources:
        kw = sum(1 for t in terms if t in s) / len(terms)
        bonus = 0.0
        for group in targets:
            if any(t in s for t in group):
                bonus = 0.45
                break
        if kw < 0.15 and bonus == 0.0:
            continue
        scored.append((s, kw + bonus + 0.4 * _authority(s)))
    scored.sort(key=lambda x: (-x[1], len(x[0])))
    return [s for s, _ in scored[:5]]


def _match_filters(meta: dict, filters: dict) -> bool:
    """结构化标签过滤：domain / building / std_class 任一命中失败即丢弃。
    building 是多标签列表，按子串匹配（"地库"→命中"地下"）。"""
    if not filters:
        return True
    d = (filters.get("domain") or "").strip()
    if d and meta.get("domain") != d:
        return False
    sc = (filters.get("std_class") or "").strip()
    if sc and meta.get("std_class", "").find(sc) < 0:
        return False
    b = (filters.get("building") or "").strip()
    if b:
        bl = meta.get("buildings") or []
        if not (isinstance(bl, list) and any(b in str(x) for x in bl)):
            return False
    code = (filters.get("standard_code") or "").strip()
    if code:
        # "GB55037" 这类不带年份的前缀也要能匹配到 "GB 55037-2022"
        if re.sub(r"\s+", "", code) not in re.sub(r"\s+", "", meta.get("standard_code", "")):
            return False
    st = (filters.get("source_type") or "").strip()
    if st and meta.get("source_type") != st:
        return False
    if filters.get("mandatory") and not meta.get("is_mandatory"):
        return False
    if filters.get("valid") and meta.get("valid") is False:
        return False
    return True


def retrieve(
    query: str,
    top_k: int = None,
    threshold: float = None,
    filters: dict = None,
) -> List[Tuple[Document, float]]:
    """
    检索与问题最相关的知识片段。

    返回: [(Document, 融合分数)]，按相关度从高到低，已过滤低于阈值的无关片段。
    分数为融合分 = 0.55*余弦相似度 + 0.45*词面命中率（0~1），越高越相关。
    """
    top_k = top_k or config.TOP_K
    threshold = config.SIMILARITY_THRESHOLD if threshold is None else threshold
    candidate_k = max(top_k * 5, 24)

    # 1) 查询向量化（整条链路只做这一次）
    emb = get_embeddings().embed_query(query)

    candidates: dict = {}      # id -> (doc, sim)，sim 为 None 表示待补算
    bm = bm25_index.get_index()

    # 2) 向量召回（一次拿足候选，减少后续补召次数）
    for d, s in _query_collection(emb, candidate_k):
        candidates[d.metadata["id"]] = (d, s)

    # 3) BM25 词面召回：替代原先的 where_document $contains（160ms/次，且只能匹配单个词组）。
    #    BM25 按词频与词罕见度加权，一次约 10ms，同时解决「条文列表被切开后向量分过低」
    #    与「OCR 正文用语不规范」两类漏召回。
    if bm is not None and bm.ids:
        pos, scores = bm.score(query, BM25_TOP_N)
        for p, sc in zip(pos, scores):
            if sc < BM25_MIN_SCORE:
                break
            d = _doc_from_bm25(bm, int(p))
            cid = d.metadata["id"]
            if cid not in candidates:
                candidates[cid] = (d, None)

    # 4) 条文号精确召回：直接在内存元数据里过滤（约 15ms），不再发 Chroma where 查询。
    articles = list(dict.fromkeys(ARTICLE_RE.findall(query)))
    article_set = set(articles)
    exact_ids = set()
    targeted_ids = set()      # 「标准名 + 条文号」双重命中的块，确定性证据
    if articles and bm is not None and bm.metas:
        src_set = set(_match_sources(query))
        for i, m in enumerate(bm.metas):
            if (m or {}).get("article") in article_set:
                d = _doc_from_bm25(bm, i)
                cid = d.metadata["id"]
                exact_ids.add(cid)
                if m.get("source") in src_set:
                    targeted_ids.add(cid)
                if cid not in candidates:
                    candidates[cid] = (d, None)

    # 3d) 标题召回：问题与条文标题词面重合度高时补召。
    #     释义式提问（「工业建筑的耐火等级不应低于二级的情形」）向量相似度只有 0.70、
    #     进不了 top15，但它几乎就是条文标题的改写，用标题倒排索引能稳定召回。
    tid = get_title_index()
    terms = _query_terms(query)
    if tid["items"] and terms:
        cnt: dict = {}
        for t in terms:
            for gi in tid["postings"].get(t, ()):
                cnt[gi] = cnt.get(gi, 0) + 1
        picked = []
        for gi, c in sorted(cnt.items(), key=lambda x: -x[1])[:24]:
            if c / len(terms) < 0.55:
                continue
            cid = tid["items"][gi][0]
            if cid not in candidates:
                picked.append(cid)
        if picked:
            rows = get_vectorstore()._collection.get(
                ids=picked[:12], include=["documents", "metadatas", "embeddings"]
            )
            row_docs = rows.get("documents") or []
            row_metas = rows.get("metadatas") or []
            row_embs = rows.get("embeddings")
            row_embs = [] if row_embs is None else row_embs
            for cid, text, meta, emb_v in zip(rows.get("ids") or [], row_docs, row_metas, row_embs):
                d = _row_to_doc(text, meta)
                d.metadata["id"] = cid
                candidates[cid] = (d, round(_cosine(emb, emb_v), 4))

    # 5) 补齐缺失的相似度：BM25 / 条文号内存过滤 / 标题召回进来的块没有现成分数，
    #    一次批量取向量后统算余弦（约 40ms），避免为它们逐个发起查询。
    need = [cid for cid, (d, s) in candidates.items() if s is None]
    if need:
        emap = _embedding_map(need[:200])
        for cid in need:
            e = emap.get(cid)
            if e is None:
                candidates.pop(cid, None)
            else:
                d = candidates[cid][0]
                candidates[cid] = (d, round(_cosine(emb, e), 4))

    # 6) 融合重排：向量相似度 + BM25 词面分 + 标题匹配 + 权威性
    #    词面项用 BM25 而不是「二元组命中率」：BM25 自带长度归一化与 IDF 加权，
    #    而命中率会让短块天然占优（汽车库条文一句话就能 100% 命中，
    #    把真正讲充电站的完整条文压下去）。
    cids = list(candidates.keys())
    bm_raw = None
    if bm is not None and bm.id_to_pos:
        bm_raw = bm.score_at(query, [bm.id_to_pos.get(c, -1) for c in cids])

    scored: List[Tuple[Document, float, float, float]] = []   # (片段, 融合分, BM25, 保底分)
    for i, cid in enumerate(cids):
        d, sim = candidates[cid]
        m_src = d.metadata.get("source", "")
        # 结构化标签过滤（domain/building/std_class/条文类型），可用于「只搜防火规范」这类限定
        if not _match_filters(d.metadata, filters):
            continue
        kw = _kw_ratio(query, _search_text(d))
        title_kw = _kw_ratio(query, _title_seg(d.metadata))
        bm_norm = min(1.0, float(bm_raw[i]) / BM25_FULL_SCORE) if bm_raw is not None else kw
        # 硬门槛：与问题零字面交集、也不是被点名的条文 → 判为无关。
        # 依据：nomic-embed-text 对中文几乎无区分度，连「今天天气怎么样」都能拿到 0.78 的
        # 相似度（真实相关条文也就 0.75 左右）。只靠阈值必然误召回，必须要求词面/标题有过交集。
        if kw <= 0.0 and title_kw <= 0.0 and cid not in exact_ids:
            continue
        final = W_SIM * sim + W_BM * bm_norm
        # 标题匹配加分：问题与该条文标题高度重合时（用户往往是照着条文名提问）显著提权
        if title_kw > TITLE_FLOOR:
            final += W_TITLE * (title_kw - TITLE_FLOOR)
        # 条文号保底分：仅当内容或标题也相关时才给。否则「5.2.2」这种编号在 200 部规范里
        # 各有同名条文（如汽车库防火规范也有 5.2.2），全部抬到 0.70 会把正确条文挤出 top-k。
        # 保底分单独记下来：精排之后还要再兜一次，避免被 Cross-Encoder 挤掉（见第 7 步）
        floor = 0.0
        if cid in targeted_ids:
            floor = TARGET_MIN_SCORE
        elif _is_exact_article(d, article_set):
            # 该块本身就是被问的条文（正文），给保底分——否则「引用了 5.2.2」的块词面分数更高会把它压下去
            floor = ARTICLE_BODY_SCORE
        elif cid in exact_ids and (kw >= 0.30 or title_kw >= 0.45):
            floor = ARTICLE_MIN_SCORE
        elif title_kw >= TITLE_EXACT_FLOOR:
            floor = TITLE_MIN_SCORE
        final = max(final, floor)
        # 文档名匹配加分：放在保底之后，否则所有同编号条文被抬成同分、无法区分。
        # 用户常把规范简称写进问题（「防火规范 5.2.2 条」），来源文件名是最好的消歧信号。
        src_kw = _kw_ratio(query, d.metadata.get("source", ""))
        if src_kw > 0.10:
            final += W_SRC * src_kw
        # 权威性：优先用打标字段 std_class，未打标的旧数据退回文件名正则。
        # 放在保底之后加，才能让「通用规范 5.2.2」压过「汽车库 5.2.2」这类同分情况。
        final += W_AUTH * _authority(m_src, d.metadata.get("std_class", ""))
        # OCR 扫出来的图示集正文多为「l 蔡|瑞均町、」这类碎字符，向量分再高也没有阅读价值
        if d.metadata.get("quality") == "noise" or _garble_ratio(d.page_content) > GARBLE_RATIO:
            final -= GARBLE_PENALTY
        # 条文正文 / 条文说明 / 编制依据 / 图示：后三类带条文号、词面分极高但不是条文本身。
        # 打标字段优先；旧数据无 source_type 时退回正文与标题路径的文本特征。
        st = d.metadata.get("source_type", "")
        if st in ("条文说明", "编制依据"):
            final -= 0.20 if st == "条文说明" else 0.15
        elif st == "图示":
            final -= 0.10
        else:
            t_head = d.page_content[:120]
            if "【编制依据】" in t_head or "条文说明" in t_head or t_head.count("GB") >= 5:
                final -= 0.15
            # 标题路径尾段是裸条文号（如「…> 5.2.2」）说明该块位于「条文说明」章节，
            # 真正的规范正文块尾段是「5.2.2 下列工业建筑的耐火等级不应低于二级：」。说明文字
            # 往往逐词复述条文（词面分极高），必须重罚，否则会冒充条文正文排到第一。
            if re.fullmatch(r"\d{1,2}(?:\.\d{1,3}){0,2}", _title_seg(d.metadata)):
                final -= 0.20
        # 作废/征求意见稿：轻微降权，只用于「正式版与草案都在库」时让正式版优先。
        # 不能扣多：本库《建筑防火通用规范》的正式版文件是残缺的，完整全文只有草案，
        # 扣 0.10 就会让「防火规范 5.2.2」输给《汽车库…防火规范》的 5.2.2（实测翻转）。
        if d.metadata.get("valid") is False:
            final -= 0.03
        # 不做 0.99 封顶：封顶会把「标题近乎照抄查询」的满分块和短句噪声拉成同分，
        # 次要排序键（BM25 分）天然偏向短块，正确条文反而被压住。允许 >1，展示时截到 1 即可。
        scored.append((d, round(final, 4), bm_norm, floor))

    scored.sort(key=lambda x: (-x[1], -x[2]))

    # 6) 阈值闸门：融合分决定「是否相关」，在精排之前过滤。
    #    精排只在相关片段内部定顺序，不能把无关片段救回来（否则防幻觉的第一道闸就失效了）。
    gated: List[Tuple[Document, float, float]] = [
        (d, s, fl) for d, s, _, fl in scored if s >= threshold
    ]
    if not gated:
        return []

    # 7) Cross-Encoder 精排：向量看「整段像不像」，BM25 看「词有没有出现」，都判断不了
    #    「这段条文到底有没有回答问题」。把问题和片段成对过 bge-reranker，可纠正
    #    「词面相近但答非所问」（问疏散距离命中疏散宽度）与「短块靠一句命中率刷分」。
    if config.RERANK_ENABLED and len(gated) > 1:
        head = gated[: config.RERANK_TOP_N]
        probs = rerank.score(query, [_rerank_text(d) for d, _, _ in head])
        if probs:
            w = config.RERANK_WEIGHT
            reranked = []
            for (d, s, fl), p in zip(head, probs):
                d.metadata["rerank_score"] = round(float(p), 3)
                if fl > 0:
                    # 确定性证据（用户点名的条文号 / 标题几乎照抄查询）不做精排改写：
                    # 这类片段的排位由条文号与规范权威性决定，Cross-Encoder 分不清
                    # 「通用规范 5.2.2」和「汽车库规范 5.2.2」，让它覆盖会把答案换错。
                    score = s
                else:
                    # 其余片段做有界微调（±w/2）：只翻转接近并列的情况，
                    # 例如问「疏散距离」时把「疏散宽度」的条文压下去，但不推翻权威性差距。
                    score = s + w * (float(p) - 0.5)
                reranked.append((d, round(max(score, max(fl, 0.0)), 4), fl))
            # 未进精排的尾部片段按「精排最低分之下」处理：两套分数不在同一量纲，
            # 直接混排会让没经过精排的块靠原始融合分反超，等于白排。
            tail = gated[len(head):]
            if tail:
                cap = min(x[1] for x in reranked) - 0.01
                tail = [(d, min(s, cap), fl) for d, s, fl in tail]
            gated = reranked + tail
            gated.sort(key=lambda x: -x[1])

    # 8) 同文档行号重叠去重 + 同来源限流，取 top_k
    #    限流原因：条文号相同的条文在 200 部规范里各有一份，不限流会被同一部规范占满，
    #    用户看不到真正想找的那部（也让大模型缺少跨规范对照信息）。
    merged: List[Tuple[Document, float]] = []
    kept: List[dict] = []
    per_src: dict = {}
    max_per_src = 1 if articles else MAX_PER_SOURCE   # 带条文号时优先跨规范对照
    for d, s, _ in gated:
        m = d.metadata
        if any(_overlaps(m, k) for k in kept):
            continue
        src = _src_key(m.get("source", ""))
        if per_src.get(src, 0) >= max_per_src:
            continue
        merged.append((d, s))
        kept.append(m)
        per_src[src] = per_src.get(src, 0) + 1
        if len(merged) >= top_k:
            break

    return merged
