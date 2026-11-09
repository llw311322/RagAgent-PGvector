"""文本分块模块：把长文档自动切割成适合向量化的片段

提供两种切分方式：
- split_documents:   递归字符切分（通用，适合 docx/xlsx/pdf 等无行号概念的文档）
- split_text_lines:  结构感知切分 + 记录原文行号（txt/md 专用）

split_text_lines 的做法（2026-11 结构化改造）：
1. 先按「章/节/条」标题把全文切成结构块，条文不会被从列表中间切断；
2. 再把小结构块合并成不超过 CHUNK_SIZE 的片段，片段携带所属标题路径（heading）；
3. 向量化时用 embed_text() 把标题路径拼在正文前——解决"块里缺条文标题上下文、向量搜不到"的问题；
4. page_content 始终等于原文某段连续行，行号可直接定位原文。
"""
import re
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
import meta_tagger

# ---------- 标题识别规则 ----------
_MD_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(\S.*?)\s*#*\s*$")          # Markdown: ## 标题
_CN_HEADING = re.compile(r"^\s*(第[一二三四五六七八九十百零〇\d]{1,6}[编章节条])\s*[、.．]?\s*(\S.*)?$")
_ART3 = re.compile(r"^\s*(\d{1,2}\.\d{1,2}\.\d{1,3})(?=[\s\u3000]|$)")     # 条文: 5.2.2
_SEC2 = re.compile(r"^\s*(\d{1,2}\.\d{1,2})(?![\d.])[\s\u3000]+(\S.*)$")   # 节:   5.2 工业建筑
_SEC1 = re.compile(r"^\s*(\d{1,2})(?![\d.])[\s\u3000]+([\u4e00-\u9fff].*)$")  # 章: 5 工业建筑

# 只有「短行且不像列表项」才认定为标题，避免把 "1 电动汽车充电站建筑；" 这类列表项误判成章标题
_SEC2_MAX_LEN = 42
_SEC1_MAX_LEN = 30
_LIST_TAIL = "；;，,。、：:"
# 章标题必须是纯名词短语：纯中文 2~12 字（可带英文副名），不能是折行后的句子片段
_CHAP_TITLE = re.compile(r"^[\u4e00-\u9fff]{2,12}(?:[ \u3000][A-Za-z][A-Za-z .\-]{0,20})?$")
# OCR 折行常把句子后半段带到行首，这些字开头的不是标题
_FUNC_HEAD = "的且时应在与或按用对其该可为有和及以从把被则即"


def _detect_heading(line: str) -> Optional[Tuple[int, str, str]]:
    """识别标题行，返回 (层级深度, 编号, 标题文本)；不是标题返回 None。
    层级深度：1=章/编、2=节、3=条；Markdown 直接用 # 个数。"""
    s = line.strip()
    if not s or len(s) > 120:
        return None

    m = _MD_HEADING.match(line)
    if m:
        return len(m.group(1)), "", m.group(2).strip()

    m = _ART3.match(line)
    if m:
        no = m.group(1)
        title = s[len(no):].strip(" \u3000\t、.．")
        return 3, no, title

    m = _CN_HEADING.match(line)
    if m and len(s) <= 40:
        head = m.group(1)
        depth = 1 if head[-1] in "编章" else (2 if head[-1] == "节" else 3)
        return depth, "", s

    m = _SEC2.match(line)
    if (
        m
        and len(s) <= _SEC2_MAX_LEN
        and _CHAP_TITLE.match(m.group(2).strip())
        and m.group(2).strip()[0] not in _FUNC_HEAD
    ):
        return 2, m.group(1), m.group(2).strip()

    m = _SEC1.match(line)
    if (
        m
        and len(s) <= _SEC1_MAX_LEN
        and _CHAP_TITLE.match(m.group(2).strip())
        and m.group(2).strip()[0] not in _FUNC_HEAD
    ):
        return 1, m.group(1), m.group(2).strip()

    return None


def _join_path(stack: List[Tuple[int, str]]) -> str:
    """把标题栈拼成路径文本：第5章 建筑构造 > 5.2 工业建筑 > 5.2.2 下列工业建筑…"""
    return " > ".join(t for _, t in stack if t)


def split_documents(docs: List[Document]) -> List[Document]:
    """
    递归字符切分：优先按段落/换行切，再按句子边界（。！？等）兜底，保证语义完整。
    chunk_size / chunk_overlap 可在 config.py 中调整。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", ". ", " ", ""],
        length_function=len,
    )
    out: List[Document] = []
    for d in splitter.split_documents(docs):
        meta_tagger.tag_chunk(
            d.metadata, d.page_content, meta_tagger.doc_status(d.metadata.get("source", ""), d.page_content)
        )
        if d.metadata.get("quality") == "ok":
            out.append(d)
    return out


def _split_span(lines: List[str], s: int, e: int, overlap: int):
    """把行区间 [s, e] 按字符数切成若干子区间，相邻子区间保留 overlap 行重叠"""
    i = s
    while i <= e:
        j, size = i, 0
        while j <= e and (size < config.CHUNK_SIZE or j == i):
            size += len(lines[j]) + 1
            j += 1
        yield i, j - 1
        if j > e:
            break
        i = max(i + 1, j - overlap)


def split_text_lines(text: str, source: str, ext: str, mtime=None) -> List[Document]:
    """
    结构感知切分（txt/md 专用），每个片段记录原文起止行号（1-based）与所属标题路径。

    规则：
    - 以「章/节/条」标题为边界切结构块，条文/列表不会被从中间截断；
    - 相邻小结构块合并到 CHUNK_SIZE 以内；单个结构块超长时才按行二次切分（保留 2 行重叠）；
    - metadata 额外记录 heading（标题路径）、article（条文号）、section（节号），供检索与引用展示。
    """
    lines = [meta_tagger.clean_line(ln) for ln in text.replace("\r\n", "\n").split("\n")]
    # 文档级有效性只判一次（正文首部 800 字），逐块传下去：
    # 否则「（征求意见稿）」这类只在首页出现的字样，只有首页块能识别出草案身份。
    doc_status = meta_tagger.doc_status(source, "\n".join(lines[:40]))

    # ---------- 第一遍：按标题切结构块 ----------
    blocks: List[Tuple[int, int, str, str, str]] = []   # (起行, 止行, 标题路径, 条文号, 节号)
    stack: List[Tuple[int, str]] = []
    cur_start = 0
    cur_path, cur_art, cur_sec = "", "", ""

    def close(end: int) -> None:
        if end >= cur_start:
            blocks.append((cur_start, end, cur_path, cur_art, cur_sec))

    for i, ln in enumerate(lines):
        h = _detect_heading(ln)
        if not h:
            continue
        if i > cur_start or not blocks:
            close(i - 1)
        depth, no, title = h
        while stack and stack[-1][0] >= depth:
            stack.pop()
        label = f"{no} {title}".strip() if no else title
        stack.append((depth, label[:40]))
        cur_path = _join_path(stack)
        cur_art = no if depth == 3 else (cur_art if depth > 3 else "")
        cur_sec = no if depth == 2 else (cur_sec if depth > 2 else "")
        cur_start = i
    close(len(lines) - 1)

    # ---------- 第二遍：每个结构块生成一个片段 ----------
    # 不跨条文合并：一条条文一块，片段标题路径才准确（合并会让 5.2.2 的块挂着 5.2.1 的标题）。
    # 特殊的「只含节/章标题行」的块并入下一块，避免产生只有一行标题的碎片。
    chunks: List[Document] = []

    def emit(s: int, e: int, path: str, art: str, sec: str) -> None:
        body = "\n".join(lines[s : e + 1])
        if not body.strip():
            return
        meta = {
            "source": source,
            "type": ext,
            "chunk_index": len(chunks),
            "start_line": s + 1,
            "end_line": e + 1,
        }
        if path:
            meta["heading"] = path
        if art:
            meta["article"] = art
        if sec:
            meta["section"] = sec
        if mtime is not None:
            meta["mtime"] = mtime
        # 数据清洗 + 结构化打标：脏行已在上面剔除；乱码块（图示扫描页）与碎片块不产出
        meta_tagger.tag_chunk(meta, body, doc_status)
        if meta.get("quality") != "ok":
            return
        chunks.append(Document(page_content=body, metadata=meta))

    pend_s: Optional[int] = None      # 待并入下一块的节/章标题行起点
    pend_e: Optional[int] = None

    for s, e, path, art, sec in blocks:
        h = _detect_heading(lines[s])
        only_heading = s == e and h is not None and h[0] <= 2
        if only_heading:
            if pend_s is None:
                pend_s = s
            pend_e = e
            continue

        start = pend_s if pend_s is not None else s
        pend_s = pend_e = None
        block_len = sum(len(lines[k]) + 1 for k in range(start, e + 1))

        if block_len <= config.CHUNK_SIZE:
            emit(start, e, path, art, sec)
        else:
            # 单条超长（整章无小标题、或超长条文）：按行二次切分，片段继承同一标题路径
            for ss, ee in _split_span(lines, start, e, config.LINE_OVERLAP_LINES):
                emit(ss, ee, path, art, sec)

    if pend_s is not None:            # 文末遗留的标题行
        emit(pend_s, pend_e, "", "", "")

    return chunks


def embed_text(doc: Document) -> str:
    """向量化用的文本：标题路径 + 正文。
    标题路径让「5.2.2」「工业建筑」这类上下文进入向量，解决块从条文列表中间切开、语义不完整的问题；
    page_content 本身不掺标题，保持与原文逐行一致（引用可精确定位）。"""
    heading = (doc.metadata or {}).get("heading") or ""
    if not heading:
        return doc.page_content
    return f"{heading}\n{doc.page_content}"
