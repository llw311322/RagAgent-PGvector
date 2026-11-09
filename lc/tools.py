"""暴露给 Agent 的 LangChain 工具

四个工具覆盖四类问题：
- 语义/术语型问题   -> search_standards（走向量+BM25+精排）
- 明确条文号        -> get_clause（走 SQL 精确匹配，比语义检索可靠）
- 有哪些规范        -> list_standards（SQL 聚合）
- 库有多大/覆盖什么 -> kb_stats（SQL 聚合）
"""
from typing import List, Optional

from langchain_core.tools import tool

import config
from lc import prompt as P
from lc import retriever as R
from lc import store


def _fmt_docs(docs: List) -> str:
    if not docs:
        return "未检索到相关条文。"
    rows = []
    for i, d in enumerate(docs, 1):
        m = d.metadata or {}
        head = f"[{i}] {m.get('source', '未知')}"
        if m.get("standard_code"):
            head += f"｜{m['standard_code']}"
        if m.get("clause_no") or m.get("article"):
            head += f"｜第{m.get('clause_no') or m.get('article')}条"
        if m.get("start_line"):
            head += f"｜第{m['start_line']}-{m['end_line']}行"
        rows.append(f"{head}\n{d.page_content[:420]}")
    return "\n\n".join(rows)


@tool
def search_standards(query: str, domain: str = "", building: str = "") -> str:
    """在建筑规范知识库中检索条文（语义 + 关键词混合检索，带精排）。

    Args:
        query: 检索内容，用规范术语描述，例如「高层住宅消防车道宽度」。
        domain: 可选专业筛选，取值：防火、电气、暖通燃气、结构、节能、装饰装修、规划总图、给排水、无障碍、综合。
        building: 可选建筑类型筛选，取值：住宅、办公、酒店、汽车库、地下、高层、工业、教育、医疗、商业。
    """
    filters = {}
    if domain:
        filters["domain"] = domain
    if building:
        filters["building"] = building
    docs = R.retrieve(query, filters=filters, k=config.LC_TOP_K)
    return _fmt_docs(docs)


@tool
def get_clause(clause_no: str, standard_code: str = "") -> str:
    """按条文号精确查询原文（不走向量检索，适合「GB 55037 第 5.2.2 条」这类明确问题）。

    Args:
        clause_no: 条文号，例如「5.2.2」或「5.1.5」。
        standard_code: 可选规范编号，例如「GB 55037」；留空则在全部规范中查找同名条文。
    """
    rows = store.find_clause(clause_no, standard_code)
    if not rows:
        return f"知识库中未找到 {standard_code or '任何规范'} 的第 {clause_no} 条。"
    out = []
    for meta, body in rows:
        m = meta or {}
        head = (f"来源: {m.get('source')}｜编号: {m.get('standard_code') or '—'}"
                f"｜条文号: {m.get('clause_no') or clause_no}")
        if m.get("start_line"):
            head += f"｜第{m.get('start_line')}-{m.get('end_line')}行"
        if m.get("heading"):
            head += f"｜{m['heading']}"
        if m.get("is_mandatory"):
            head += "｜强制性条文"
        out.append(f"{head}\n{(body or '')[:600]}")
    return "\n\n".join(out)


@tool
def list_standards(keyword: str = "") -> str:
    """列出知识库收录的规范清单（规范名 + 编号 + 条文块数）。

    Args:
        keyword: 可选关键字，按规范名或文件名过滤，例如「防火」「GB 50016」。
    """
    rows = store.list_standards(keyword)
    if not rows:
        return "未匹配到规范。"
    lines = [f"- {name or '（未识别名称）'}｜{code or '—'}｜{n} 块" for name, code, n in rows]
    return f"共匹配 {len(rows)} 部规范（最多显示 40 部）：\n" + "\n".join(lines)


@tool
def kb_stats() -> str:
    """查看知识库规模与结构分布：文档数、条文块数、标准大类、专业主题、建筑类型。"""
    s = store.stats()
    lines = [f"文档 {s['docs']} 篇，条文块 {s['chunks']} 块（集合 {s['collection']}）"]
    for field, label in (("std_class", "标准大类"), ("domain", "专业主题"), ("buildings", "建筑类型")):
        dist = store.distribution(field, limit=6)
        if dist:
            lines.append(label + "：" + "、".join(f"{k or '未标注'} {n}" for k, n in dist))
    return "\n".join(lines)


TOOLS = [search_standards, get_clause, list_standards, kb_stats]
