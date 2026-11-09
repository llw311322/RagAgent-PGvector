"""提示词与上下文格式化（防幻觉约束与旧链路一致）"""
from typing import Dict, List

import config
from langchain_core.documents import Document

SYSTEM_PROMPT = """你是建筑规范检索助手。用户消息中会给出编号的【参考条文】，每条都标注了规范编号、条文号、来源文件与原文行号。

回答规则（必须严格遵守）：
1. 只能使用【参考条文】中的内容回答，禁止使用参考条文之外的记忆、常识或推断作答；
2. 每个结论句末必须标注引用编号，格式如 [1] 或 [1][3]；引用规范时必须写出规范编号与条文号，
   例如「GB 55037-2022 第 5.2.2 条 [1]」；
3. 如果参考条文不足以回答，直接回复「当前规范库未检索到对应条文」，并说明缺什么，
   禁止编造规范名称、条文号、参数、数值；
4. 不得把不同条文的数字拼接成新结论；条文有前提条件（如建筑类别、高度、耐火等级）时必须一并写出；
5. 标有「条文说明」或「编制依据」的片段只能用于解释，不能当作条文正文引用；
6. 片段标注「效力: 征求意见稿/非现行版本」时，必须在回答中明确提示「该条文出自征求意见稿，
   请以正式发布版为准」，不得直接当作现行强条给结论；
7. 用简体中文，条理清晰、设计师可直接阅读；不确定的地方明确说"依据不足"。"""

HUMAN_PROMPT = """【参考条文】
{context}

【用户问题】{question}"""

CONDENSE_PROMPT = """根据下面的对话历史，把用户最新的追问改写成一个不依赖上下文、可独立检索的完整问题。
要求：保留原问题中的专业术语、规范编号、条文号与数值；只输出改写后的问题本身，不要解释。

【对话历史】
{history}

【最新追问】{question}

改写后的问题："""

NO_EVIDENCE_TEXT = (
    "⚠️ 知识库中未检索到与您问题相关的条文（精排相关度低于 {min_score}）。\n\n"
    "为避免凭模型记忆编造，本次不作答。可以：① 在 docs 目录补充相关规范后重新入库；"
    "② 换个说法（带上规范编号或条文号更易命中）；③ 放宽左侧的筛选条件。"
)


def _loc_label(meta: dict) -> str:
    if meta.get("start_line"):
        return f"第 {meta['start_line']}-{meta['end_line']} 行"
    idx = meta.get("chunk_index")
    return f"片段 #{idx + 1}" if idx is not None else "位置未知"


def format_context(docs: List[Document], max_chars: int = None) -> str:
    """把命中片段组装成带编号、来源、条文号、行号、效力的上下文"""
    limit = max_chars or config.LC_CONTEXT_MAX_CHARS
    lines: List[str] = []
    used = 0
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        body = doc.page_content
        if used + len(body) > limit and lines:
            break
        used += len(body)
        source = meta.get("source", "未知来源")
        code = meta.get("standard_code") or ""
        clause = meta.get("clause_no") or meta.get("article") or ""
        head = f"[片段{i}｜来源: {source}"
        if code:
            head += f"｜规范编号: {code}"
        if clause:
            head += f"｜条文号: {clause}"
        head += f"｜位置: {_loc_label(meta)}"
        if meta.get("heading"):
            head += f"｜所属条文: {meta['heading']}"
        if meta.get("source_type") and meta["source_type"] != "条文正文":
            head += f"｜类型: {meta['source_type']}"
        if meta.get("is_mandatory"):
            head += "｜强制性条文"
        if meta.get("doc_status") == "draft" or meta.get("valid") is False:
            head += "｜效力: 征求意见稿/非现行版本（引用时必须说明）"
        score = meta.get("rerank") or meta.get("sim") or 0
        head += f"｜相关度: {min(float(score), 1.0):.0%}]"
        lines.append(f"{head}\n{body}")
    return "\n\n".join(lines)


def evidence_of(docs: List[Document]) -> List[Dict]:
    """依据卡片数据（字段与旧界面保持一致，前端无需改动）"""
    out: List[Dict] = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        score = meta.get("rerank") or meta.get("sim") or 0.0
        out.append({
            "no": i,
            "source": meta.get("source", "未知来源"),
            "type": meta.get("type", ""),
            "start_line": meta.get("start_line"),
            "end_line": meta.get("end_line"),
            "heading": meta.get("heading", ""),
            "article": meta.get("article", ""),
            "clause_no": meta.get("clause_no", "") or meta.get("article", ""),
            "standard_code": meta.get("standard_code", ""),
            "standard_name": meta.get("standard_name", ""),
            "source_type": meta.get("source_type", ""),
            "is_mandatory": bool(meta.get("is_mandatory")),
            "valid": meta.get("valid", True),
            "doc_status": meta.get("doc_status", ""),
            "rerank": meta.get("rerank"),
            "score": float(score),
            "std_class": meta.get("std_class", ""),
            "domain": meta.get("domain", ""),
            "buildings": meta.get("buildings", []),
            "content": doc.page_content,
        })
    return out
