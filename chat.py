"""RAG 主流程：检索 -> 组装上下文 -> 大模型回答（流式，含阈值过滤与引用溯源）"""
from typing import Dict, List, Tuple

from langchain_core.documents import Document

import config
from llm import get_llm
from retriever import retrieve

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
7. 用简体中文，条理清晰、设计师可直接阅读；不确定的地方明确说"依据不足"。

【参考条文】
{context}
"""


def _loc_label(meta: dict) -> str:
    """片段的位置标签：txt/md 有行号，其他类型显示片段序号"""
    if meta.get("start_line"):
        return f"第 {meta['start_line']}-{meta['end_line']} 行"
    idx = meta.get("chunk_index")
    return f"片段 #{idx + 1}" if idx is not None else "位置未知"


def build_context(scored: List[Tuple[Document, float]]) -> str:
    """把检索到的片段组装成给大模型的上下文（含来源、标题路径与行号）"""
    lines = []
    for i, (doc, score) in enumerate(scored, 1):
        meta = doc.metadata
        source = meta.get("source", "未知来源")
        heading = meta.get("heading")
        code = meta.get("standard_code") or meta.get("std_no") or ""
        clause = meta.get("clause_no") or meta.get("article") or ""
        head = f"[片段{i}｜来源: {source}"
        if code:
            head += f"｜规范编号: {code}"
        if clause:
            head += f"｜条文号: {clause}"
        head += f"｜位置: {_loc_label(meta)}"
        if heading:
            head += f"｜所属条文: {heading}"
        if meta.get("source_type") and meta["source_type"] != "条文正文":
            head += f"｜类型: {meta['source_type']}"
        if meta.get("is_mandatory"):
            head += "｜强制性条文"
        # 告知模型这是草案：否则它会当成现行强条直接给结论，用户拿去设计就出问题了
        if meta.get("doc_status") == "draft" or meta.get("valid") is False:
            head += "｜效力: 征求意见稿/非现行版本（引用时必须说明）"
        head += f"｜相关度: {min(score, 1.0):.0%}]"
        lines.append(f"{head}\n{doc.page_content}")
    return "\n\n".join(lines)


def evidence_of(scored: List[Tuple[Document, float]]) -> List[Dict]:
    """把命中片段整理成前端可展示的依据列表"""
    out = []
    for i, (doc, score) in enumerate(scored, 1):
        meta = doc.metadata
        out.append({
            "no": i,
            "source": meta.get("source", "未知来源"),
            "type": meta.get("type", ""),
            "start_line": meta.get("start_line"),
            "end_line": meta.get("end_line"),
            "heading": meta.get("heading", ""),
            "article": meta.get("article", ""),
            "clause_no": meta.get("clause_no", "") or meta.get("article", ""),
            "standard_code": meta.get("standard_code", "") or meta.get("std_no", ""),
            "standard_name": meta.get("standard_name", ""),
            "source_type": meta.get("source_type", ""),
            "is_mandatory": bool(meta.get("is_mandatory")),
            "valid": meta.get("valid", True),
            "doc_status": meta.get("doc_status", ""),
            "rerank": meta.get("rerank_score"),      # 精排分（未启用时为 None）
            "score": score,
            "std_class": meta.get("std_class", ""),
            "domain": meta.get("domain", ""),
            "buildings": meta.get("buildings", []),
            "content": doc.page_content,
        })
    return out


def ask(question: str, llm=None) -> Dict:
    """
    问答入口（一次性返回完整答案，兼容旧调用）。

    返回 dict:
      - answer:      最终回答文本
      - context:     使用的知识库上下文
      - chunks:      命中的片段列表 [(Document, 分数)]
      - evidence:    依据列表（含文档名/行号/相关度）
      - need_search: 知识库是否未能回答，需要联网搜索
    """
    scored = retrieve(question)

    # 知识库中未检索到足够相关的片段：直接明说，不调用模型凭记忆作答
    if not scored:
        return {
            "answer": (
                "⚠️ 知识库中未检索到与您问题相关的信息（相关度低于设定阈值 "
                f"{config.SIMILARITY_THRESHOLD}）。\n\n"
                "为避免凭模型记忆编造，本次不作答。您可以将相关文档放入 docs 目录后点击「扫描 docs 目录并入库」，"
                "或换个问法再试。"
            ),
            "context": "",
            "chunks": [],
            "evidence": [],
            "need_search": True,
        }

    context = build_context(scored)
    llm = llm or get_llm()
    answer = llm.generate(SYSTEM_PROMPT.format(context=context), question)

    return {
        "answer": answer,
        "context": context,
        "chunks": scored,
        "evidence": evidence_of(scored),
        "need_search": False,
    }


def ask_stream(question: str, llm=None):
    """
    流式问答入口（供网页界面使用）。llm 可传入界面所选提供商的实例，缺省按 config 构建。

    依次 yield:
      {"type": "chunks",   "chunks": [...], "context": str}   —— 先给出命中片段（无命中则不 yield 此项）
      {"type": "delta",    "text": str}                        —— 逐段回答文本
      {"type": "done",     "answer": str}                      —— 结束，携带完整回答
    若知识库无相关片段，yield {"type": "no_evidence"} 后结束。
    """
    scored = retrieve(question)

    if not scored:
        yield {
            "type": "no_evidence",
            "message": (
                "⚠️ 知识库中未检索到与您问题相关的信息（相关度低于设定阈值 "
                f"{config.SIMILARITY_THRESHOLD}）。\n\n"
                "为避免凭模型记忆编造，本次不作答。您可以将相关文档放入 docs 目录后点击「扫描 docs 目录并入库」，"
                "或换个问法再试。"
            ),
        }
        return

    context = build_context(scored)
    yield {"type": "chunks", "chunks": scored, "context": context}

    llm = llm or get_llm()
    answer = ""
    for delta in llm.stream(SYSTEM_PROMPT.format(context=context), question):
        answer += delta
        yield {"type": "delta", "text": delta}

    yield {"type": "done", "answer": answer, "chunks": scored}
