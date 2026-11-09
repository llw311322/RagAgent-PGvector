"""LCEL 问答链：追问改写 → 检索 → 提示词 → 模型 → 文本解析

链路（LangChain 表达式语言，管道即组合）：
  condense:   prompt | model | StrOutputParser
  qa:         ChatPromptTemplate | model | StrOutputParser

会话记忆用 RunnableWithMessageHistory 的底层组件 InMemoryChatMessageHistory；
因为检索需要「改写后的问题」，这里把改写步骤显式放在链前，而不是用
ConversationalRetrievalChain（它不便于在 Streamlit 里逐步展示依据）。
"""
from typing import Dict, Iterator, List, Optional, Tuple

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import config
from lc import prompt as P
from lc import retriever as R

_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", P.SYSTEM_PROMPT),
    MessagesPlaceholder("history", optional=True),
    ("human", P.HUMAN_PROMPT),
])

_CONDENSE_PROMPT = ChatPromptTemplate.from_messages([("human", P.CONDENSE_PROMPT)])


def build_qa_chain(llm: BaseChatModel):
    """LCEL：提示词 → 模型 → 纯文本"""
    return _QA_PROMPT | llm | StrOutputParser()


def build_condense_chain(llm: BaseChatModel):
    return _CONDENSE_PROMPT | llm | StrOutputParser()


def _fmt_history(history: List[Tuple[str, str]], limit: int = 6) -> str:
    if not history:
        return ""
    rows = []
    for role, content in history[-limit:]:
        tag = "用户" if role == "user" else "助手"
        rows.append(f"{tag}：{content[:220]}")
    return "\n".join(rows)


def rewrite_question(question: str, history: List[Tuple[str, str]], llm: BaseChatModel) -> str:
    """多轮追问改写成独立问题（只取最近 3 轮，避免带偏）"""
    if not history:
        return question
    try:
        out = build_condense_chain(llm).invoke(
            {"history": _fmt_history(history[-6:]), "question": question}
        ).strip()
        out = out.strip("“”\"' \n")
        # 改写失败时会返回一长段解释，宁可用原问题
        return out if 0 < len(out) <= 120 else question
    except Exception:
        return question


class RAGChat:
    """一次会话的问答对象：持有模型、筛选条件与检索器缓存"""

    def __init__(self, llm: BaseChatModel, filters: Optional[dict] = None):
        self.llm = llm
        self.filters = filters or {}
        self._retriever = R.build_retriever(self.filters)

    # ---- 检索 ----
    def retrieve(self, question: str, k: Optional[int] = None) -> List[Document]:
        docs = list(self._retriever.invoke(question))
        return docs[: (k or config.LC_TOP_K)]

    # ---- 一次性回答 ----
    def ask(self, question: str, history: Optional[List[Tuple[str, str]]] = None) -> Dict:
        history = history or []
        query = rewrite_question(question, history, self.llm)
        docs = self.retrieve(query)
        if not docs:
            return {
                "answer": P.NO_EVIDENCE_TEXT.format(min_score=config.LC_RERANK_MIN),
                "evidence": [],
                "docs": [],
                "query": query,
                "ok": False,
            }
        context = P.format_context(docs)
        answer = build_qa_chain(self.llm).invoke({
            "context": context,
            "question": question,
            "history": self._as_messages(history),
        })
        return {
            "answer": answer,
            "evidence": P.evidence_of(docs),
            "docs": docs,
            "query": query,
            "ok": True,
        }

    # ---- 流式回答（供 Streamlit 使用）----
    def ask_stream(self, question: str, history: Optional[List[Tuple[str, str]]] = None) -> Iterator[Dict]:
        history = history or []
        query = rewrite_question(question, history, self.llm)
        if query != question:
            yield {"type": "query", "text": query}

        docs = self.retrieve(query)
        if not docs:
            yield {
                "type": "no_evidence",
                "message": P.NO_EVIDENCE_TEXT.format(min_score=config.LC_RERANK_MIN),
            }
            return

        yield {"type": "evidence", "evidence": P.evidence_of(docs), "docs": docs}

        context = P.format_context(docs)
        answer = ""
        for delta in build_qa_chain(self.llm).stream({
            "context": context,
            "question": question,
            "history": self._as_messages(history),
        }):
            if not delta:
                continue
            answer += delta
            yield {"type": "delta", "text": delta}
        yield {"type": "done", "answer": answer, "docs": docs, "query": query}

    @staticmethod
    def _as_messages(history: List[Tuple[str, str]], limit: int = 6):
        msgs = []
        for role, content in history[-limit:]:
            msgs.append(HumanMessage(content=content) if role == "user" else AIMessage(content=content))
        return msgs


_HISTORY: Dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """进程内会话记忆（重启即清空；接 Redis 时只需换这一处）"""
    if session_id not in _HISTORY:
        _HISTORY[session_id] = InMemoryChatMessageHistory()
    return _HISTORY[session_id]


def reset_session(session_id: str) -> None:
    _HISTORY.pop(session_id, None)
