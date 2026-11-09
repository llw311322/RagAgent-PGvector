"""Agent 模式：create_agent + 工具调用（LangChain v1）

与 RAG 链的分工：
- RAG 链：一次检索一次作答，快，适合 90% 的「查条文」问题；
- Agent ：模型自己决定调哪个工具（精确条文 / 规范清单 / 库存统计 / 语义检索），
          适合「知识库里有没有 XX 规范」「5.2.2 条跟 5.1.5 条有什么区别」这类需要多步的问题。
"""
from typing import Dict, Iterator, List, Optional, Tuple

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from lc.tools import TOOLS

AGENT_PROMPT = """你是建筑规范检索助手，可以调用工具查询本地规范知识库。

工作方式：
1. 先判断问题类型再选工具：明确条文号用 get_clause；术语/做法类用 search_standards；
   问「有哪些规范」用 list_standards；问库存规模用 kb_stats；
2. 工具返回的条文就是唯一依据，禁止使用工具结果之外的记忆或常识作答；
3. 回答必须写出规范编号与条文号，并标注来源文件与行号；
4. 工具没查到就明确说「当前规范库未检索到对应条文」，不得编造规范号或数值；
5. 条文涉及前提条件（建筑类别、高度、耐火等级）时必须一并说明；
6. 用简体中文，条理清晰，设计师可直接阅读。"""


def get_agent(llm: BaseChatModel):
    return create_agent(model=llm, tools=TOOLS, system_prompt=AGENT_PROMPT)


def _to_messages(history: List[Tuple[str, str]], question: str):
    msgs = []
    for role, content in history[-6:]:
        msgs.append(HumanMessage(content=content) if role == "user" else AIMessage(content=content))
    msgs.append(HumanMessage(content=question))
    return msgs


def stream_agent(agent, question: str, history: Optional[List[Tuple[str, str]]] = None) -> Iterator[Dict]:
    """把 LangGraph 的消息流翻译成界面事件：tool 调用提示 + 文本增量"""
    history = history or []
    state = {"messages": _to_messages(history, question)}
    seen_tools = set()
    answer = ""
    try:
        for chunk, meta in agent.stream(state, stream_mode="messages"):
            # 工具调用阶段：chunk 带 tool_call_chunks，先把「正在调用什么」告诉用户
            for tc in getattr(chunk, "tool_call_chunks", None) or []:
                name = (tc or {}).get("name")
                if name and name not in seen_tools:
                    seen_tools.add(name)
                    yield {"type": "tool", "name": name}
            text = getattr(chunk, "content", "") or ""
            if not text:
                continue
            if not isinstance(chunk, AIMessageChunk):
                continue
            # 跳过工具节点内部的模型输出（那是给工具的参数/中间推理）
            if (meta or {}).get("langgraph_node") == "tools":
                continue
            answer += text
            yield {"type": "delta", "text": text}
    except Exception as e:  # noqa: BLE001
        yield {"type": "error", "message": f"Agent 执行失败：{e}"}
        return
    yield {"type": "done", "answer": answer}
