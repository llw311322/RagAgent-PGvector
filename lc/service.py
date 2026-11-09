"""门面层：给 Streamlit 提供稳定接口，界面不直接碰 LangChain 内部对象"""
import threading
import time
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import config
from lc import bm25, chain, embeddings, ingest, llm_factory, store


# ---------------- 状态 ----------------
def status() -> Dict:
    emb = embeddings.status()
    db = store.health()
    n = 0
    try:
        n = store.stats()["chunks"]
    except Exception:
        pass
    return {
        "engine": "LangChain + PGVector",
        "embed_ok": emb.get("ok", False),
        "embed_model": emb.get("model"),
        "embed_dim": emb.get("dim"),
        "embed_error": emb.get("error"),
        "db_ok": db.get("ok", False),
        "db_error": db.get("error"),
        "collection": config.PG_COLLECTION,
        "chunks": n,
    }


_CACHE: Dict[str, tuple] = {"t": 0.0, "stats": None}


def stats(ttl: float = 15.0) -> Dict:
    """带 TTL 的统计缓存：每次侧栏渲染都打一遍 SQL 没必要"""
    now = time.time()
    if _CACHE["stats"] and now - _CACHE["t"] < ttl:
        return _CACHE["stats"]
    s = store.stats()
    _CACHE.update(t=now, stats=s)
    return s


def invalidate_stats() -> None:
    _CACHE.update(t=0.0, stats=None)


# ---------------- 预热 ----------------
def warmup(background: bool = True) -> Dict:
    """预热：向量模型常驻 + BM25 索引载入 + 数据库连通"""
    state = {"embed": "pending", "bm25": "pending", "db": "pending"}

    def work():
        try:
            embeddings.dimension(probe=True)
            state["embed"] = "ready"
        except Exception as e:  # noqa: BLE001
            state["embed"] = f"failed: {e}"
        try:
            state["bm25"] = "ready" if bm25.get_index() is not None else "empty"
        except Exception as e:  # noqa: BLE001
            state["bm25"] = f"failed: {e}"
        try:
            state["db"] = "ready" if store.health().get("ok") else "failed"
        except Exception as e:  # noqa: BLE001
            state["db"] = f"failed: {e}"

    if background:
        threading.Thread(target=work, daemon=True).start()
    else:
        work()
    return state


# ---------------- 入库 ----------------
def scan_docs(progress_cb: Optional[Callable] = None, force: bool = False) -> Tuple[int, List[str]]:
    n, details = ingest.scan_docs_dir(progress_cb, force=force)
    invalidate_stats()
    return n, details


def rebuild_all(progress_cb: Optional[Callable] = None) -> Tuple[int, List[str]]:
    n, details = ingest.rebuild_all(progress_cb)
    invalidate_stats()
    return n, details


def delete_source(name: str) -> int:
    n = store.delete_source(name)
    bm25.invalidate()
    invalidate_stats()
    return n


# ---------------- 问答 ----------------
def _make_llm(provider: str, key: Optional[str]):
    return llm_factory.get_chat_model(provider, key or None)


def ask_stream(
    question: str,
    provider: str = None,
    api_key: Optional[str] = None,
    filters: Optional[dict] = None,
    history: Optional[List[Tuple[str, str]]] = None,
    mode: str = "rag",
) -> Iterator[Dict]:
    """统一事件流：evidence / query / tool / delta / no_evidence / done / error"""
    history = history or []
    try:
        llm = _make_llm(provider or config.LLM_PROVIDER, api_key)
    except Exception as e:  # noqa: BLE001
        yield {"type": "error", "message": str(e)}
        return

    if mode == "agent":
        from lc import agent as agent_mod

        yield {"type": "notice", "message": "Agent 模式：模型会自行选择检索工具"}
        try:
            ag = agent_mod.get_agent(llm)
        except Exception as e:  # noqa: BLE001
            yield {"type": "error", "message": f"Agent 初始化失败：{e}"}
            return
        for ev in agent_mod.stream_agent(ag, question, history):
            yield ev
        return

    try:
        chat = chain.RAGChat(llm, filters)
    except Exception as e:  # noqa: BLE001
        yield {"type": "error", "message": f"检索器初始化失败：{e}"}
        return
    for ev in chat.ask_stream(question, history):
        yield ev


def ask(
    question: str,
    provider: str = None,
    api_key: Optional[str] = None,
    filters: Optional[dict] = None,
    history: Optional[List[Tuple[str, str]]] = None,
) -> Dict:
    llm = _make_llm(provider or config.LLM_PROVIDER, api_key)
    return chain.RAGChat(llm, filters).ask(question, history)
