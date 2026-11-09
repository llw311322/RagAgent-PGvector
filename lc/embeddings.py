"""向量模型工厂：bge-m3（1024 维，本地 Ollama）

为什么用 bge-m3 而不是原来的 nomic-embed-text：
- 维度 1024，与库中 vector(1024) 对齐；
- 中文/多语训练充分，规范条文这类长句语义区分度明显更好；
- 支持中英混排（GB 50016 这类编号 + 中文术语的场景）。
"""
import threading
from typing import List

from langchain_ollama import OllamaEmbeddings

import config

_LOCK = threading.Lock()
_EMB: OllamaEmbeddings | None = None
_DIM: int | None = None


def get_embeddings() -> OllamaEmbeddings:
    """进程级单例：构造时会校验模型存在，重复构造约 1.7s，必须复用"""
    global _EMB
    if _EMB is None:
        with _LOCK:
            if _EMB is None:
                _EMB = OllamaEmbeddings(
                    model=config.EMBED_MODEL_PG,
                    base_url=config.OLLAMA_BASE_URL,
                    # 该字段只接受整数秒；设 30 分钟让向量模型常驻，避免每次查询冷加载
                    keep_alive=1800,
                )
    return _EMB


def dimension(probe: bool = True) -> int:
    """探测向量维度，并校验与配置一致（不一致会让写入直接失败，提前拦截更友好）"""
    global _DIM
    if _DIM is not None:
        return _DIM
    if not probe:
        return config.EMBED_DIM
    vec = get_embeddings().embed_query("防火间距")
    _DIM = len(vec)
    if _DIM != config.EMBED_DIM:
        raise RuntimeError(
            f"向量维度不匹配：模型 {config.EMBED_MODEL_PG} 输出 {_DIM} 维，"
            f"但 config.EMBED_DIM={config.EMBED_DIM}（PG 表为 vector({config.EMBED_DIM})）。"
            f"请改配置或重建表，否则写入会报维度错误。"
        )
    return _DIM


def embed_documents(texts: List[str]) -> List[List[float]]:
    return get_embeddings().embed_documents(texts)


def embed_query(text: str) -> List[float]:
    return get_embeddings().embed_query(text)


def status() -> dict:
    try:
        d = dimension(probe=True)
        return {"ok": True, "model": config.EMBED_MODEL_PG, "dim": d}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "model": config.EMBED_MODEL_PG, "error": str(e)[:200]}
