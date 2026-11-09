"""Cross-Encoder 精排（BAAI/bge-reranker-base）

为什么需要它：
    向量召回看的是「整段文本压成一个向量后像不像」，BM25 看的是「词有没有出现」，两者都是粗排。
    要判断「这段条文到底能不能回答这个问题」，必须把问题和片段成对过一遍模型（Cross-Encoder）。
    它能纠正两类典型错误：
      - 词面很像但答非所问（问「疏散距离」，命中「疏散宽度」的条文）；
      - 短块靠一句命中率刷分，压过真正完整的条文。

两种后端（同一份权重）：
    onnx  ：量化 ONNX（267MB，onnxruntime CPU）。实测 12 条 0.9 秒，首选；
    torch ：transformers + torch。仅作兜底（ONNX 模型文件缺失时），同样 12 条要 3 秒以上，
            CPU 上太慢，所以不作为默认。

模型文件缺失时不报错，检索自动退化为融合分排序（功能不受影响，只是少了精排这一层）。
"""
import os
import threading
from typing import List, Optional

# HuggingFace 国内镜像：必须在 import transformers 之前设置，否则走 huggingface.co 会超时
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

_STATE = {
    "backend": "",       # "onnx" / "torch"
    "session": None,
    "tok": None,
    "model": None,
    "device": "cpu",
    "inputs": [],
    "failed": "",
}
_LOCK = threading.Lock()


def _onnx_path() -> str:
    import config

    return os.path.join(config.RERANK_ONNX_DIR, "model_quantized.onnx")


def _load_onnx() -> bool:
    import onnxruntime as ort
    from transformers import AutoTokenizer

    import config

    path = _onnx_path()
    if not os.path.isfile(path):
        return False
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, min(8, os.cpu_count() or 4))
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    tok = AutoTokenizer.from_pretrained(config.RERANK_ONNX_DIR)
    _STATE.update(
        backend="onnx",
        session=sess,
        tok=tok,
        inputs=[i.name for i in sess.get_inputs()],
        failed="",
    )
    return True


def _load_torch() -> bool:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    import config

    tok = AutoTokenizer.from_pretrained(config.RERANK_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(config.RERANK_MODEL)
    model.eval().to(config.RERANK_DEVICE)
    _STATE.update(backend="torch", tok=tok, model=model, device=config.RERANK_DEVICE, failed="")
    return True


def _load() -> bool:
    """加载模型（线程安全）。失败时把原因记在 _STATE['failed'] 并返回 False。"""
    if _STATE["backend"]:
        return True
    if _STATE["failed"]:
        return False
    import config

    with _LOCK:
        if _STATE["backend"]:
            return True
        if _STATE["failed"]:
            return False
        errors = []
        if config.RERANK_BACKEND in ("onnx", "auto"):
            try:
                if _load_onnx():
                    return True
            except Exception as e:  # noqa: BLE001
                errors.append(f"onnx: {type(e).__name__}: {e}")
        if config.RERANK_BACKEND in ("torch", "auto"):
            try:
                if _load_torch():
                    return True
            except Exception as e:  # noqa: BLE001
                errors.append(f"torch: {type(e).__name__}: {e}")
        _STATE["failed"] = " | ".join(errors) or "no backend"
        return False


def available() -> bool:
    import config

    return bool(config.RERANK_ENABLED)


def status() -> dict:
    return {
        "enabled": available(),
        "backend": _STATE["backend"],
        "loaded": bool(_STATE["backend"]),
        "failed": _STATE["failed"],
    }


def warmup() -> bool:
    """应用启动时后台调用，避免第一次提问多等模型加载"""
    if not available():
        return False
    try:
        return _load()
    except Exception:  # noqa: BLE001
        return False


def _sigmoid(x):
    import numpy as np

    return 1.0 / (1.0 + np.exp(-x))


def score(query: str, texts: List[str], batch_size: int = None) -> Optional[List[float]]:
    """给 (query, text) 打分，返回 0~1 的相关性概率；不可用时返回 None。

    bge-reranker 输出单个 logit，经 sigmoid 即「相关概率」。
    实测：真正能回答问题的条文 0.9+，词面相近但答非所问的 0.1~0.4。
    """
    if not texts or not available():
        return None
    if not _load():
        return None

    import config

    tok = _STATE["tok"]
    max_len = config.RERANK_MAX_LEN
    if not batch_size:
        batch_size = getattr(config, "RERANK_BATCH", 4)
    out: List[float] = []
    try:
        if _STATE["backend"] == "onnx":
            import numpy as np

            sess = _STATE["session"]
            names = _STATE["inputs"]
            for i in range(0, len(texts), batch_size):
                batch = [[query, t] for t in texts[i : i + batch_size]]
                enc = tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="np",
                )
                feeds = {k: np.asarray(enc[k]) for k in names if k in enc}
                logits = np.asarray(sess.run(None, feeds)[0]).reshape(-1)
                out.extend(_sigmoid(logits).astype(float).tolist())
        else:
            import torch

            model, device = _STATE["model"], _STATE["device"]
            with torch.inference_mode():
                for i in range(0, len(texts), batch_size):
                    batch = [[query, t] for t in texts[i : i + batch_size]]
                    enc = tok(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=max_len,
                        return_tensors="pt",
                    ).to(device)
                    logits = model(**enc).logits.view(-1)
                    out.extend(torch.sigmoid(logits).float().cpu().tolist())
    except Exception as e:  # noqa: BLE001  模型缺失/内存不足：本次退化，不中断问答
        _STATE["failed"] = f"score: {type(e).__name__}: {e}"
        _STATE["backend"] = ""
        return None
    return out
