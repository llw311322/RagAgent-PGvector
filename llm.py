"""大模型调用模块：本地 Ollama + OpenAI 兼容 API（DeepSeek/通义/豆包等），界面可切换"""
import os

import config


class BaseLLM:
    """LLM 统一抽象基类，方便扩展其他模型厂商"""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError

    def stream(self, system_prompt: str, user_prompt: str):
        """流式生成，逐段 yield 文本；默认退化为一次性生成"""
        yield self.generate(system_prompt, user_prompt)


class OllamaLLM(BaseLLM):
    """本地 ollama 模型（默认）"""

    def __init__(self, model: str = None):
        from langchain_ollama import ChatOllama
        self._llm = ChatOllama(
            model=model or config.CHAT_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.3,
            # 三个参数直接决定「回复快不快」：keep_alive 让模型不被 Ollama 卸载
            # （默认 5 分钟，重新加载要 40s+）；num_ctx 越小 KV cache 越小、prefill 越快；
            # num_predict 限制回答长度，避免长尾生成。
            keep_alive=config.OLLAMA_KEEP_ALIVE,
            num_ctx=config.OLLAMA_NUM_CTX,
            num_predict=config.OLLAMA_NUM_PREDICT,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        resp = self._llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        return resp.content

    def stream(self, system_prompt: str, user_prompt: str):
        from langchain_core.messages import HumanMessage, SystemMessage
        for chunk in self._llm.stream(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        ):
            if chunk.content:
                yield chunk.content


class OpenAICompatibleLLM(BaseLLM):
    """兼容 OpenAI 格式的云端 API（DeepSeek / 通义 / 豆包 / Moonshot 等）"""

    def __init__(self, base_url: str = None, api_key: str = None, model: str = None):
        from langchain_openai import ChatOpenAI
        self._llm = ChatOpenAI(
            base_url=base_url or config.OPENAI_BASE_URL,
            api_key=api_key or config.OPENAI_API_KEY,
            model=model or config.OPENAI_MODEL,
            temperature=0.3,
            timeout=120,
            max_retries=2,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        resp = self._llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        return resp.content

    def stream(self, system_prompt: str, user_prompt: str):
        from langchain_core.messages import HumanMessage, SystemMessage
        for chunk in self._llm.stream(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        ):
            if chunk.content:
                yield chunk.content


def resolve_api_key(provider_id: str, api_key: str = None) -> str:
    """解析某提供商的 API Key，优先级：显式传入 > 环境变量 > config 预填字段"""
    preset = config.LLM_PROVIDERS.get(provider_id) or {}
    return (
        api_key
        or os.environ.get(preset.get("key_env", ""), "")
        or preset.get("api_key", "")
    ).strip()


def warmup(provider: str = None) -> bool:
    """预热本地模型：提前把模型加载进显存，避免用户第一次提问干等 40 秒以上。
    云端接口无需预热。返回是否真的做了预热。"""
    provider = provider or config.LLM_PROVIDER
    preset = config.LLM_PROVIDERS.get(provider) or {}
    if preset.get("type") != "ollama":
        return False
    try:
        import requests

        requests.post(
            config.OLLAMA_BASE_URL.rstrip("/") + "/api/generate",
            json={
                "model": preset.get("model"),
                "prompt": "hi",
                "stream": False,
                "keep_alive": config.OLLAMA_KEEP_ALIVE,
                "options": {"num_predict": 1, "num_ctx": config.OLLAMA_NUM_CTX},
            },
            timeout=300,
        )
        return True
    except Exception:
        return False


def get_llm(provider: str = None, api_key: str = None) -> BaseLLM:
    """按提供商返回 LLM 实例；provider 为 config.LLM_PROVIDERS 的键，缺省用 LLM_PROVIDER"""
    provider = provider or config.LLM_PROVIDER
    preset = config.LLM_PROVIDERS.get(provider)
    if preset is None:
        raise ValueError(f"未知的模型提供商: {provider}")

    if preset["type"] == "ollama":
        return OllamaLLM(preset.get("model"))

    if preset["type"] == "openai_compatible":
        key = resolve_api_key(provider, api_key)
        if not key:
            raise ValueError(
                f"「{preset['label']}」还未配置 API Key：请在侧边栏输入并保存，"
                f"或设置环境变量 {preset.get('key_env', '')}（获取地址: {preset.get('key_url', '')}）"
            )
        return OpenAICompatibleLLM(preset["base_url"], key, preset["model"])

    raise ValueError(f"未知的提供商类型: {preset['type']}")
