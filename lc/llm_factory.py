"""LangChain 聊天模型工厂：ChatOllama / ChatOpenAI（统一走 BaseChatModel 接口）

与旧 llm.py 的区别：这里返回的是 LangChain 原生 ChatModel，
可以被 prompt | model | parser 管道、create_agent、工具绑定直接消费。
"""
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

import config
from llm import resolve_api_key


def get_chat_model(provider_id: Optional[str] = None, api_key: Optional[str] = None) -> BaseChatModel:
    provider_id = provider_id or config.LLM_PROVIDER
    preset = config.LLM_PROVIDERS.get(provider_id)
    if preset is None:
        raise ValueError(f"未知的模型提供商: {provider_id}")

    if preset["type"] == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=preset.get("model", config.CHAT_MODEL),
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.1,            # 规范问答要稳定复述，不要创造性
            num_ctx=config.OLLAMA_NUM_CTX,
            num_predict=config.OLLAMA_NUM_PREDICT,
            keep_alive=1800,            # 整数秒；局域网单机场景下常驻最省时间
        )

    if preset["type"] == "openai_compatible":
        key = resolve_api_key(provider_id, api_key)
        if not key:
            raise ValueError(
                f"「{preset['label']}」还未配置 API Key：请在侧边栏输入并保存，"
                f"或设置环境变量 {preset.get('key_env', '')}（获取地址: {preset.get('key_url', '')}）"
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=preset["model"],
            base_url=preset["base_url"],
            api_key=key,
            temperature=0.1,
            timeout=120,
            max_retries=2,
        )

    raise ValueError(f"不支持的提供商类型: {preset['type']}")
