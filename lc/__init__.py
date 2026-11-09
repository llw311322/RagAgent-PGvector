"""LangChain 组件层（PGVector + bge-m3 + LCEL）

模块划分：
- embeddings  : 向量模型工厂（bge-m3 / 1024 维，本地 Ollama）
- store       : PGVector 向量库（LangChain PGVector + 原生 SQL 维护）
- bm25        : PG 数据源上的 BM25 词面索引（复用旧索引类）
- ingest      : 文档加载 → 分块打标 → 批量写入
- retriever   : 向量召回 + BM25 召回 → EnsembleRetriever 融合 → 重排压缩
- prompt      : 提示词与上下文格式化
- chain       : LCEL 问答链（含追问改写、会话记忆、流式输出）
- tools       : 暴露给 Agent 的 LangChain 工具
- agent       : create_agent 工具调用代理
- service     : 供 Streamlit 调用的一层薄门面
"""
