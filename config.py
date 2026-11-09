"""全局配置中心 - 想换模型/换参数，只改这个文件即可"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # 项目根目录

# ============ Ollama 本地模型配置 ============
OLLAMA_BASE_URL = "http://127.0.0.1:11434"   # 用 IPv4 显式地址：localhost 可能解析到 ::1，ollama 只监听 127.0.0.1
EMBED_MODEL = "nomic-embed-text"             # 向量模型（本地，用于文本向量化）
CHAT_MODEL = "qwen2.5:7b"                    # 问答大模型（本地）。可切换: qwen2.5:3b / gemma3:4b / qwen2.5-coder:3b

# ---- 本地模型常驻（关键：Ollama 默认闲置 5 分钟就卸载模型，下次提问要重新加载 40 秒以上，
#      这是「回复慢」的首要原因。实测 qwen2.5:7b 冷加载首字 42s、热状态首字 0.3s）----
OLLAMA_KEEP_ALIVE = "30m"                    # 模型保持驻留时长；-1 表示永久（会长期占用约 5GB 显存）
OLLAMA_NUM_CTX = 3072                        # 上下文窗口。检索 5 块约 2800 字（≈1900 token），
                                             # 3072 有余量；设更大只会白占显存、拖慢 prefill
OLLAMA_NUM_PREDICT = 800                     # 单次回答最大 token 数，防止个别问题生成过长

# ============ 文本分块参数 ============
CHUNK_SIZE = 500                             # 每块最大字符数
CHUNK_OVERLAP = 80                           # 相邻块重叠字符数（保证上下文连贯）
LINE_OVERLAP_LINES = 2                       # txt/md 按行分块时，相邻块重叠行数（保证行号定位准确）

# ============ 检索参数 ============
SIMILARITY_THRESHOLD = 0.65                  # 融合分阈值(0~1)：低于该值的片段视为无关，会被过滤。
                                             # 融合分 = 0.40*向量相似度 + 0.60*BM25词面分。
                                             # 实测（200 篇规范 / 4.9 万块）：相关问题 0.84~0.99，
                                             # 无关问题（"今天天气怎么样""帮我写简历"）≤0.53
TOP_K = 5                                    # 检索返回的候选片段条数

# ============ 重排（Cross-Encoder）============
# 两段式检索：向量 + BM25 先粗排召回，再用 bge-reranker 精排。
# 粗排负责「不漏」，精排负责「排得准」——问「疏散距离」却命中「疏散宽度」这类错误只能靠精排纠正。
RERANK_ENABLED = True
RERANK_BACKEND = "onnx"                       # onnx（推荐，量化 ONNX，12 条 0.9 秒）/ torch（兜底）/ auto
RERANK_ONNX_DIR = os.path.join(BASE_DIR, "models", "bge-reranker-base-onnx")
RERANK_MODEL = "BAAI/bge-reranker-base"       # torch 兜底时才用，首次自动下载（约 1.1GB，走 hf-mirror）
RERANK_DEVICE = "cpu"                         # torch 后端的设备；本地问答模型已占显存，默认 CPU
RERANK_MAX_LEN = 112                          # 单条 (问题, 条文) 截断长度。CPU 上耗时随长度近线性，
                                              # 112 已覆盖条文开头的适用条件与数值，再长只是变慢
RERANK_TOP_N = 8                              # 进入精排的候选数。粗排已保证召回，精排只需在头部区分；
                                              # 实测 8 条一批（batch=4×2）约 0.5 秒，10 条要 1.3 秒
RERANK_BATCH = 4                              # 单批条数。实测同数量下 batch=4 比 8/10 快 30%，短序列填充浪费更少
RERANK_WEIGHT = 0.20                          # 精排对排序的有界影响：±0.10 微调。
                                              # 不做全量线性融合——精排分不清「通用规范 5.2.2」与
                                              # 「汽车库规范 5.2.2」，权重给大了会把点名的条文换错

# ============ 引擎切换（2026-11 迁移 LangChain + PGVector）============
# "langchain_pg"  -> LangChain LCEL 链 + PGVector（当前默认）
# "chroma_legacy" -> 旧自研链路 + Chroma（保留可回退，需注意向量模型不同）
ENGINE = "langchain_pg"

# ============ PostgreSQL + pgvector（Docker: pgvector-pg17）============
PG_HOST = "localhost"
PG_PORT = 5432
PG_DB = "vector_db"
PG_USER = "postgres"
PG_PASSWORD = os.getenv("PG_PASSWORD", "")   # 数据库口令从环境变量读取，勿硬编码
# SQLAlchemy 连接串（PGVector 用）
PG_DSN = f"postgresql+psycopg://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
# psycopg 原生连接串（统计/维护类 SQL 用）
PG_DSN_RAW = f"host={PG_HOST} port={PG_PORT} dbname={PG_DB} user={PG_USER} password={PG_PASSWORD}"
PG_COLLECTION = "rag_standards"       # 集合名（自定义表固定为 langchain_pg_embedding）
PG_PRE_DELETE = False                  # True = 启动时清空该集合（灌库脚本单独控制）

# ---- 向量模型（1024 维，与库中 vector(1024) 对齐）----
EMBED_MODEL_PG = "bge-m3"              # 中文/多语，1024 维，本地 Ollama
EMBED_DIM = 1024                       # 必须与 PGVector 表维度一致，否则写入报错
EMBED_BATCH_PG = 64                    # 单批向量化条数（实测 64 比 32 快 2.5 倍，128 反而略降）

# ---- 召回与融合参数（LangChain 检索组件）----
LC_VECTOR_K = 30                       # 向量召回条数
LC_BM25_K = 30                         # BM25 召回条数
LC_ENSEMBLE_WEIGHTS = [0.6, 0.4]       # [向量, BM25] 权重，EnsembleRetriever 加权 RRF
LC_RERANK_TOP_N = 10                   # 精排后保留的候选条数
LC_RERANK_MAX_CAND = 10                # 送进精排的候选上限（本机 6 核，10 条约 1.1s）
LC_RERANK_MIN = 0.15                   # 精排分下限：低于它视为无关（拒答依据）
LC_KEYWORD_GATE = 0.85                 # 精排分不到这个值时，必须与问题有词面交集，否则丢弃
                                       # （实测：无关问题精排分最高 0.61，相关问题 ≥0.95）
LC_TOP_K = 5                           # 最终交给大模型的片段条数
LC_CONTEXT_MAX_CHARS = 2600            # 上下文总字符上限，防止超 num_ctx

# ============ 回答模式 ============
# "rag"   = 规范问答（默认）：一次混合检索 + 一次作答，快、稳、可溯源
# "agent" = Agent 工具：模型自行选择工具，支持多步查询（较慢、消耗更多 token）
ANSWER_MODE = "rag"

# ============ 向量库（旧 Chroma 链路，ENGINE=chroma_legacy 时生效）============
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma")   # Chroma 持久化目录
COLLECTION_NAME = "local_rag_kb"                        # 向量集合名
DOCS_DIR = os.path.join(BASE_DIR, "docs")               # 上传文档存放目录

# ============ LLM 提供商（预留其他 API 厂商接入空间）============
# LLM_PROVIDER 可选值:
#   "ollama"              -> 本地模型（免费无需联网）
#   "openai_compatible"   -> 兼容 OpenAI 格式的云端 API（DeepSeek / 通义千问 / 豆包 / Moonshot 等均可）
#   "deepseek" 等厂商 ID 见下方 LLM_PROVIDERS
LLM_PROVIDER = "deepseek"

# ---- 云端 API 配置（仅当 LLM_PROVIDER="openai_compatible" 时生效）----
OPENAI_BASE_URL = "https://api.deepseek.com/v1"   # 换成你接入厂商的 base_url
OPENAI_API_KEY = ""                               # 填入你的 API Key
OPENAI_MODEL = "deepseek-chat"                    # 换成厂商模型名，如 qwen-plus / doubao-pro / moonshot-v1-8k

# ============ 问答模型多提供商（网页界面可切换）============
# 说明：
#   - 界面侧边栏选择提供商，切换立即生效，无需改代码；
#   - 云端 Key 三种填法（优先级从高到低）：界面输入并保存 -> 环境变量 -> 下面的 api_key 字段；
#   - Key 保存在本机 data/api_keys.json，不会上传；
#   - 无论用哪家问答模型，向量检索都在本地 Ollama 完成，切换后无需重建向量库。
LLM_PROVIDERS = {
    "ollama": {
        "label": "🏠 本地 Ollama（免费，数据不出本机）",
        "type": "ollama",
        "model": CHAT_MODEL,
    },
    "deepseek": {
        "label": "🐋 DeepSeek 云端（按量付费，约 1~2 分/次）",
        "type": "openai_compatible",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "key_env": "DEEPSEEK_API_KEY",
        "api_key": "",
        "key_url": "https://platform.deepseek.com/api_keys",
    },
    "dashscope": {
        "label": "☁️ 通义千问 Qwen·百炼（新用户有免费额度）",
        "type": "openai_compatible",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "key_env": "DASHSCOPE_API_KEY",
        "api_key": "",
        "key_url": "https://bailian.console.aliyun.com/?apiKey=1",
    },
    "zhipu": {
        "label": "🔮 智谱 GLM（glm-4-flash 免费）",
        "type": "openai_compatible",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-flash",
        "key_env": "ZHIPU_API_KEY",
        "api_key": "",
        "key_url": "https://open.bigmodel.cn/usercenter/apikeys",
    },
    "moonshot": {
        "label": "🌙 Moonshot Kimi（按量付费）",
        "type": "openai_compatible",
        "base_url": "https://api.moonshot.cn/v1",
        "model": "kimi-k2.6",
        "key_env": "MOONSHOT_API_KEY",
        "api_key": "",
        "key_url": "https://platform.kimi.ai/console/api-keys",
    },
}

# ============ 联网搜索（默认关闭，预留接入空间）============
# SEARCH_PROVIDER 可选值:
#   ""       -> 关闭（默认。知识库检索不到时，只提示"需要联网查询"，不会自动联网）
#   "bing"   -> 微软 Bing 搜索（需 SEARCH_API_KEY）
#   "tavily" -> Tavily 搜索（需 SEARCH_API_KEY）
#   "serpapi"-> SerpAPI 搜索（需 SEARCH_API_KEY）
SEARCH_PROVIDER = ""
SEARCH_API_KEY = ""
