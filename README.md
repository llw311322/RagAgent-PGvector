# 建筑规范知识问答 Agent（LangChain + PGVector）

> **合规说明**：本仓库仅包含代码与框架，不含任何规范全文、标准图集或受版权保护的资料；知识库文档需由使用者自行准备合法来源，放入 `docs/`（已在 .gitignore 中排除）。

本地运行的 RAG 知识库问答系统：把建筑规范文档放入 `docs/`，自动清洗、按条文分块、打元数据标签，
写入 PostgreSQL（pgvector）向量库；提问时双路召回 + 精排，再由大模型作答，**每条结论都带规范编号、条文号与原文行号**，查不到依据直接说明。

## ✨ 技术栈

| 层 | 组件 |
|---|---|
| 编排框架 | **LangChain 1.x**（LCEL 管道、`create_agent` 工具调用、`ChatPromptTemplate`、`StrOutputParser`） |
| 检索组件 | `PGVector`（LangChain 官方 PG 集成）、`EnsembleRetriever`（加权 RRF 融合）、`ContextualCompressionRetriever`（精排压缩） |
| 向量库 | **PostgreSQL 17 + pgvector 0.8.7**（Docker 容器 `pgvector-pg17`，库 `vector_db`，集合 `rag_standards`） |
| 向量模型 | **bge-m3**（1024 维，本地 Ollama） |
| 精排模型 | **BAAI/bge-reranker-base**（ONNX 量化，CPU 约 0.5s/批） |
| 词面检索 | 中文二元组 BM25（numpy 倒排 + pickle 缓存，查询 <16ms） |
| 问答模型 | 本地 `qwen2.5:7b`，或 DeepSeek / 通义 / 智谱 / Kimi（界面切换） |
| 界面 | Streamlit |

## 📁 目录结构

```
<项目根目录>/
├─ app.py                  Streamlit 界面（问答 / 依据卡片 / 筛选 / 模型切换 / 入库）
├─ config.py               配置中心（PG 连接、模型、召回与阈值参数）
├─ lc\                     ← LangChain 组件层（新）
│  ├─ llm_factory.py       ChatOllama / ChatOpenAI 工厂
│  ├─ embeddings.py        bge-m3 向量模型（1024 维，单例 + 维度校验）
│  ├─ store.py             PGVector 读写 + 原生 SQL 统计/维护
│  ├─ bm25.py              PG 数据源上的 BM25 索引
│  ├─ ingest.py            加载 → 分块打标 → 批量向量化 → 写入
│  ├─ retriever.py         向量召回 + BM25 召回 + Ensemble 融合 + 精排压缩
│  ├─ prompt.py            防幻觉提示词、上下文格式化、依据抽取
│  ├─ chain.py             LCEL 问答链（追问改写 / 流式 / 会话记忆）
│  ├─ tools.py             4 个 LangChain 工具（检索/条文/清单/统计）
│  ├─ agent.py             create_agent 工具调用代理
│  └─ service.py           给 Streamlit 的门面
├─ 旧链路（ENGINE=chroma_legacy 时启用）
│  ├─ chat.py / retriever.py / vector_store.py / bm25_index.py
├─ meta_tagger.py          清洗规则 + 结构化元数据打标（两条链路共用）
├─ text_splitter.py        结构感知分块：一条条文一块 + 行号定位（共用）
├─ document_loader.py      txt/md/pdf/docx/xlsx 解析（共用）
├─ rerank.py               bge-reranker ONNX / torch 推理封装
├─ llm.py                  旧 LLM 适配层（回退用）
├─ tools\
│  ├─ pg_reindex.py        全量/增量重建 PG 库（--scan / --stats）
│  ├─ reindex.py           旧 Chroma 链路重建
│  ├─ tag_rebuild.py       只重打标不重嵌入
│  └─ bench.py / bench_llm.py  性能基准
├─ docs\                   规范文档（200 篇 markdown）
├─ data\                   bm25_pg.pkl / api_keys.json（向量本体在 PostgreSQL）
├─ models\                 bge-reranker-base-onnx
└─ venv\
```

## 🚀 快速开始

```bat
:: 1. 启动 pgvector（Docker Desktop 需已运行）
docker start pgvector-pg17
:: 2. 启动 ollama（拉过 bge-m3 与 qwen2.5:7b）
ollama serve
:: 3. 首次全量灌库（约 15 分钟，可按文件增量重跑）
venv\Scripts\python tools\pg_reindex.py
:: 4. 启动界面
venv\Scripts\streamlit run app.py
```

打开 `http://localhost:8501`，把文档放进 `docs/` 后点侧栏「扫描 docs 目录并入库」即可增量更新。

## 🔍 检索链路

```
问题 ─┬─ PGVector 语义召回（cosine, top30, SQL 元数据过滤）
      └─ BM25 词面召回（top30）
            └─ EnsembleRetriever 加权 RRF（向量 0.6 / BM25 0.4）
                  └─ bge-reranker 精排（top10 → 阈值 0.15 过滤）
                        └─ top5 拼上下文 → 大模型 → 带 [n] 引用的答案
```

**为什么还要 BM25**：向量对「5.2.2」「3.00h」「GB 50016」这类精确符号不敏感，
而规范问答恰恰大量依赖条文号与数值，词面召回是硬需求。

**为什么融合用 RRF 而不是直接加权分数**：余弦相似度与 BM25 分量纲不同（bge-m3 相关块约 0.6~0.8，
BM25 原始分可到 40+），RRF 只看排名，不需要归一化假设；绝对相关度交给精排分判断。

## 🧩 LangChain 能力清单

| 能力 | 用法 |
|---|---|
| LCEL 管道 | `ChatPromptTemplate \| ChatOllama \| StrOutputParser`，`.stream()` 直接产出增量 |
| 多轮记忆 | `InMemoryChatMessageHistory` + 追问改写链（把「那高层呢」改写成独立问题再检索） |
| 结构化检索 | `PGVector.similarity_search_with_score(filter={...})`，元数据下推 SQL |
| 融合检索 | `EnsembleRetriever(weights=[0.6, 0.4], c=60)` |
| 重排 | 自定义 `BaseDocumentCompressor` 包装 bge-reranker，接入 `ContextualCompressionRetriever` |
| 自定义检索器 | `PGKeywordRetriever(BaseRetriever)` 把 BM25 索引接进 LangChain 生态 |
| 工具 | `@tool` 定义：`search_standards` / `get_clause` / `list_standards` / `kb_stats` |
| Agent | `langchain.agents.create_agent(model, tools, system_prompt=...)`，界面可切换模式 |

## 🗄 数据库

表由 `langchain-postgres` 自动创建（首次写入时）：

```sql
langchain_pg_collection(uuid, name, cmetadata)
langchain_pg_embedding(id uuid, collection_id uuid,
                       embedding vector(1024), document text, cmetadata jsonb)
```

维护类操作直接用 SQL（`lc/store.py`）：按 `source` 删除、按 `std_class`/`domain` 分组统计、条文号精确查询。
这正是换到 PG 相对 Chroma 的主要收益——元数据过滤与聚合不再受 Chroma `where` 语法和变量上限限制。

```sql
-- 例：看看哪些规范块最多
SELECT cmetadata->>'standard_name', count(*) FROM langchain_pg_embedding
GROUP BY 1 ORDER BY 2 DESC LIMIT 10;
```

## 🧹 数据清洗与元数据

入库时自动清洗并打标（`meta_tagger.py`），标签写入 `cmetadata`，可直接用于检索过滤：

| 字段 | 取值示例 |
|---|---|
| `std_class` | 国家标准-强制性通用规范 / 国家标准 / 行业标准 / 地方标准 / 政策文件 / 参考资料 |
| `domain` | 防火 / 电气 / 结构 / 暖通燃气 / 给排水 / 节能 / 无障碍 / 规划总图 / 装饰装修 / 综合 |
| `buildings` | 住宅 / 办公 / 酒店 / 汽车库 / 地下 / 高层 / 工业 / 教育 / 医疗 / 商业（多标签） |
| `clause_no` / `standard_code` | `5.2.2` / `GB 55037-2022` |
| `is_mandatory` / `valid` / `doc_status` | 强制性条文、是否现行、草案/废止 |
| `quality` / `garble` | 乱码页与碎片块的质检结果（非 ok 不入库） |

## 🛠 常用命令

```bat
venv\Scripts\python tools\pg_reindex.py            :: 全量重建（清空集合重灌）
venv\Scripts\python tools\pg_reindex.py --scan     :: 增量（只处理新增/修改文件）
venv\Scripts\python tools\pg_reindex.py --stats    :: 查看规模与分布
venv\Scripts\python tools\tag_rebuild.py --apply   :: 只重打标（Chroma 旧链路）
```

## ⚙️ 关键配置（config.py）

| 配置项 | 说明 | 默认 |
|---|---|---|
| `ENGINE` | `langchain_pg` / `chroma_legacy` | `langchain_pg` |
| `PG_DSN` | PostgreSQL 连接串 | `postgresql+psycopg://postgres:postgres@localhost:5432/vector_db` |
| `PG_COLLECTION` | 集合名 | `rag_standards` |
| `EMBED_MODEL_PG` / `EMBED_DIM` | 向量模型 / 维度 | `bge-m3` / `1024` |
| `LC_VECTOR_K` / `LC_BM25_K` | 两路召回条数 | 30 / 30 |
| `LC_ENSEMBLE_WEIGHTS` | 融合权重 | `[0.6, 0.4]` |
| `LC_RERANK_MIN` | 精排阈值（低于则拒答） | `0.15` |
| `LC_TOP_K` | 交给模型的片段数 | 5 |
| `CHUNK_SIZE` | 分块上限 | 500 |

## ❓ 常见问题

- **启动报维度错误**：模型输出维度与 `vector(1024)` 不一致；确认 `ollama list` 里有 `bge-m3`，或改 `EMBED_DIM` 并重建集合。
- **检索为空**：先 `--stats` 看库是否为空；再检查左侧「检索筛选」是否把范围限死。
- **回答很慢**：本地 7B 首次加载约 60s（之后常驻 30 分钟）；换 DeepSeek 等云端模型首字约 2.8s。
- **改了分块规则**：必须全量重建（`pg_reindex.py`），因为块边界变了，向量也要重算。
- **想回退旧链路**：`config.ENGINE = "chroma_legacy"`，但 Chroma 库是用 nomic-embed-text（768 维）建的，需先重建。
