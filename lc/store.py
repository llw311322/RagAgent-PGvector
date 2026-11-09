"""PGVector 向量库：LangChain PGVector 负责向量读写，原生 SQL 负责统计与维护

表结构由 langchain-postgres 自动创建：
  langchain_pg_collection(uuid, name, cmetadata)
  langchain_pg_embedding(id uuid, collection_id uuid, embedding vector(1024),
                         document text, cmetadata jsonb)

为什么维护类操作走原生 SQL：
- PGVector 的 delete 只支持按 id，按 source 删除要先查 id 再删，多一次往返；
- 统计（按标准大类/专业主题分组、规范清单）本质是 SQL 聚合，
  这是换到 PG 相对 Chroma 最实在的收益之一。
"""
from typing import Dict, List, Optional

import psycopg
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import DistanceStrategy

import config
from lc import embeddings

_STORE: Optional[PGVector] = None


def get_store(refresh: bool = False) -> PGVector:
    """PGVector 单例。refresh=True 用于灌库/清空后重建连接。"""
    global _STORE
    if _STORE is None or refresh:
        _STORE = PGVector(
            embeddings=embeddings.get_embeddings(),
            connection=config.PG_DSN,
            collection_name=config.PG_COLLECTION,
            embedding_length=config.EMBED_DIM,
            distance_strategy=DistanceStrategy.COSINE,
            use_jsonb=True,
            create_extension=False,   # 扩展已由镜像装好，避免权限问题
            pre_delete_collection=config.PG_PRE_DELETE,
        )
        # 触发一次连接，尽早暴露 DSN/权限问题
        _STORE.create_collection()
    return _STORE


# ---------------- 原生 SQL 维护 ----------------
def _sql(query: str, params: tuple = (), fetch: bool = True):
    with psycopg.connect(config.PG_DSN_RAW, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            if fetch and cur.description is not None:
                return cur.fetchall()
            return None


def _collection_id() -> Optional[str]:
    rows = _sql(
        "SELECT uuid FROM langchain_pg_collection WHERE name = %s",
        (config.PG_COLLECTION,),
    )
    return str(rows[0][0]) if rows else None


def table_ready() -> bool:
    try:
        rows = _sql(
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'langchain_pg_embedding'",
            fetch=True,
        )
        return bool(rows)
    except Exception:
        return False


def stats() -> Dict[str, object]:
    """文档数 / 块数 / 集合名，供界面顶栏与侧栏展示"""
    cid = _collection_id()
    if not cid:
        return {"docs": 0, "chunks": 0, "sources": [], "collection": config.PG_COLLECTION}
    row = _sql(
        "SELECT count(*), count(DISTINCT cmetadata->>'source') "
        "FROM langchain_pg_embedding WHERE collection_id = %s",
        (cid,),
    )
    chunks, docs = (row[0] if row else (0, 0))
    return {
        "docs": int(docs or 0),
        "chunks": int(chunks or 0),
        "sources": [r[0] for r in sources()],
        "collection": config.PG_COLLECTION,
    }


def sources() -> List[tuple]:
    """[(source, 块数, mtime)]，按文件名排序"""
    cid = _collection_id()
    if not cid:
        return []
    return _sql(
        "SELECT cmetadata->>'source' AS src, count(*) AS n, "
        "       max((cmetadata->>'mtime')::float8) AS mt "
        "FROM langchain_pg_embedding WHERE collection_id = %s "
        "GROUP BY 1 ORDER BY 1",
        (cid,),
    )


def existing_sources() -> Dict[str, tuple]:
    """{source: (mtime, 块数)}，用于增量入库判断（mtime 未变则跳过）"""
    return {r[0]: (r[2] or 0, int(r[1])) for r in sources() if r[0]}


def delete_sources(names: List[str]) -> int:
    cid = _collection_id()
    if not cid or not names:
        return 0
    rows = _sql(
        "WITH d AS (DELETE FROM langchain_pg_embedding "
        "WHERE collection_id = %s AND cmetadata->>'source' = ANY(%s) RETURNING 1) "
        "SELECT count(*) FROM d",
        (cid, names),
    )
    return int(rows[0][0]) if rows else 0


def delete_source(name: str) -> int:
    return delete_sources([name])


def clear() -> None:
    cid = _collection_id()
    if cid:
        _sql("DELETE FROM langchain_pg_embedding WHERE collection_id = %s", (cid,))


def drop_collection() -> None:
    """彻底删除集合（表保留），用于全量重建"""
    cid = _collection_id()
    if cid:
        _sql("DELETE FROM langchain_pg_embedding WHERE collection_id = %s", (cid,))
        _sql("DELETE FROM langchain_pg_collection WHERE uuid = %s", (cid,))
    global _STORE
    _STORE = None


# ---------------- 结构化查询（元数据即 SQL 字段）----------------
def distribution(field: str, limit: int = 12) -> List[tuple]:
    """按某个元数据字段分组计数，如 std_class / domain / buildings"""
    cid = _collection_id()
    if not cid:
        return []
    return _sql(
        "SELECT cmetadata->>%s AS k, count(*) FROM langchain_pg_embedding "
        "WHERE collection_id = %s GROUP BY 1 ORDER BY 2 DESC LIMIT %s",
        (field, cid, limit),
    )


def list_standards(keyword: str = "", limit: int = 40) -> List[tuple]:
    """规范清单：[(标准名, 标准号, 块数)]，按块数降序"""
    cid = _collection_id()
    if not cid:
        return []
    kw = f"%{keyword.strip()}%"
    return _sql(
        "SELECT cmetadata->>'standard_name' AS name, "
        "       cmetadata->>'standard_code' AS code, count(*) AS n "
        "FROM langchain_pg_embedding WHERE collection_id = %s "
        "AND (%s = '%%' OR cmetadata->>'standard_name' ILIKE %s "
        "     OR cmetadata->>'source' ILIKE %s) "
        "AND cmetadata->>'standard_name' IS NOT NULL "
        "GROUP BY 1,2 ORDER BY 3 DESC LIMIT %s",
        (cid, kw, kw, kw, limit),
    )


def find_clause(clause_no: str, standard_code: str = "", limit: int = 6) -> List[tuple]:
    """精确条文查询：[(cmetadata, document)]

    先按「条文号完全相等」查（绝大多数情况够用），查不到才退回 ILIKE 模糊匹配。
    不用一条 SQL 同时做两件事：ILIKE '%5.2.2%' 会把 15.2.2 / 25.2.2 全带进来，
    候选被撑到几百条，真正那一条反而被 LIMIT 截掉。
    """
    cid = _collection_id()
    if not cid:
        return []
    clause = clause_no.strip().strip("第条 ")
    code_like = f"%{standard_code.strip()}%" if standard_code.strip() else "%"
    order = (
        "ORDER BY (cmetadata->>'std_class' = '国家标准-强制性通用规范') DESC, "
        "         (cmetadata->>'source_type' = '条文正文') DESC, "
        "         (cmetadata->>'is_mandatory')::text = 'true' DESC, "
        "         length(document) ASC LIMIT %s"
    )
    common = (
        "SELECT cmetadata, document FROM langchain_pg_embedding WHERE collection_id = %s "
        "AND (cmetadata->>'standard_code' ILIKE %s OR cmetadata->>'standard_name' ILIKE %s) "
    )
    rows = _sql(
        common + "AND (cmetadata->>'clause_no' = %s OR cmetadata->>'article' = %s) " + order,
        (cid, code_like, code_like, clause, clause, limit),
    )
    if rows:
        return rows
    return _sql(
        common + "AND cmetadata->>'clause_no' ILIKE %s " + order,
        (cid, code_like, code_like, f"%{clause}%", limit),
    ) or []


def health() -> dict:
    try:
        _sql("SELECT 1")
        cid = _collection_id()
        return {"ok": True, "collection": config.PG_COLLECTION, "collection_id": cid}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)[:200]}
