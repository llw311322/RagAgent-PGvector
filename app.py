"""本地知识问答 Agent - Streamlit 主界面（Chroma 向量库 + Ollama 本地模型）

启动方式: streamlit run app.py
使用方式: 把 txt / md 文档放入 docs 文件夹 -> 点侧边栏扫描入库 -> 提问即可
"""
import html
import os
import sys

import requests
import streamlit as st

# 保证能 import 到同目录下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import document_loader

# 引擎开关：langchain_pg = LangChain LCEL + PGVector（当前默认）；chroma_legacy = 旧自研链路
USE_LC = getattr(config, "ENGINE", "chroma_legacy") == "langchain_pg"
if USE_LC:
    from lc import service as lc_service
else:
    import vector_store
    from chat import ask_stream
    from llm import get_llm

st.set_page_config(
    page_title="知识问答 Agent · 本地知识库",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

TXT_EXTS = {"txt", "md"}

SAMPLE_QUESTIONS = [
    "防火规范 5.2.2 条的具体要求是什么？",
    "防火墙的耐火极限有什么要求？",
    "疏散楼梯的最小净宽度是多少？",
    "哪些工业建筑的耐火等级不应低于二级？",
]


# ---------------- 后台预热 ----------------
# Streamlit 每次交互都会重跑脚本，用 cache_resource 保证只预热一次。
# 预热三件事：BM25 索引（105MB 缓存，约 8s）、Chroma 连接、本地大模型（约 40s）。
# 不预热的话，用户第一次提问会依次踩这三个坑，看起来就像「卡死了」。
@st.cache_resource(show_spinner=False)
def preload_all(provider_id: str) -> dict:
    if USE_LC:
        # 新链路：向量模型常驻 + BM25(pg) 载入 + 数据库连通，全部后台线程
        return lc_service.warmup(background=True)

    import threading

    import bm25_index
    import rerank
    from llm import warmup

    state = {"bm25": "pending", "llm": "pending", "rerank": "pending"}

    def work():
        try:
            vector_store.get_vectorstore()
            bm25_index.get_index()
            state["bm25"] = "ready"
        except Exception as e:  # noqa: BLE001
            state["bm25"] = f"failed: {e}"
        # 重排模型约 1.1GB，首次要从镜像下载；放后台线程里加载，不阻塞页面渲染
        try:
            state["rerank"] = "ready" if rerank.warmup() else "skipped"
        except Exception as e:  # noqa: BLE001
            state["rerank"] = f"failed: {e}"
        try:
            state["llm"] = "ready" if warmup(provider_id) else "skipped"
        except Exception as e:  # noqa: BLE001
            state["llm"] = f"failed: {e}"

    t = threading.Thread(target=work, daemon=True)
    t.start()
    return state


_PRELOAD = preload_all(config.LLM_PROVIDER)


# ---------------- Ollama 连接状态 ----------------
@st.cache_data(ttl=10)
def ollama_ok() -> bool:
    try:
        r = requests.get(config.OLLAMA_BASE_URL.rstrip("/") + "/api/tags", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


# ---------------- 模型提供商与 Key 管理 ----------------
KEYS_FILE = os.path.join(config.BASE_DIR, "data", "api_keys.json")


def load_keys() -> dict:
    try:
        import json
        with open(KEYS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_key(provider_id: str, key: str):
    import json
    keys = load_keys()
    if key:
        keys[provider_id] = key
    else:
        keys.pop(provider_id, None)
    os.makedirs(os.path.dirname(KEYS_FILE), exist_ok=True)
    with open(KEYS_FILE, "w", encoding="utf-8") as f:
        json.dump(keys, f, ensure_ascii=False)


# 当前选中的提供商（跨刷新记忆，默认取 config.LLM_PROVIDER）
if "active_provider" not in st.session_state:
    st.session_state.active_provider = getattr(config, "LLM_PROVIDER", "ollama")
# 横幅在侧边栏之前渲染，这里先同步下拉框的上一次选择，
# 否则切换模型后顶部蓝色框里的厂商标签会慢一轮才更新。
_sel = st.session_state.get("provider_select")
if _sel in config.LLM_PROVIDERS:
    st.session_state.active_provider = _sel
if "answer_mode" not in st.session_state:
    st.session_state.answer_mode = getattr(config, "ANSWER_MODE", "rag")
if "filters" not in st.session_state:
    st.session_state.filters = {}


# ---------------- docs 目录扫描入库 ----------------
def scan_docs_dir(progress_cb=None) -> tuple:
    """扫描 docs 目录下所有受支持的文件，新文件/已修改文件自动入库。
    progress_cb(done_files, total_files, cur_name, done_chunks, total_chunks) 用于刷新进度条。
    返回 (新入库块数, 明细列表)"""
    if USE_LC:
        return lc_service.scan_docs(progress_cb)

    os.makedirs(config.DOCS_DIR, exist_ok=True)
    files = [
        f for f in sorted(os.listdir(config.DOCS_DIR))
        if os.path.isfile(os.path.join(config.DOCS_DIR, f))
        and os.path.splitext(f)[1].lower() in document_loader.SUPPORTED_EXTS
    ]
    total = 0
    details = []
    existing = vector_store.load_existing_by_source()  # 全程只读一次库内索引
    for idx, f in enumerate(files, 1):
        path = os.path.join(config.DOCS_DIR, f)
        mtime = os.path.getmtime(path)
        try:
            docs = document_loader.load_document(path)
            for d in docs:
                d.metadata["mtime"] = mtime

            def cb(done: int, tot: int, _f=f, _idx=idx, _n=len(files)):
                if progress_cb:
                    progress_cb(_idx, _n, _f, done, tot)

            n = vector_store.add_documents(docs, existing=existing, progress_cb=cb)
            total += n
            if n > 0:
                details.append(f"✅ {f}: 新入库 {n} 块")
        except Exception as e:
            details.append(f"❌ {f}: 处理失败 - {e}")
    return total, details


# ---------------- 全局样式 ----------------
st.markdown(
    """
<style>
  /* 全局 */
  html, body, [class*="css"] { font-family: "Segoe UI", "Microsoft YaHei", -apple-system, sans-serif; }
  .block-container { padding-top: 1.2rem; max-width: 960px; }

  /* 隐藏 Streamlit 默认装饰 */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  header[data-testid="stHeader"] { background: transparent; }
  /* 隐藏右上角灰色英文（Running… / STOP / Deploy） */
  [data-testid="stStatusWidget"] { display: none !important; }
  [data-testid="stToolbar"] { display: none !important; }
  [data-testid="stDecoration"] { display: none !important; }
  [data-testid="stAppDeployButton"] { display: none !important; }

  /* 顶部横幅 */
  .app-hero {
    border-radius: 16px; padding: 1.1rem 1.5rem; margin-bottom: .6rem;
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 55%, #3b82f6 100%);
    color: #fff; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: .5rem;
  }
  .app-hero .t { font-size: 1.35rem; font-weight: 700; letter-spacing: .5px; }
  .app-hero .s { font-size: .82rem; opacity: .85; margin-top: .15rem; }
  .pill {
    display: inline-flex; align-items: center; gap: .35rem;
    background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.25);
    border-radius: 999px; padding: .18rem .7rem; font-size: .78rem; color: #fff;
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
  .dot.ok { background: #4ade80; box-shadow: 0 0 6px #4ade80; }
  .dot.bad { background: #f87171; box-shadow: 0 0 6px #f87171; }

  /* 欢迎卡片 */
  .welcome {
    border: 1px solid #e5e7eb; border-radius: 16px; padding: 1.4rem 1.6rem;
    background: linear-gradient(180deg, #fafbff 0%, #f3f6fc 100%); margin-bottom: .5rem;
  }
  .welcome h3 { margin: 0 0 .3rem 0; color: #1e3a5f; }
  .welcome p { color: #4b5563; font-size: .9rem; margin: .15rem 0; }
  .feature { display: flex; gap: .5rem; align-items: flex-start; margin-top: .5rem; font-size: .86rem; color: #374151; }
  .feature .ico { font-size: 1rem; }

  /* 证据卡片 */
  .ev-card {
    border: 1px solid #e5e7eb; border-left: 4px solid #2563eb; border-radius: 10px;
    background: #f8fafc; padding: .65rem .9rem; margin: .45rem 0;
  }
  .ev-head { display: flex; align-items: center; gap: .45rem; flex-wrap: wrap; margin-bottom: .35rem; }
  .ev-no {
    background: #2563eb; color: #fff; font-size: .72rem; font-weight: 700;
    border-radius: 6px; padding: .06rem .42rem;
  }
  .ev-src { font-weight: 600; color: #1f2937; font-size: .84rem; word-break: break-all; }
  .ev-tag {
    background: #e0e9ff; color: #1e40af; font-size: .72rem; border-radius: 6px; padding: .06rem .5rem;
  }
  .ev-score { font-size: .74rem; color: #059669; font-weight: 700; }
  .ev-body {
    font-family: "Cascadia Code", Consolas, "Microsoft YaHei", monospace;
    font-size: .8rem; color: #334155; white-space: pre-wrap; word-break: break-all;
    background: #fff; border: 1px dashed #dbe2ea; border-radius: 8px; padding: .5rem .7rem; max-height: 180px; overflow-y: auto;
  }
  .ev-loc { font-size: .75rem; color: #6b7280; margin-top: .3rem; }

  /* 聊天气泡 */
  [data-testid="stChatMessage"] { border-radius: 14px; }
  [data-testid="stChatMessageContent"] { border-radius: 14px; }
  /* 回答里的加粗标题与正文同字号，只保留加粗区别（避免标题过大、层级突兀） */
  [data-testid="stChatMessageContent"] h1,
  [data-testid="stChatMessageContent"] h2,
  [data-testid="stChatMessageContent"] h3,
  [data-testid="stChatMessageContent"] h4,
  [data-testid="stChatMessageContent"] h5,
  [data-testid="stChatMessageContent"] h6 {
    font-size: 1rem !important;
    font-weight: 700 !important;
    line-height: 1.7 !important;
    margin: .75rem 0 .35rem !important;
    padding: 0 !important;
  }
  [data-testid="stChatMessageContent"] p,
  [data-testid="stChatMessageContent"] li { font-size: 1rem; line-height: 1.75; }

  /* 侧边栏 */
  [data-testid="stSidebar"] { background: #f7f9fc; }
  [data-testid="stSidebar"] hr { margin: .6rem 0; }
  .stat-card {
    border-radius: 12px; padding: .55rem .8rem; background: #fff;
    border: 1px solid #e5e7eb; text-align: center;
  }
  .stat-card .num { font-size: 1.25rem; font-weight: 800; color: #1e3a5f; }
  .stat-card .lbl { font-size: .72rem; color: #6b7280; margin-top: .05rem; }

  /* 按钮 */
  .stButton > button { border-radius: 10px; font-size: .88rem; }

  /* 示例问题按钮 */
  .chip-btn .stButton > button {
    width: 100%; text-align: left; border: 1px solid #dbeafe; background: #eff6ff;
    color: #1e40af; border-radius: 10px; padding: .45rem .8rem; font-size: .85rem;
  }
  .chip-btn .stButton > button:hover { background: #dbeafe; border-color: #93c5fd; }

  /* 滚动条 */
  ::-webkit-scrollbar { width: 8px; height: 8px; }
  ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
</style>
""",
    unsafe_allow_html=True,
)


def esc(s: str) -> str:
    return html.escape(str(s))


# ---------------- 顶部横幅 ----------------
ok = ollama_ok()          # 仅用于 Ollama 相关提示


def provider_ok(provider_id: str) -> bool:
    """当前问答模型是否可用：本地看 Ollama 服务，云端看是否已配置 Key"""
    preset = config.LLM_PROVIDERS.get(provider_id, {})
    if preset.get("type") == "ollama":
        return ok
    if st.session_state.get("active_key"):
        return True
    return bool(load_keys().get(provider_id))


def provider_status_text(provider_id: str) -> str:
    preset = config.LLM_PROVIDERS.get(provider_id, {})
    if preset.get("type") == "ollama":
        return "已连接" if ok else "未启动"
    return "已就绪" if provider_ok(provider_id) else "未配置 Key"


def kb_stats() -> dict:
    """两套引擎的统计口径不同，这里统一成 docs/chunks/sources/threshold"""
    if USE_LC:
        s = lc_service.stats()
        return {
            "docs": s["docs"], "chunks": s["chunks"], "sources": s["sources"],
            "threshold": config.LC_RERANK_MIN,
        }
    s = vector_store.get_stats()
    s["threshold"] = config.SIMILARITY_THRESHOLD
    return s


stats = None
try:
    stats = kb_stats()
except Exception:
    pass

_provider = config.LLM_PROVIDERS.get(st.session_state.active_provider, {})
_model_label = _provider.get("label", "本地 Ollama")

st.markdown(
    f"""
<div class="app-hero">
  <div>
    <div class="t">📚 知识问答 Agent</div>
    <div class="s">先检索知识库再回答 ｜ 每条回答标注依据（文档 + 行号），查不到依据直接明说</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------- 侧边栏 ----------------
with st.sidebar:
    st.subheader("🤖 问答模型")
    _pids = list(config.LLM_PROVIDERS)
    _idx = _pids.index(st.session_state.active_provider) if st.session_state.active_provider in _pids else 0
    pid = st.selectbox(
        "提供商（切换立即生效）",
        _pids,
        index=_idx,
        format_func=lambda p: config.LLM_PROVIDERS[p]["label"],
        label_visibility="collapsed",
        key="provider_select",
    )
    st.session_state.active_provider = pid
    preset = config.LLM_PROVIDERS[pid]

    if preset["type"] == "openai_compatible":
        # Key 统一从 data/api_keys.json 读取，界面不再暴露输入框
        st.session_state.active_key = load_keys().get(pid, "")
    else:
        st.session_state.active_key = None
        if not ok:
            st.warning("Ollama 未启动，请运行 `ollama serve` 或启动托盘程序", icon="⚠️")

    if USE_LC:
        st.divider()
        st.subheader("🧭 回答模式")
        # 两种模式的说明放进悬停提示：不选择时不占版面，切换时可随时查看
        st.session_state.answer_mode = st.radio(
            "回答模式",
            ["rag", "agent"],
            horizontal=True,
            index=0 if st.session_state.get("answer_mode", "rag") == "rag" else 1,
            format_func=lambda m: "📖 规范问答" if m == "rag" else "🛠 Agent 工具",
            label_visibility="collapsed",
            key="answer_mode_select",
            help=(
                "📖 规范问答：检索一次后直接作答，约 2–5 秒，适合 90% 的查条文问题。\n\n"
                "🛠 Agent 工具：模型自己决定调用哪个工具（精确条文 / 规范清单 / 库统计），"
                "可多轮查询，约 10–20 秒，适合「库里有哪些防火规范」这类多步问题。"
            ),
        )

        with st.expander("🎯 检索筛选（走 SQL 元数据）", expanded=False):
            _domains = ["（不限）", "防火", "电气", "暖通燃气", "结构", "节能", "装饰装修",
                        "规划总图", "给排水", "无障碍", "综合"]
            _buildings = ["（不限）", "住宅", "办公", "酒店", "汽车库", "地下", "高层",
                          "工业", "教育", "医疗", "商业"]
            _classes = ["（不限）", "国家标准-强制性通用规范", "国家标准", "行业标准",
                        "地方标准", "政策文件", "参考资料"]
            _d = st.selectbox("专业主题", _domains, key="flt_domain")
            _b = st.selectbox("建筑类型", _buildings, key="flt_building")
            _c = st.selectbox("标准大类", _classes, key="flt_class")
            st.session_state.filters = {
                "domain": "" if _d.startswith("（") else _d,
                "building": "" if _b.startswith("（") else _b,
                "std_class": "" if _c.startswith("（") else _c,
            }
            _on = [v for v in st.session_state.filters.values() if v]
            if _on:
                st.caption("已启用：" + "、".join(_on))
            else:
                st.caption("未启用筛选（全库检索）")

    st.divider()
    st.subheader("📂 知识库管理")

    def run_ingest(title: str):
        """对 docs 目录做一次增量扫描入库（新文件/有修改的文件才处理），带进度条"""
        bar = st.progress(0.0, text=title)

        def on_file(done_f: int, total_f: int, name: str, done_c: int, total_c: int):
            frac = (done_f - 1 + (done_c / total_c if total_c else 1)) / total_f
            bar.progress(min(frac, 1.0), text=f"({done_f}/{total_f}) {name} ｜ 向量化 {done_c}/{total_c} 块")

        try:
            total, details = scan_docs_dir(on_file)
            bar.progress(1.0, text="入库完成")
            errs = [d for d in details if d.startswith("❌")]
            if USE_LC:
                lc_service.invalidate_stats()
            st.cache_data.clear()   # 刷新统计卡片与文档列表
            if total > 0:
                st.success(f"本次新入库 {total} 块\n\n" + "\n".join(details[-20:]))
            elif errs:
                st.error("以下文件处理失败：\n\n" + "\n".join(errs[:20]))
            else:
                st.info("没有需要新入库的文件（新文件或内容有修改的文件才会入库）")
        except Exception as e:
            st.error(f"入库失败：{e}\n\n请确认 Ollama 已启动（任务栏托盘图标或运行 `ollama serve`）")

    # 上传与扫描合并为同一个入口：拖入文件即自动入库；下方按钮可重新扫描 docs 目录
    with st.container(border=True):
        up_files = st.file_uploader(
            "⬆️ 拖拽文件到此处，自动入库",
            type=document_loader.SUPPORTED_EXTS,
            accept_multiple_files=True,
            help=f"支持 {', '.join(document_loader.SUPPORTED_EXTS)}；保存到 docs 目录后自动向量化",
        )
        if up_files:
            saved = []
            os.makedirs(config.DOCS_DIR, exist_ok=True)
            for f in up_files:
                dest = os.path.join(config.DOCS_DIR, os.path.basename(f.name))
                with open(dest, "wb") as fh:
                    fh.write(f.getbuffer())
                saved.append(os.path.basename(f.name))
            st.caption("已保存：" + "、".join(saved[:10]) + ("…" if len(saved) > 10 else ""))
            run_ingest("正在向量化上传的文件…")
        elif st.button("🔄 重新扫描 docs 目录", use_container_width=True):
            run_ingest("准备扫描…")

    st.divider()

    # 知识库状态 + 已入库文档管理（可搜索单独删除）
    try:
        if stats is None:
            stats = kb_stats()
        c1, c2, c3 = st.columns(3)
        for col, num, lbl in (
            (c1, stats["docs"], "文档"),
            (c2, stats["chunks"], "文本块"),
            (c3, f"{stats['threshold']}", "精排阈值" if USE_LC else "相似度阈值"),
        ):
            col.markdown(
                f'<div class="stat-card"><div class="num">{num}</div><div class="lbl">{lbl}</div></div>',
                unsafe_allow_html=True,
            )

        with st.expander(f"🗂 已入库文档（{len(stats['sources'])}）", expanded=False):
            kw = st.text_input("🔍 按文件名筛选", "", placeholder="输入关键字过滤…").strip().lower()
            shown = [s for s in stats["sources"] if kw in s.lower()] if kw else stats["sources"]
            for i, s in enumerate(shown[:80]):
                row = st.columns([0.86, 0.14])
                row[0].markdown(f"<div style='font-size:.78rem; word-break:break-all;'>• {esc(s)}</div>", unsafe_allow_html=True)
                if row[1].button("删", key=f"del_{i}_{hash(s)}", help=f"删除 {s} 的全部向量块"):
                    n = lc_service.delete_source(s) if USE_LC else vector_store.delete_source(s)
                    st.toast(f"已删除 {s}（{n} 块）", icon="🗑")
                    st.cache_data.clear()
                    st.rerun()
    except Exception:
        st.warning("向量库暂不可用（Ollama 未启动？）", icon="⚠️")

# ---------------- 会话状态 ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []   # {role, content, evidence?}


# ---------------- 证据卡片 ----------------
def render_evidence(evidence: list):
    """展示回答依据；txt/md 支持展开定位原文（高亮所引行）"""
    if not evidence:
        return
    with st.expander(f"📎 回答依据（{len(evidence)} 个片段 · 可定位原文）", expanded=False):
        for ev in evidence:
            loc = f"第 {ev['start_line']}-{ev['end_line']} 行" if ev["start_line"] else f"片段 #{ev['no']}"
            heading = (ev.get("heading") or "").strip()
            path_html = (
                f'<div class="ev-path" style="font-size:.78rem;opacity:.72;margin:.15rem 0 .35rem;'
                f'line-height:1.45;">🏷 {esc(heading)}</div>'
                if heading else ""
            )
            # 结构化标签徽章：标准大类 / 专业主题 / 适用建筑类型（打标字段可能缺失，逐个判空）
            badges = []
            code = (ev.get("standard_code") or "").strip()
            clause = (ev.get("clause_no") or "").strip()
            if code or clause:
                label = " ".join(x for x in (code, f"第{clause}条" if clause else "") if x)
                badges.append(f'<span class="ev-tag">📐 {esc(label)}</span>')
            if ev.get("is_mandatory"):
                badges.append('<span class="ev-tag">❗ 强制性条文</span>')
            # 草案/已废止必须显式提示：库里《建筑防火通用规范》的完整全文只有征求意见稿，
            # 不标注的话用户会把它当成现行强条直接用于设计。
            status = (ev.get("doc_status") or "").strip()
            if status == "draft" or (ev.get("valid") is False and not status):
                badges.append('<span class="ev-tag" style="background:#7c2d12;color:#fed7aa;border-color:#9a3412">⚠ 征求意见稿</span>')
            elif status == "abolished":
                badges.append('<span class="ev-tag" style="background:#7c2d12;color:#fed7aa;border-color:#9a3412">⚠ 已废止</span>')
            src_type = (ev.get("source_type") or "").strip()
            if src_type and src_type != "条文正文":
                badges.append(f'<span class="ev-tag">📝 {esc(src_type)}</span>')
            for icon, key in (("🏛", "std_class"), ("🧭", "domain")):
                v = (ev.get(key) or "").strip()
                if v and v != "综合":
                    badges.append(f'<span class="ev-tag">{icon} {esc(v)}</span>')
            bl = ev.get("buildings") or []
            if isinstance(bl, list) and bl and bl[0] != "通用":
                badges.append(f'<span class="ev-tag">🏢 {esc("/".join(map(str, bl[:3])))}</span>')
            badge_html = "".join(badges)
            head = (
                f'<div class="ev-head"><span class="ev-no">[{ev["no"]}]</span>'
                f'<span class="ev-src">{esc(ev["source"])}</span>'
                f'<span class="ev-tag">📄 {esc(loc)}</span>'
                f'<span class="ev-score">相关度 {min(ev["score"], 1.0):.0%}</span></div>'
                f'{badge_html}'
                f'{path_html}'
                f'<div class="ev-body">{esc(ev["content"])}</div>'
            )
            # txt/md：从 docs 目录读原文件，展示所引行号前后各 3 行，引用行加 👉 标记
            if ev["type"] in TXT_EXTS and ev["start_line"]:
                src_path = os.path.join(config.DOCS_DIR, ev["source"])
                if os.path.isfile(src_path):
                    try:
                        raw = document_loader._load_txt(src_path)
                        lines = raw.replace("\r\n", "\n").split("\n")
                        lo, hi = ev["start_line"], ev["end_line"]
                        ctx_lines = []
                        for i in range(max(1, lo - 3), min(len(lines), hi + 3) + 1):
                            mark = "👉" if lo <= i <= hi else "  "
                            ctx_lines.append(f"{mark} {i:>4} | {lines[i - 1]}")
                        head += (
                            f'<div class="ev-loc">原文定位：{esc(ev["source"])} 第 {lo}-{hi} 行'
                            f'（👈 所引行，含前后 3 行上下文）</div>'
                            f'<div class="ev-body" style="max-height:260px;">{esc(chr(10).join(ctx_lines))}</div>'
                        )
                    except Exception:
                        pass
            st.markdown(f'<div class="ev-card">{head}</div>', unsafe_allow_html=True)


# ---------------- 欢迎页（已移除，直接进入提问） ----------------
pending = None

# ---------------- 历史消息 ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_evidence(msg.get("evidence", []))

# 工具条
if st.session_state.messages:
    top = st.columns([0.12, 0.88])
    if top[0].button("🧹 清空对话"):
        st.session_state.messages = []
        st.rerun()

question = st.chat_input("请输入你的问题…") or pending

# ---------------- 提问处理 ----------------
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        evidence = []
        full = ""

        if USE_LC:
            box = st.empty()        # 流式回答容器
            tip = st.empty()        # 追问改写 / 工具调用提示
            mode = st.session_state.get("answer_mode", "rag")
            # 只取本轮之前的对话作为多轮上下文
            hist = [(m["role"], m["content"]) for m in st.session_state.messages[:-1]]

            def gen():
                for ev in lc_service.ask_stream(
                    question,
                    provider=st.session_state.active_provider,
                    api_key=st.session_state.get("active_key"),
                    filters=st.session_state.get("filters") or {},
                    history=hist,
                    mode=mode,
                ):
                    typ = ev.get("type")
                    if typ == "evidence":
                        evidence.extend(ev["evidence"])
                    elif typ == "delta":
                        yield ev["text"]
                    elif typ == "no_evidence":
                        yield ev["message"]
                        return
                    elif typ == "error":
                        yield f"❌ {ev['message']}"
                        return
                    elif typ == "notice":
                        tip.caption(f"ℹ️ {ev['message']}")
                    elif typ == "query":
                        tip.caption(f"🔗 结合上下文改写为：{ev['text']}")
                    elif typ == "tool":
                        tip.caption(f"🛠 调用工具：`{ev['name']}`")

            try:
                full = st.write_stream(gen())
            except Exception as e:  # noqa: BLE001
                full = f"❌ 生成失败：{e}"
                if st.session_state.active_provider == "ollama":
                    full += f"\n\n请确认 Ollama 已启动、模型已下载（`ollama pull {config.CHAT_MODEL}`）。"
            box.markdown(full)
            # Agent 模式的工具结果不单独抽取依据（工具已把来源写进答案），故可能为空
            render_evidence(evidence)
        else:
            try:
                llm = get_llm(st.session_state.active_provider, st.session_state.get("active_key") or None)
            except ValueError as e:
                llm = None
                st.error(f"❌ {e}")
            except Exception as e:
                llm = None
                st.error(f"❌ 模型初始化失败：{e}")

            if llm is not None:
                box = st.empty()          # 流式回答容器
                try:
                    def gen_legacy():
                        """把 chat.ask_stream 的事件流转成 write_stream 需要的文本迭代器"""
                        for item in ask_stream(question, llm=llm):
                            if item["type"] == "no_evidence":
                                yield item["message"]
                                return
                            if item["type"] == "chunks":
                                from chat import evidence_of
                                evidence.extend(evidence_of(item["chunks"]))
                            elif item["type"] == "delta":
                                yield item["text"]

                    full = st.write_stream(gen_legacy())
                except Exception as e:  # noqa: BLE001
                    full = f"❌ 生成失败：{e}"
                    if st.session_state.active_provider == "ollama":
                        full += f"\n\n请确认 Ollama 已启动、模型已下载（`ollama pull {config.CHAT_MODEL}`）。"

                box.markdown(full)
                render_evidence(evidence)

    if full:
        st.session_state.messages.append(
            {"role": "assistant", "content": full, "evidence": evidence}
        )
