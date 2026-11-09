"""冒烟自检脚本：验证 RAG 全流程是否可用（文档解析->分块->向量化->检索->模型回答）

用法: venv\\Scripts\\python test.py
"""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


def step(title):
    print(f"\n{'=' * 50}\n{title}\n{'=' * 50}")


# [1] 检查依赖
step("[1/5] 检查核心依赖")
try:
    import langchain  # noqa
    import langchain_chroma  # noqa
    import langchain_ollama  # noqa
    import chromadb  # noqa
    import streamlit  # noqa
    import pypdf  # noqa
    import docx  # noqa
    import openpyxl  # noqa
    print("✅ 全部核心依赖导入成功")
except Exception as e:
    print(f"❌ 依赖导入失败: {e}")
    sys.exit(1)

# [2] 检查 ollama 服务与模型
step(f"[2/5] 检查 ollama 服务 ({config.OLLAMA_BASE_URL})")
try:
    with urllib.request.urlopen(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5) as r:
        models = [m["name"] for m in json.loads(r.read().decode()).get("models", [])]
    print(f"✅ ollama 运行中，已安装: {models}")
    _base_models = {m.split(":")[0] for m in models}  # 兼容 ollama 返回的 :latest 标签
    _emb, _chat = config.EMBED_MODEL.split(":")[0], config.CHAT_MODEL.split(":")[0]
    if _emb not in _base_models:
        print(f"⚠️  缺少向量模型 {config.EMBED_MODEL}，请运行: ollama pull {config.EMBED_MODEL}")
    if _chat not in _base_models:
        print(f"⚠️  缺少问答模型 {config.CHAT_MODEL}，请运行: ollama pull {config.CHAT_MODEL}")
except Exception as e:
    print(f"❌ ollama 未运行或无法访问: {e}")
    print("  请先启动 ollama（任务栏托盘图标，或运行 ollama serve）")
    sys.exit(1)

# [3] 文档加载 + 分块
step("[3/5] 测试文档加载与自动分块")
sample_path = os.path.join(config.DOCS_DIR, "_自检样例.txt")
sample_text = (
    "本地RAG问答机器人说明：\n"
    "本系统用于对企业内部文档进行智能问答。它支持上传txt、word、excel和pdf文档。\n"
    "系统会自动将文档切割成多个文本块，并通过向量模型将文本块转化为向量，存入Chroma向量数据库。\n"
    "当用户提问时，系统会先检索知识库中最相关的文本片段，再交给大模型生成精炼回答。\n"
    "相似度阈值用于过滤无关内容，确保回答不冗余、不串味。\n" * 3
)
with open(sample_path, "w", encoding="utf-8") as f:
    f.write(sample_text)
print(f"✅ 已生成样例文档: {sample_path}")

import document_loader
import text_splitter

docs = document_loader.load_document(sample_path)
chunks = text_splitter.split_documents(docs)
print(f"✅ 解析 {len(docs)} 个文档 -> 自动切分为 {len(chunks)} 个文本块")
if not chunks:
    print("❌ 文本分块结果为空")
    sys.exit(1)

# [4] 向量化入库 + 检索
step("[4/5] 测试向量化入库与语义检索")
import vector_store
from retriever import retrieve

n = vector_store.add_documents(chunks)
stats = vector_store.get_stats()
print(f"✅ 已入库 {n} 块，知识库现有 {stats['chunks']} 块")

query = "系统支持哪些文档格式？"
scored = retrieve(query)
if scored:
    print(f"✅ 检索命中 {len(scored)} 个片段，最高相关度 {scored[0][1]}")
    for doc, score in scored[:3]:
        print(f"   - 相关度 {score:.3f} | 来源 {doc.metadata.get('source')} | {doc.page_content[:50]}...")
else:
    print("⚠️ 未检索到高于阈值的片段（可适当调低 SIMILARITY_THRESHOLD）")

# [5] 大模型回答
step("[5/5] 测试大模型生成回答")
from chat import ask

print(f"问题: {query}")
print("生成回答中（首次加载模型可能需要1~2分钟）...")
result = ask(query)
print(f"\n💬 回答:\n{result['answer']}")

print("\n" + "=" * 50)
print("🎉 自检完成，全流程可用！")
print(f"启动界面: venv\\Scripts\\streamlit run app.py  (或双击 run.bat)")
print("=" * 50)
