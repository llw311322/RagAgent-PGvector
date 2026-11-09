"""文档解析模块：支持 txt / md / docx / xlsx / pdf"""
import os
from typing import List

from langchain_core.documents import Document


def _load_txt(path: str) -> str:
    """读取文本文件，自动尝试多种常见编码"""
    for enc in ("utf-8", "gbk", "gb18030"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, OSError):
            continue
    with open(path, "rb") as f:
        return f.read().decode("utf-8", errors="ignore")


def _load_docx(path: str) -> str:
    """读取 Word 文档：正文段落 + 表格"""
    from docx import Document as DocxDocument
    doc = DocxDocument(path)
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _load_xlsx(path: str) -> str:
    """读取 Excel：所有工作表内容"""
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    parts = []
    for ws in wb.worksheets:
        parts.append(f"# 工作表: {ws.title}")
        for row in ws.iter_rows(values_only=True):
            vals = [str(v).strip() for v in row if v is not None and str(v).strip()]
            if vals:
                parts.append(" | ".join(vals))
    return "\n".join(parts)


def _load_pdf(path: str) -> str:
    """读取 PDF：按页提取文本"""
    from pypdf import PdfReader
    reader = PdfReader(path)
    parts = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            parts.append(f"[第{i + 1}页]\n{text}")
    return "\n".join(parts)


LOADERS = {
    ".txt": _load_txt,
    ".md": _load_txt,
    ".docx": _load_docx,
    ".xlsx": _load_xlsx,
    ".pdf": _load_pdf,
}

SUPPORTED_EXTS = list(LOADERS.keys())


def load_document(path: str) -> List[Document]:
    """解析单个文档，返回 Document 列表（一个文件 = 一个 Document，后续再分块）"""
    ext = os.path.splitext(path)[1].lower()
    if ext not in LOADERS:
        raise ValueError(f"不支持的文件类型: {ext}，当前支持: {', '.join(SUPPORTED_EXTS)}")

    text = LOADERS[ext](path)
    if not text.strip():
        return []

    return [Document(
        page_content=text,
        metadata={"source": os.path.basename(path), "type": ext.lstrip(".")},
    )]
