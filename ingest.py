"""知识库导入脚本。用法：python ingest.py <文件路径>"""
import sys
from app.services.indexer import index_documents

# 支持 PDF 和 Markdown
file_path = sys.argv[1] if len(sys.argv) > 1 else "data/test_knowledge.md"

if file_path.endswith(".pdf"):
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(file_path)
else:
    from langchain_community.document_loaders import TextLoader
    loader = TextLoader(file_path, encoding="utf-8")

docs = loader.load()
print(f"Loaded {len(docs)} pages/sections")
count = index_documents(docs)
print(f"Indexed {count} chunks")
