from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.db.vector_store import get_vector_store


def split_documents(
    docs: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Document]:
    """
    将原始 Document 切分成更适合检索的小块。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_docs = splitter.split_documents(docs)
    return split_docs


def add_chunk_ids(docs: List[Document]) -> List[Document]:
    """
    给切分后的文档块补充 chunk_id，方便追踪和返回 sources。
    """
    for i, doc in enumerate(docs):
        if "chunk_id" not in doc.metadata:
            doc.metadata["chunk_id"] = f"chunk-{i}"
    return docs


def index_documents(docs: List[Document]) -> int:
    """
    将文档切分后写入 Chroma，返回写入的 chunk 数量。
    """
    vector_store = get_vector_store()

    split_docs = split_documents(docs)
    split_docs = add_chunk_ids(split_docs)

    vector_store.add_documents(split_docs)
    return len(split_docs)