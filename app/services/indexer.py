import hashlib
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
    for doc in docs:
        if "chunk_id" not in doc.metadata:
            identity = "|".join([
                str(doc.metadata.get("source", "")),
                str(doc.metadata.get("page", "")),
                doc.page_content,
            ])
            digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
            doc.metadata["chunk_id"] = f"chunk-{digest}"
        doc.metadata.setdefault("type", "knowledge")
    return docs


def index_documents(docs: List[Document]) -> int:
    """
    将文档切分后写入 Chroma，返回写入的 chunk 数量。
    """
    vector_store = get_vector_store()

    split_docs = split_documents(docs)
    split_docs = add_chunk_ids(split_docs)

    vector_store.add_documents(
        split_docs,
        ids=[doc.metadata["chunk_id"] for doc in split_docs],
    )
    # BM25 持久化索引和检索缓存必须与新写入的向量库同步。
    from app.services.hybrid_retriever import rebuild_bm25_index
    from app.services.retriever import clear_retrieval_cache

    rebuild_bm25_index()
    clear_retrieval_cache()
    return len(split_docs)
