from threading import Lock

from langchain_chroma import Chroma
from app.db.minimax_embeddings import MiniMaxEmbeddings
from app.core.config import (
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    KNOWLEDGE_COLLECTION_NAME,
    MEMORY_COLLECTION_NAME,
    MINIMAX_API_KEY,
    MINIMAX_BASE_URL,
)

_vector_stores: dict[str, Chroma] = {}
_store_lock = Lock()


def get_vector_store(collection_name: str = KNOWLEDGE_COLLECTION_NAME):
    """Return the knowledge store by default; memory uses a separate collection."""
    with _store_lock:
        if collection_name not in _vector_stores:
            embeddings = MiniMaxEmbeddings(
                api_key=MINIMAX_API_KEY,
                base_url=MINIMAX_BASE_URL,
                model=EMBEDDING_MODEL,
            )
            _vector_stores[collection_name] = Chroma(
                collection_name=collection_name,
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=embeddings,
            )
        return _vector_stores[collection_name]


def get_memory_vector_store():
    return get_vector_store(collection_name=MEMORY_COLLECTION_NAME)


def clear_vector_store_cache():
    """Testing hook and process-local reset after collection reconfiguration."""
    with _store_lock:
        _vector_stores.clear()
