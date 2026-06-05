from threading import Lock

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from app.db.hash_embeddings import HashEmbeddings
from app.db.minimax_embeddings import MiniMaxEmbeddings
from app.core.config import (
    CHROMA_PERSIST_DIR,
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    HASH_EMBEDDING_DIM,
    KNOWLEDGE_COLLECTION_NAME,
    MEMORY_COLLECTION_NAME,
    MINIMAX_API_KEY,
    MINIMAX_BASE_URL,
)

_vector_stores: dict[str, Chroma] = {}
_store_lock = Lock()


def build_embeddings():
    """Build the configured embedding function."""
    provider = EMBEDDING_PROVIDER.lower()
    if provider == "minimax":
        return MiniMaxEmbeddings(
            api_key=EMBEDDING_API_KEY or MINIMAX_API_KEY,
            base_url=EMBEDDING_BASE_URL or MINIMAX_BASE_URL,
            model=EMBEDDING_MODEL,
        )
    if provider in {"openai", "openai-compatible"}:
        if not EMBEDDING_API_KEY:
            raise ValueError("EMBEDDING_API_KEY is required for openai-compatible embeddings")
        kwargs = {
            "api_key": EMBEDDING_API_KEY,
            "model": EMBEDDING_MODEL,
        }
        if EMBEDDING_BASE_URL:
            kwargs["base_url"] = EMBEDDING_BASE_URL
        return OpenAIEmbeddings(**kwargs)
    if provider == "hash":
        return HashEmbeddings(dimension=HASH_EMBEDDING_DIM)
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")


def get_vector_store(collection_name: str = KNOWLEDGE_COLLECTION_NAME):
    """Return the knowledge store by default; memory uses a separate collection."""
    with _store_lock:
        if collection_name not in _vector_stores:
            embeddings = build_embeddings()
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
