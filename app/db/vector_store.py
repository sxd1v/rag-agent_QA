from langchain_chroma import Chroma
from app.db.minimax_embeddings import MiniMaxEmbeddings
from app.core.config import EMBEDDING_MODEL, CHROMA_PERSIST_DIR, MINIMAX_API_KEY, MINIMAX_BASE_URL


def get_vector_store():
    embeddings = MiniMaxEmbeddings(
        api_key=MINIMAX_API_KEY,
        base_url=MINIMAX_BASE_URL,
        model=EMBEDDING_MODEL,
    )

    vector_store = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
    )
    return vector_store
