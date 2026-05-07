import os
from pathlib import Path
from dotenv import load_dotenv

# 显式指定 .env 路径（项目根目录 rag_api/）
ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(ENV_PATH)

# MiniMax API 配置（Embedding 用）
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
MINIMAX_BASE_URL = "https://api.minimax.chat/v1"

# Gemini API 配置
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# SiliconFlow API 配置
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")

# Embedding 提供者
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "minimax")

# Embedding 模型
EMBEDDING_MODEL = "embo-01"

# Chat 模型
CHAT_MODEL = "abab5-chat"

# Chroma 向量库持久化路径（默认项目内 data/chroma_db）
CHROMA_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "chroma_db")
)

# 默认 top_k
DEFAULT_TOP_K = 5

# chunk 默认大小
CHUNK_SIZE = 500

# chunk 重叠
CHUNK_OVERLAP = 50
