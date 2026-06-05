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

# SiliconFlow API 配置（旧字段保留，作为默认 Chat API key）
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")

# OpenAI-compatible Chat API 配置
CHAT_PROVIDER = os.getenv("CHAT_PROVIDER", "openai-compatible")
CHAT_API_KEY = os.getenv("CHAT_API_KEY") or SILICONFLOW_API_KEY
CHAT_BASE_URL = os.getenv("CHAT_BASE_URL", "https://api.siliconflow.cn/v1")
CHAT_MODEL = os.getenv("CHAT_MODEL", "deepseek-ai/DeepSeek-V3.2")
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.7"))

# Embedding 提供者：minimax / openai-compatible / hash
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "minimax")
_EMBEDDING_PROVIDER_NORMALIZED = EMBEDDING_PROVIDER.lower()
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY") or (
    SILICONFLOW_API_KEY
    if _EMBEDDING_PROVIDER_NORMALIZED in {"openai", "openai-compatible"}
    else MINIMAX_API_KEY
)
EMBEDDING_BASE_URL = os.getenv(
    "EMBEDDING_BASE_URL",
    "https://api.siliconflow.cn/v1"
    if _EMBEDDING_PROVIDER_NORMALIZED in {"openai", "openai-compatible"}
    else MINIMAX_BASE_URL,
)

# Embedding 模型
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embo-01")
HASH_EMBEDDING_DIM = int(os.getenv("HASH_EMBEDDING_DIM", "384"))

# Chroma 向量库持久化路径（默认项目内 data/chroma_db）
CHROMA_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "chroma_db")
)
KNOWLEDGE_COLLECTION_NAME = os.getenv("KNOWLEDGE_COLLECTION_NAME", "knowledge")
MEMORY_COLLECTION_NAME = os.getenv("MEMORY_COLLECTION_NAME", "memory")

# 默认 top_k
DEFAULT_TOP_K = 5

# Agent 成本与延迟预算
AGENT_MAX_LLM_CALLS = int(os.getenv("AGENT_MAX_LLM_CALLS", "6"))
AGENT_TIMEOUT_SECONDS = float(os.getenv("AGENT_TIMEOUT_SECONDS", "120"))

# 高级检索开关
ENABLE_RERANK = os.getenv("ENABLE_RERANK", "true").lower() in {"1", "true", "yes", "on"}

# chunk 默认大小
CHUNK_SIZE = 500

# chunk 重叠
CHUNK_OVERLAP = 50
