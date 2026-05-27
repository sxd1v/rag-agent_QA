"""
会话记忆管理：短期记忆（当前会话）+ 长期记忆（跨会话）。

设计：
- 短期记忆：当前 session 的 Q&A 历史，存内存
- 长期记忆：关键问答对向量化存入 Chroma，跨会话可检索
- 面试时可讲"三层记忆架构"：
  Working Memory(ReActState) → Short-Term(session dict) → Long-Term(Chroma)
"""

from langchain_core.documents import Document
from app.core.cache import delete as cache_delete, get as cache_get, set as cache_set

# Redis 可用时 cache 层提供跨 worker 会话；否则这里是单进程回退。
_sessions: dict = {}
_SESSION_TTL_SECONDS = 24 * 60 * 60


def get_history(session_id: str) -> list[dict]:
    """获取会话历史"""
    cached = cache_get(f"session:{session_id}")
    if cached is not None:
        return cached
    return _sessions.get(session_id, [])


def add_turn(session_id: str, question: str, answer: str):
    """添加一轮 Q&A 到短期记忆 + 长期记忆"""
    history = get_history(session_id)
    history.append({
        "question": question,
        "answer": answer,
    })

    # 短期记忆：只保留最近 10 轮
    history = history[-10:]
    _sessions[session_id] = history
    cache_set(f"session:{session_id}", history, ttl=_SESSION_TTL_SECONDS)

    # 长期记忆：向量化存入 Chroma
    _save_to_long_term(question, answer)


def _save_to_long_term(question: str, answer: str):
    """把关键 Q&A 向量化存入 Chroma（长期记忆）"""
    try:
        from app.db.vector_store import get_memory_vector_store

        store = get_memory_vector_store()
        doc = Document(
            page_content=f"用户问：{question}\n系统答：{answer[:500]}",
            metadata={"type": "memory", "source": "conversation"},
        )
        store.add_documents([doc])
    except Exception:
        pass  # Chroma 不可用时静默跳过


def search_memories(query: str, top_k: int = 3) -> list[Document]:
    """
    从长期记忆中检索相关历史问答。

    用于新会话启动时，让 Agent 了解"之前和这个用户聊过什么"。
    """
    try:
        from app.db.vector_store import get_memory_vector_store

        store = get_memory_vector_store()
        results = store.similarity_search(query, k=top_k)
        return results[:top_k]
    except Exception:
        return []


def build_history_summary(history: list[dict], max_turns: int = 5) -> str:
    """
    把历史对话转成 LLM 可读的摘要。

    只取最近 max_turns 轮，避免 prompt 过长。
    """
    if not history:
        return "（这是第一次对话，没有历史记录）"

    recent = history[-max_turns:]
    lines = []
    for i, turn in enumerate(recent, 1):
        lines.append(f"第{i}轮 - 用户问：{turn['question']}")
        lines.append(f"第{i}轮 - 系统答：{turn['answer'][:200]}...")
    return "\n".join(lines)


def clear_session(session_id: str):
    """清除短期记忆（用户主动重置时调用）"""
    cache_delete(f"session:{session_id}")
    _sessions.pop(session_id, None)
