from app.schemas import AskResponse, RetrieveDebugResponse, SourceItem
from app.services.retriever import search_docs
from app.core.llm_client import get_chat_llm


def build_context(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def format_sources(docs):
    sources = []
    for doc in docs:
        sources.append(
            SourceItem(
                chunk_id=doc.metadata.get("chunk_id", "unknown"),
                source=doc.metadata.get("source"),
                content=doc.page_content,
            )
        )
    return sources


def answer_question(question: str, top_k: int = 3) -> AskResponse:
    """普通 RAG 模式：检索 → 构建上下文 → LLM 生成答案"""
    docs = search_docs(question, top_k=top_k)
    context = build_context(docs)

    if not docs:
        answer = "抱歉，知识库中没有找到相关的内容来回答这个问题。"
    else:
        prompt = (
            f"你是一个专业的问答助手。请根据以下参考资料回答用户问题。\n\n"
            f"【参考资料】\n{context}\n\n"
            f"【用户问题】{question}\n\n"
            f"请结合参考资料给出准确、完整的回答。如果资料不足以回答，请如实说明。"
        )
        llm = get_chat_llm()
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)

    return AskResponse(
        answer=answer,
        sources=format_sources(docs),
    )


def retrieve_only(question: str, top_k: int = 3) -> RetrieveDebugResponse:
    docs = search_docs(question, top_k=top_k)
    return RetrieveDebugResponse(
        docs=format_sources(docs)
    )