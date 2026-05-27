from app.schemas import AskResponse, RetrieveDebugResponse, SourceItem
from app.services.retriever import search_docs
from app.agent.tools import GenerateAnswerTool

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


def answer_question(
    question: str,
    top_k: int = 3,
    retrieval_strategy: str = "hybrid",
) -> AskResponse:
    """普通 RAG 模式：检索 → 构建上下文 → LLM 生成答案"""
    docs = search_docs(question, top_k=top_k, strategy=retrieval_strategy)
    result = GenerateAnswerTool().execute(question, docs)

    return AskResponse(
        answer=result["answer"],
        sources=[SourceItem(**source) for source in result.get("sources", [])],
        citations=result.get("citations", []),
        abstained=result.get("abstained", False),
    )


def retrieve_only(question: str, top_k: int = 3, retrieval_strategy: str = "hybrid") -> RetrieveDebugResponse:
    docs = search_docs(question, top_k=top_k, strategy=retrieval_strategy)
    return RetrieveDebugResponse(
        docs=format_sources(docs)
    )
