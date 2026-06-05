from fastapi import APIRouter
from app.schemas import (
    AgentAskResponse,
    AgentHistoryItem,
    AskRequest,
    AskResponse,
    EvalRequest,
    EvalResponse,
    RetrieveDebugResponse,
)
from starlette.concurrency import run_in_threadpool

from app.core.config import AGENT_MAX_LLM_CALLS, AGENT_TIMEOUT_SECONDS
from app.services.qa_service import answer_question, retrieve_only
from app.agent.react_loop import run_react_loop
from app.agent.multi_agent import orchestrate
from app.services.ragas_eval import evaluate, evaluate_agent_behavior

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """普通 RAG 模式（固定流程，非 Agent）"""
    strategy = req.retrieval_strategy or "hybrid"
    result = await run_in_threadpool(
        answer_question,
        req.question,
        req.top_k,
        strategy,
        req.enable_rerank,
    )
    return result


@router.post("/agent_ask", response_model=AgentAskResponse)
async def agent_ask(req: AskRequest):
    """ReAct Agent 模式（支持多轮记忆，传入 session_id 即可）"""
    strategy = req.retrieval_strategy or "enhanced"
    result = await run_in_threadpool(
        run_react_loop,
        req.question,
        req.session_id,
        strategy,
        req.enable_rerank,
        req.max_llm_calls or AGENT_MAX_LLM_CALLS,
        req.timeout_seconds or AGENT_TIMEOUT_SECONDS,
    )
    return AgentAskResponse(
        answer=result["answer"],
        retrieval_attempts=result["retrieval_attempts"],
        final_query=result["final_query"],
        history=[AgentHistoryItem(**h) for h in result["history"]],
        sources=result["sources"],
        citations=result["citations"],
        abstained=result["abstained"],
        trace_metrics=evaluate_agent_behavior(result),
    )


@router.post("/multi_agent")
async def multi_agent_ask(req: AskRequest):
    """Multi-Agent 模式：Researcher + Writer 协作"""
    result = await run_in_threadpool(orchestrate, req.question)
    return {
        "answer": result["answer"],
        "rounds": result["rounds"],
        "confident": result["confident"],
        "citations": result["citations"],
        "sources": result["sources"],
        "abstained": result["abstained"],
    }


@router.post("/retrieve_debug", response_model=RetrieveDebugResponse)
async def retrieve_debug_endpoint(req: AskRequest):
    strategy = req.retrieval_strategy or "hybrid"
    return await run_in_threadpool(
        retrieve_only,
        req.question,
        req.top_k,
        strategy,
        req.enable_rerank,
    )


@router.post("/evaluate", response_model=EvalResponse)
async def eval_endpoint(req: EvalRequest):
    """RAGAs 四指标评估：先跑 Agent，再评估检索和答案质量"""
    result = await run_in_threadpool(run_react_loop, req.question)
    docs = result["context"]

    scores = await run_in_threadpool(evaluate, req.question, result["answer"], docs, None)
    if req.ground_truth_chunk_ids:
        retrieved_ids = {doc.metadata.get("chunk_id") for doc in docs}
        scores["context_recall"] = round(
            len(retrieved_ids & set(req.ground_truth_chunk_ids)) / len(req.ground_truth_chunk_ids),
            2,
        )
    return EvalResponse(
        question=req.question,
        answer=result["answer"],
        retrieval_attempts=result["retrieval_attempts"],
        scores=scores,
        agent_metrics=evaluate_agent_behavior(result, req.expected_answerable),
    )
