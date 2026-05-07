from fastapi import APIRouter
from app.schemas import AskRequest, AskResponse, AgentAskResponse, AgentHistoryItem, EvalRequest, EvalResponse
from app.services.qa_service import answer_question
from app.agent.react_loop import run_react_loop
from app.services.ragas_eval import evaluate

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """普通 RAG 模式（固定流程，非 Agent）"""
    result = answer_question(req.question, req.top_k)
    return result


@router.post("/agent_ask", response_model=AgentAskResponse)
def agent_ask(req: AskRequest):
    """ReAct Agent 模式"""
    result = run_react_loop(req.question)
    return AgentAskResponse(
        answer=result["answer"],
        retrieval_attempts=result["retrieval_attempts"],
        final_query=result["final_query"],
        history=[AgentHistoryItem(**h) for h in result["history"]],
    )


@router.post("/evaluate", response_model=EvalResponse)
def eval_endpoint(req: EvalRequest):
    """RAGAs 四指标评估：先跑 Agent，再评估检索和答案质量"""
    result = run_react_loop(req.question)
    # 用 search_docs 获取检索 context（Agent 内部用的是混合检索）
    from app.services.retriever import search_docs
    docs = search_docs(req.question, top_k=5)

    scores = evaluate(
        question=req.question,
        answer=result["answer"],
        retrieved_docs=docs,
        ground_truth_docs=None,
    )
    return EvalResponse(
        question=req.question,
        answer=result["answer"],
        retrieval_attempts=result["retrieval_attempts"],
        scores=scores,
    )
