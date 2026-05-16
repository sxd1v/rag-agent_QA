from pydantic import BaseModel
from typing import Optional


class AskRequest(BaseModel):
    """问答请求"""
    question: str
    top_k: int = 5
    session_id: str = ""  # 非空时启用多轮记忆


class SourceItem(BaseModel):
    """证据来源"""
    chunk_id: str
    source: Optional[str] = None
    content: str


class AskResponse(BaseModel):
    """普通 RAG 问答响应"""
    answer: str
    sources: list[SourceItem] = []


class RetrieveDebugResponse(BaseModel):
    """仅检索调试响应"""
    docs: list[SourceItem] = []


class AgentHistoryItem(BaseModel):
    """ReAct Loop 单轮记录"""
    step: int
    thought: str
    action: str
    action_args: dict
    observation: str


class AgentAskResponse(BaseModel):
    """Agent 模式问答响应"""
    answer: str
    retrieval_attempts: int
    final_query: str
    history: list[AgentHistoryItem]


class EvalRequest(BaseModel):
    """评估请求"""
    question: str


class EvalResponse(BaseModel):
    """RAGAs 评估响应"""
    question: str
    answer: str
    retrieval_attempts: int
    scores: dict
