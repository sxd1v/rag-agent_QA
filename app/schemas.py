from typing import Literal, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """问答请求"""
    question: str
    top_k: int = Field(default=5, ge=1, le=20)
    session_id: str = ""  # 非空时启用多轮记忆
    retrieval_strategy: Optional[Literal["vector", "hybrid", "enhanced"]] = None


class SourceItem(BaseModel):
    """证据来源"""
    chunk_id: str
    source: Optional[str] = None
    content: str


class AskResponse(BaseModel):
    """普通 RAG 问答响应"""
    answer: str
    sources: list[SourceItem] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    abstained: bool = False


class RetrieveDebugResponse(BaseModel):
    """仅检索调试响应"""
    docs: list[SourceItem] = Field(default_factory=list)


class AgentHistoryItem(BaseModel):
    """ReAct Loop 单轮记录"""
    step: int
    thought: str
    action: str
    action_args: dict
    observation: str
    query: Optional[str] = None
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    new_chunk_ids: list[str] = Field(default_factory=list)


class AgentAskResponse(BaseModel):
    """Agent 模式问答响应"""
    answer: str
    retrieval_attempts: int
    final_query: str
    history: list[AgentHistoryItem]
    sources: list[SourceItem] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    abstained: bool = False
    trace_metrics: dict = Field(default_factory=dict)


class EvalRequest(BaseModel):
    """评估请求"""
    question: str
    expected_answerable: Optional[bool] = None
    ground_truth_chunk_ids: list[str] = Field(default_factory=list)


class EvalResponse(BaseModel):
    """RAGAs 评估响应"""
    question: str
    answer: str
    retrieval_attempts: int
    scores: dict
    agent_metrics: dict = Field(default_factory=dict)
