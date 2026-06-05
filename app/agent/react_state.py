from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ReActState:
    """
    ReAct Agent 的运行时状态。

    每次循环结束后，State 都会被更新，
    LLM 根据当前 State 决定下一步做什么。
    """

    # 任务目标
    goal: str = "有证据地回答用户问题"

    # 用户原始问题
    user_question: str = ""

    # 当前正在检索的 query（可被 rewrite_query 改写）
    query: str = ""

    # 已检索次数
    retrieval_attempts: int = 0
    max_retrieval_attempts: int = 3

    # 检索历史（每次检索的 query 和结果，用于 LLM 决策）
    retrieval_history: list = field(default_factory=list)

    # 上一次检索返回的 chunks
    last_retrieval_result: list = field(default_factory=list)

    # 已确认使用的证据块（会被 build_context 使用）
    context: list = field(default_factory=list)

    # 最终答案
    final_answer: Optional[str] = None
    final_citations: list[str] = field(default_factory=list)
    abstained: bool = False

    # 检索流程约束与可观测字段
    retrieval_strategy: str = "enhanced"
    enable_rerank: bool | None = None
    retrieved_chunk_ids: set[str] = field(default_factory=set)
    searched_queries: set[str] = field(default_factory=set)
    no_progress_attempts: int = 0
    max_steps: int = 8
    max_llm_calls: int = 6
    timeout_seconds: float = 120.0
    started_at: float = 0.0
    routed_to: Optional[str] = None

    # 任务是否完成
    done: bool = False

    # 如果失败，原因是什么
    failed_reason: Optional[str] = None

    # 异常处理：错误计数 + 熔断
    error_count: int = 0       # 连续错误次数
    max_errors: int = 3        # 连续几次后触发熔断
    last_error: str = ""       # 最近一次错误信息

    # 思考-行动-观察 的完整历史
    history: list = field(default_factory=list)

    def reset(
        self,
        question: str,
        retrieval_strategy: str = "enhanced",
        enable_rerank: bool | None = None,
        max_llm_calls: int = 6,
        timeout_seconds: float = 120.0,
    ):
        """初始化/重置 State"""
        import time

        self.user_question = question
        self.query = question
        self.retrieval_attempts = 0
        self.retrieval_history = []
        self.last_retrieval_result = []
        self.context = []
        self.final_answer = None
        self.final_citations = []
        self.abstained = False
        self.retrieval_strategy = retrieval_strategy
        self.enable_rerank = enable_rerank
        self.retrieved_chunk_ids = set()
        self.searched_queries = set()
        self.no_progress_attempts = 0
        self.max_llm_calls = max_llm_calls
        self.timeout_seconds = timeout_seconds
        self.started_at = time.perf_counter()
        self.routed_to = None
        self.done = False
        self.failed_reason = None
        self.error_count = 0
        self.last_error = ""
        self.history = []
