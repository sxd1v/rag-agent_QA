import json
import time
from app.agent.react_state import ReActState
from app.agent.tools import REFUSAL_ANSWER, GenerateAnswerTool, document_source, get_tool_by_name
from app.core.config import AGENT_MAX_LLM_CALLS, AGENT_TIMEOUT_SECONDS
from app.core.llm_client import get_chat_llm, get_llm_call_count, reset_llm_call_count
from app.core.memory import get_history, build_history_summary, add_turn
from app.core.logger import log_agent_trace, log_retry, log_circuit_breaker
from app.services.retriever import search_docs


# ============== LLM 决策相关 ==============

LLM_DECISION_PROMPT_TEMPLATE = """你是一个智能 Agent 决策器。根据当前状态，决定下一步行动。

【用户问题】
{user_question}

【历史对话】
{chat_history}

【当前已收集的证据块】
{context_summary}

【检索历史】
{retrieval_history}

【上一步执行结果】
{last_result}

【剩余检索次数】
{remaining_attempts} 次（最多 {max_attempts} 次）

【可用工具（只能选这三个）】
- search_docs: 检索文档（参数：query, top_k；系统会固定检索策略）
- rewrite_query: 改写后续检索使用的查询（参数：original_query）
- generate_answer: 基于已收集的证据生成最终答案

【你的任务】
查看检索历史中每次返回的 chunk 内容，判断当前状态，选择下一步行动。

判断时请考虑：
- 已有 chunk 的内容是否真正回答了用户问题？不只是数量够不够
- 如果 chunk 内容相关但不够完整，是否需要补充检索？
- 如果 chunk 完全不相关，是否需要改写 query 换个角度搜？
- 如果上一步执行失败（如工具名错误），请修正后重试

输出 JSON 格式的决策：
{{
  "reasoning": "你的分析（说明为什么这样选，引用具体 chunk 内容）",
  "action": "工具名",
  "args": {{"参数": "值"}}
}}
系统会记录召回 chunk 并对最终引用做校验；不要在证据不足时要求直接回答。
"""


def build_retrieval_history_text(retrieval_history: list) -> str:
    """构建检索历史描述"""
    if not retrieval_history:
        return "暂无检索记录"

    lines = []
    for i, entry in enumerate(retrieval_history, 1):
        lines.append(f"第 {i} 次检索:")
        lines.append(f"  Query: {entry['query']}")
        lines.append(f"  返回: {entry['count']} 个 chunk")
        if entry.get("top_chunks"):
            lines.append(f"  内容摘要: {entry['top_chunks'][:200]}...")
    return "\n".join(lines)


def decide_action(state: ReActState, chat_history: str = "") -> tuple[str, dict, str]:
    """
    根据当前 State + 聊天历史，调用 LLM 决定下一步行动。

    返回：(action_name, action_args, reasoning)
    """
    # 构建 context 摘要
    if state.context:
        context_summary = "\n".join([
            f"- [{doc.metadata.get('chunk_id', '?')}] {doc.page_content[:150]}..."
            for doc in state.context[:3]
        ])
    else:
        context_summary = "暂无已收集的证据块"

    # 构建检索历史
    retrieval_history = []
    for entry in state.retrieval_history:
        retrieval_history.append({
            "query": entry["query"],
            "count": entry["count"],
            "top_chunks": " | ".join([doc.page_content[:80] for doc in entry.get("docs", [])[:2]])
        })

    # 计算剩余次数
    remaining = state.max_retrieval_attempts - state.retrieval_attempts

    # 获取上一步执行结果（用于错误反馈）
    if state.history:
        last_observation = state.history[-1].get("observation", "无")
    else:
        last_observation = "（首次执行，无上一步）"

    # 填充 prompt
    prompt = LLM_DECISION_PROMPT_TEMPLATE.format(
        user_question=state.user_question,
        chat_history=chat_history,
        context_summary=context_summary,
        retrieval_history=build_retrieval_history_text(retrieval_history),
        last_result=last_observation,
        remaining_attempts=remaining,
        max_attempts=state.max_retrieval_attempts,
    )

    # 调用 LLM
    llm = get_chat_llm()
    response = llm.invoke(prompt)

    # 解析 LLM 返回
    response_text = response.content if hasattr(response, "content") else str(response)

    try:
        # 尝试提取 JSON（LLM 有时会包裹在 ```json 里）
        response_text = response_text.strip()
        if response_text.startswith("```"):
            # 去掉 markdown 代码块
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1])  # 去掉首尾的 ```
        decision = json.loads(response_text)
    except json.JSONDecodeError:
        # 解析失败，用规则兜底
        print(f"[WARN] LLM 返回格式错误: {response_text}")
        decision = {
            "reasoning": "JSON解析失败，使用规则兜底",
            "action": "generate_answer" if state.context else "search_docs",
            "args": {}
        }

    action_name = decision.get("action", "generate_answer")
    action_args = decision.get("args", {})
    reasoning = decision.get("reasoning", "")

    return action_name, action_args, reasoning


# ============== 原有函数（think 改为描述性） ==============

def think(state: ReActState, reasoning: str = "") -> str:
    """
    思考：根据当前 State，生成 Thought 字符串。

    升级后：reasoning 由 LLM decision 提供，不再独立生成。
    """
    if reasoning:
        return f"【LLM 决策】{reasoning}"

    # fallback（正常不会走到这里）
    if state.retrieval_attempts == 0:
        return f"用户问：{state.user_question}。我需要先从知识库中检索相关内容。"

    remaining = state.max_retrieval_attempts - state.retrieval_attempts
    return f"当前已检索 {state.retrieval_attempts} 次，剩余 {remaining} 次检索机会。"


def execute_tool(action_name: str, action_args: dict, state: ReActState) -> str:
    """
    执行指定的 Tool，带三层防御：
    1. 工具层重试：失败后指数退避重试（最多3次）
    2. 错误追踪：记录连续错误次数，成功时归零
    3. 熔断由 should_finish 统一处理
    """
    import time

    tool = get_tool_by_name(action_name)
    if not tool:
        state.error_count += 1
        state.last_error = f"Unknown tool: {action_name}"
        return f"Error: 未知工具 '{action_name}'"

    tool_args = dict(action_args)
    if action_name == "search_docs":
        requested_top_k = action_args.get("top_k", 5)
        try:
            requested_top_k = int(requested_top_k)
        except (TypeError, ValueError):
            requested_top_k = 5
        tool_args = {
            "query": action_args.get("query", state.query),
            "top_k": min(max(requested_top_k, 5), 8),
            "strategy": state.retrieval_strategy,
            "enable_rerank": state.enable_rerank,
        }
    elif action_name == "rewrite_query":
        tool_args = {"original_query": action_args.get("original_query", state.query)}
    elif action_name == "generate_answer":
        tool_args = {"context": state.context, "question": state.user_question}

    # 工具层重试：最多3次，指数退避
    result = None
    last_exception = None
    for attempt in range(3):
        try:
            result = tool.execute(**tool_args)
            break  # 成功，跳出重试循环
        except Exception as e:
            last_exception = e
            if attempt < 2:  # 还有重试机会
                wait_time = 1 * (2 ** attempt)  # 1s, 2s, 4s
                log_retry(action_name, attempt + 1, str(e))
                time.sleep(wait_time)

    # 全部重试失败
    if result is None:
        state.error_count += 1
        state.last_error = str(last_exception)
        return f"Error: {action_name} 执行失败(已重试3次): {last_exception}"

    # 成功：归零错误计数
    state.error_count = 0
    state.last_error = ""

    # 更新 State
    if action_name == "search_docs":
        state.last_retrieval_result = result["docs"]
        state.retrieval_attempts += 1
        returned_ids = [
            doc.metadata.get("chunk_id", "unknown") for doc in result["docs"]
        ]
        new_ids = [
            chunk_id for chunk_id in returned_ids
            if chunk_id not in state.retrieved_chunk_ids
        ]
        repeated_query = result["query"] in state.searched_queries
        state.searched_queries.add(result["query"])
        state.retrieved_chunk_ids.update(returned_ids)
        if repeated_query or not new_ids:
            state.no_progress_attempts += 1
        else:
            state.no_progress_attempts = 0

        # 记录检索历史（用于 LLM 决策参考）
        state.retrieval_history.append({
            "query": result["query"],
            "count": result["count"],
            "docs": result["docs"],
            "retrieved_chunk_ids": returned_ids,
            "new_chunk_ids": new_ids,
        })

        # 将召回结果纳入证据池；是否足以回答由有引用校验的生成步骤决定。
        if result["docs"]:
            state.context.extend(result["docs"])
            state.context = deduplicate_context(state.context)[:8]

    elif action_name == "rewrite_query":
        state.query = result["rewritten_query"]

    elif action_name == "generate_answer":
        state.final_answer = result["answer"]
        state.final_citations = result.get("citations", [])
        state.abstained = result.get("abstained", False)
        state.done = True

    # 构造 Observation 描述
    observation = format_observation(action_name, result)
    return observation


def format_observation(action_name: str, result: dict) -> str:
    """把 Tool 执行结果格式化为 Observation 字符串"""
    if action_name == "search_docs":
        count = result["count"]
        return f"检索返回了 {count} 个相关 chunk。"

    elif action_name == "rewrite_query":
        return f"query 改写为：'{result['rewritten_query']}'"

    elif action_name == "generate_answer":
        return f"生成了答案，共参考了 {result['source_count']} 个证据块。"

    return str(result)


def deduplicate_context(context: list) -> list:
    """去除重复的 chunk"""
    seen_ids = set()
    unique = []
    for doc in context:
        chunk_id = doc.metadata.get("chunk_id", "")
        if chunk_id and chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            unique.append(doc)
        elif not chunk_id:
            unique.append(doc)
    return unique


def should_finish(state: ReActState) -> bool:
    """判断任务是否应该结束——含三层保护"""
    elapsed = time.perf_counter() - state.started_at if state.started_at else 0.0
    llm_budget_hit = get_llm_call_count() >= max(state.max_llm_calls - 1, 1)
    timeout_hit = elapsed >= state.timeout_seconds

    # 保护1：检索次数耗尽，或某轮没有带来新证据。
    exhausted = state.retrieval_attempts >= state.max_retrieval_attempts
    stalled = state.no_progress_attempts >= 1 and state.retrieval_attempts > 1
    too_many_steps = len(state.history) >= state.max_steps
    if (exhausted or stalled or too_many_steps or llm_budget_hit or timeout_hit) and not state.done:
        if state.context:
            observation = execute_tool("generate_answer", {}, state)
            if timeout_hit:
                reason = f"超时预算触发({state.timeout_seconds:.0f}s)"
            elif llm_budget_hit:
                reason = f"LLM 调用预算触发({state.max_llm_calls})"
            elif stalled:
                reason = "检索无新增证据"
            else:
                reason = "流程约束触发"
            state.history.append({
                "step": len(state.history) + 1,
                "thought": f"【系统收敛】{reason}，仅基于现有证据生成或拒答。",
                "action": "generate_answer",
                "action_args": {},
                "observation": observation,
            })
        else:
            state.final_answer = REFUSAL_ANSWER
            state.abstained = True
            state.done = True

    # 保护2：熔断——连续错误超过阈值
    if state.error_count >= state.max_errors and not state.done:
        state.final_answer = f"系统异常，连续 {state.error_count} 次错误触发熔断。最近错误：{state.last_error}"
        state.failed_reason = f"CircuitBreaker: {state.last_error}"
        state.done = True
        log_circuit_breaker(state.error_count, state.last_error)

    return state.done


def _build_result(state: ReActState) -> dict:
    cited_docs = [
        doc for doc in state.context
        if doc.metadata.get("chunk_id") in state.final_citations
    ]
    return {
        "answer": state.final_answer,
        "history": state.history,
        "retrieval_attempts": state.retrieval_attempts,
        "final_query": state.query,
        "sources": [document_source(doc) for doc in cited_docs],
        "citations": state.final_citations,
        "abstained": state.abstained,
        "context": state.context,
        "llm_calls": get_llm_call_count(),
        "routed_to": state.routed_to,
    }


def _is_simple_question(question: str) -> bool:
    """低成本路由：定义/列举类短问题直接走 Hybrid RAG。"""
    simple_markers = [
        "是什么",
        "什么是",
        "有哪些",
        "列举",
        "英文全称",
        "作用是什么",
        "用途",
    ]
    complex_markers = [
        "比较",
        "区别",
        "为什么",
        "如何",
        "流程",
        "多跳",
        "跨段",
        "综合",
        "优化",
        "组合",
        "因素",
        "分别",
        "策略",
        "场景",
    ]
    return (
        len(question) <= 32
        and any(marker in question for marker in simple_markers)
        and not any(marker in question for marker in complex_markers)
    )


def _run_hybrid_route(question: str, session_id: str, enable_rerank: bool | None) -> dict:
    """简单问题绕过 ReAct 决策和 enhanced 检索，降低 Agent 成本。"""
    docs = search_docs(
        question,
        top_k=5,
        strategy="hybrid",
        enable_rerank=enable_rerank,
    )
    answer = GenerateAnswerTool().execute(question, docs)
    if session_id:
        add_turn(session_id, question, answer["answer"])
    result = {
        "answer": answer["answer"],
        "history": [{
            "step": 1,
            "thought": "【Query Router】简单问题降级到 Hybrid RAG，跳过 ReAct 决策和 enhanced rerank。",
            "action": "route_to_hybrid",
            "action_args": {"strategy": "hybrid"},
            "observation": f"Hybrid RAG 返回 {len(docs)} 个 chunk 并生成答案。",
            "query": question,
            "retrieved_chunk_ids": [doc.metadata.get("chunk_id", "unknown") for doc in docs],
            "new_chunk_ids": [doc.metadata.get("chunk_id", "unknown") for doc in docs],
        }],
        "retrieval_attempts": 1,
        "final_query": question,
        "sources": answer.get("sources", []),
        "citations": answer.get("citations", []),
        "abstained": answer.get("abstained", False),
        "context": docs,
        "llm_calls": get_llm_call_count(),
        "routed_to": "hybrid_rag",
    }
    log_agent_trace(question, result)
    return result


def _resolve_references(question: str, chat_history: str) -> str:
    """
    利用聊天历史消解指代词。
    示例："它的缺点是什么？" + 历史中有"RAG" → "RAG的缺点是什么？"
    """
    if not chat_history or "（这是第一次对话" in chat_history:
        return question

    # 如果 question 中没有明显的指代词，无需改写
    ref_words = ["它", "他", "她", "这个", "那个", "这些", "那些", "其"]
    has_reference = any(w in question for w in ref_words)
    if not has_reference:
        return question

    prompt = (
        "你是一个对话理解专家。把用户问题中的指代词替换为具体内容。\n"
        "规则：只替换指代词本身，不要展开或补充额外信息，保持简洁。\n\n"
        f"【历史对话】\n{chat_history}\n\n"
        f"【当前问题】{question}\n\n"
        "直接输出替换后的简洁问题（不要解释）："
    )

    try:
        llm = get_chat_llm()
        response = llm.invoke(prompt)
        resolved = response.content.strip() if hasattr(response, "content") else str(response).strip()
        if resolved and resolved != question:
            print(f"[Memory] 指代消解: '{question}' → '{resolved}'")
            return resolved
    except Exception:
        pass

    return question


def run_react_loop(
    question: str,
    session_id: str = "",
    retrieval_strategy: str = "enhanced",
    enable_rerank: bool | None = None,
    max_llm_calls: int = AGENT_MAX_LLM_CALLS,
    timeout_seconds: float = AGENT_TIMEOUT_SECONDS,
) -> dict:
    """
    运行 ReAct Loop（带记忆），返回最终结果。

    session_id: 非空时从 Memory 加载历史对话并存入结果，实现多轮记忆。

    返回格式：
    {
        "answer": str,
        "history": list[dict],
        "retrieval_attempts": int,
        "final_query": str,
    }
    """
    # 加载聊天历史
    reset_llm_call_count()
    chat_history = ""
    if session_id:
        history = get_history(session_id)
        chat_history = build_history_summary(history)

    # 有历史对话时，消解指代词（如"它的缺点"→"RAG的缺点"）
    if chat_history and session_id:
        question = _resolve_references(question, chat_history)

    if retrieval_strategy == "enhanced" and _is_simple_question(question):
        return _run_hybrid_route(question, session_id, enable_rerank)

    # 重置 RewriteQueryTool 的状态（避免跨请求污染）
    rewrite_tool = get_tool_by_name("rewrite_query")
    if rewrite_tool:
        rewrite_tool.reset()

    # 初始化 State
    state = ReActState()
    state.reset(
        question,
        retrieval_strategy=retrieval_strategy,
        enable_rerank=enable_rerank,
        max_llm_calls=max_llm_calls,
        timeout_seconds=timeout_seconds,
    )

    # 第一次检索：固定用 search_docs（首次必须先有检索结果，LLM 才能决策）
    first_action_name = "search_docs"
    first_action_args = {"query": state.query, "top_k": 5}
    first_observation = execute_tool(first_action_name, first_action_args, state)

    state.history.append({
        "step": 1,
        "thought": "【首次检索】用户问题：" + question,
        "action": first_action_name,
        "action_args": first_action_args,
        "observation": first_observation,
        "query": state.query,
        "retrieved_chunk_ids": state.retrieval_history[-1]["retrieved_chunk_ids"] if state.retrieval_history else [],
        "new_chunk_ids": state.retrieval_history[-1]["new_chunk_ids"] if state.retrieval_history else [],
    })

    # 检查是否直接结束（首次检索无结果）
    if should_finish(state):
        if session_id:
            add_turn(session_id, question, state.final_answer)
        result = _build_result(state)
        log_agent_trace(question, result)
        return result

    # 主循环：LLM 决策后续行动
    while not state.done:
        step_num = len(state.history) + 1

        # 1. LLM 决策下一步
        action_name, action_args, reasoning = decide_action(state, chat_history)

        # 2. 生成 Thought（基于 LLM reasoning）
        thought = think(state, reasoning)

        # 3. Execute
        observation = execute_tool(action_name, action_args, state)

        # 4. 记录到 history
        state.history.append({
            "step": step_num,
            "thought": thought,
            "action": action_name,
            "action_args": action_args,
            "observation": observation,
            "query": state.retrieval_history[-1]["query"] if action_name == "search_docs" else None,
            "retrieved_chunk_ids": (
                state.retrieval_history[-1]["retrieved_chunk_ids"]
                if action_name == "search_docs" else []
            ),
            "new_chunk_ids": (
                state.retrieval_history[-1]["new_chunk_ids"]
                if action_name == "search_docs" else []
            ),
        })

        # 5. 检查是否完成
        if should_finish(state):
            break

    # 存入会话记忆
    if session_id:
        add_turn(session_id, question, state.final_answer)

    result = _build_result(state)
    log_agent_trace(question, result)
    return result
