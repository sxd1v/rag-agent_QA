import json
from app.agent.react_state import ReActState
from app.agent.tools import get_tool_by_name
from app.core.llm_client import get_chat_llm


# ============== LLM 决策相关 ==============

LLM_DECISION_PROMPT_TEMPLATE = """你是一个智能 Agent 决策器。根据当前状态，决定下一步行动。

【用户问题】
{user_question}

【当前已收集的证据块】
{context_summary}

【检索历史】
{retrieval_history}

【上一步执行结果】
{last_result}

【剩余检索次数】
{remaining_attempts} 次（最多 {max_attempts} 次）

【可用工具（只能选这三个）】
- search_docs: 检索文档（参数：query, top_k）
- rewrite_query: 改写查询后自动重新检索（参数：original_query）
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
如果 action 是 search_docs 且检索后有可用的 chunk，可以在决策中加入 "keep_chunks": ["chunk_id_1", "chunk_id_2"] 来指定哪些 chunk 值得保留到 context。
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


def decide_action(state: ReActState) -> tuple[str, dict, str]:
    """
    根据当前 State，调用 LLM 决定下一步行动。

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
    执行指定的 Tool，返回 Observation 字符串。
    """
    tool = get_tool_by_name(action_name)
    if not tool:
        return f"Error: Unknown tool '{action_name}'"

    if action_name == "generate_answer":
        action_args["context"] = state.context
        action_args["question"] = state.user_question

    result = tool.execute(**action_args)

    # 更新 State
    if action_name == "search_docs":
        state.last_retrieval_result = result["docs"]
        state.retrieval_attempts += 1

        # 记录检索历史（用于 LLM 决策参考）
        state.retrieval_history.append({
            "query": result["query"],
            "count": result["count"],
            "docs": result["docs"],
        })

        # 按 LLM 决策添加 context（而非全部自动添加）
        if result["docs"]:
            if action_args.get("keep_chunks"):
                # LLM 指定了要保留的 chunk
                kept = [d for d in result["docs"] if d.metadata.get("chunk_id") in action_args["keep_chunks"]]
                state.context.extend(kept)
            # 如果 LLM 没指定 keep_chunks，不自动添加（等 LLM 下一轮决定）
            state.context = deduplicate_context(state.context)

    elif action_name == "rewrite_query":
        state.query = result["rewritten_query"]

    elif action_name == "generate_answer":
        state.final_answer = result["answer"]
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
    """判断任务是否应该结束"""
    # 检索次数耗尽，强制结束
    if state.retrieval_attempts >= state.max_retrieval_attempts and not state.done:
        # 有 context 就强行生成，没有就坦诚无法回答
        if state.context:
            state.final_answer = "（系统兜底：检索次数已用完，使用已有证据生成答案）"
        else:
            state.final_answer = "抱歉，经过多次检索仍未找到足够的证据来回答这个问题。"
        state.done = True
    return state.done


def run_react_loop(question: str) -> dict:
    """
    运行 ReAct Loop，返回最终结果。

    返回格式：
    {
        "answer": str,
        "history": list[dict],   # 每轮的 thought/action/observation
        "retrieval_attempts": int,
        "final_query": str,
    }
    """
    # 重置 RewriteQueryTool 的状态（避免跨请求污染）
    rewrite_tool = get_tool_by_name("rewrite_query")
    if rewrite_tool:
        rewrite_tool.reset()

    # 初始化 State
    state = ReActState()
    state.reset(question)

    # 第一次检索：固定用 search_docs（首次必须先有检索结果，LLM 才能决策）
    first_action_name = "search_docs"
    first_action_args = {"query": state.query, "top_k": 3}
    first_observation = execute_tool(first_action_name, first_action_args, state)

    # 首次检索启动特权：全部加入 context（后续检索由 LLM 通过 keep_chunks 筛选）
    if state.last_retrieval_result:
        state.context.extend(state.last_retrieval_result)
        state.context = deduplicate_context(state.context)

    state.history.append({
        "step": 1,
        "thought": "【首次检索】用户问题：" + question,
        "action": first_action_name,
        "action_args": first_action_args,
        "observation": first_observation,
    })

    # 检查是否直接结束（首次检索无结果）
    if should_finish(state):
        return {
            "answer": state.final_answer,
            "history": state.history,
            "retrieval_attempts": state.retrieval_attempts,
            "final_query": state.query,
        }

    # 主循环：LLM 决策后续行动
    while not state.done:
        step_num = len(state.history) + 1

        # 1. LLM 决策下一步
        action_name, action_args, reasoning = decide_action(state)

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
        })

        # 5. 检查是否完成
        if should_finish(state):
            break

    return {
        "answer": state.final_answer,
        "history": state.history,
        "retrieval_attempts": state.retrieval_attempts,
        "final_query": state.query,
    }
