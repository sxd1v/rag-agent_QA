"""
RAGAs 四指标评估模块（轻量实现，LLM-as-Judge）

指标：
- Faithfulness：答案是否忠于检索到的文档
- Answer Relevancy：答案是否回答原始问题
- Context Precision：召回文档中有多少是真正有用的
- Context Recall：需要的文档都召回了没有（需 ground truth）
"""

import json

from app.core.llm_client import get_chat_llm


def faithfulness_score(question: str, answer: str, context_text: str) -> float:
    """
    评估答案是否忠于检索文档（0-1）。

    LLM 逐条检查答案中的陈述是否可以从文档中推导出来。
    """
    if not answer or not context_text:
        return 0.0

    prompt = (
        "你是一个评估专家。请判断以下答案是否忠于提供的文档内容。\n\n"
        f"【文档内容】\n{context_text}\n\n"
        f"【需要评估的答案】\n{answer}\n\n"
        "请严格判断：答案中的每句话是否能从文档中推导出来？\n"
        "输出 0 到 1 之间的分数（1=完全忠于文档，0=有严重幻觉/胡编）：\n"
        "分数："
    )

    llm = get_chat_llm()
    response = llm.invoke(prompt)
    text = response.content.strip() if hasattr(response, "content") else str(response).strip()

    try:
        return float(text)
    except ValueError:
        # 如果 LLM 返回的不是纯数字，给一个保守估计
        return 0.5


def answer_relevancy_score(question: str, answer: str) -> float:
    """
    评估答案是否回答原始问题（0-1）。
    """
    if not answer or not question:
        return 0.0

    prompt = (
        "你是一个评估专家。请判断以下答案是否真正回答了用户的问题。\n\n"
        f"【用户问题】\n{question}\n\n"
        f"【生成的答案】\n{answer}\n\n"
        "请判断：答案是否直接、完整地回答了问题？有没有答非所问？\n"
        "输出 0 到 1 之间的分数（1=完美回答，0=完全答非所问）：\n"
        "分数："
    )

    llm = get_chat_llm()
    response = llm.invoke(prompt)
    text = response.content.strip() if hasattr(response, "content") else str(response).strip()

    try:
        return float(text)
    except ValueError:
        return 0.5


def context_precision_score(question: str, docs: list) -> float:
    """
    评估召回的文档有多少是真正相关的（0-1）。

    对每个文档，用 LLM 判断是否与问题相关，取比例。
    """
    if not docs:
        return 0.0

    relevant_count = 0
    llm = get_chat_llm()

    for doc in docs:
        content = doc.page_content[:300] if hasattr(doc, "page_content") else str(doc)[:300]
        prompt = (
            '判断以下文档内容是否与用户问题相关。只回答"是"或"否"。\n\n'
            f"【用户问题】{question}\n"
            f"【文档内容】{content}\n\n"
            "相关吗？（是/否）："
        )
        response = llm.invoke(prompt)
        text = response.content.strip() if hasattr(response, "content") else str(response).strip()
        if "是" in text:
            relevant_count += 1

    return relevant_count / len(docs)


def context_recall_score(question: str, docs: list, ground_truth_docs: list = None) -> float:
    """
    评估需要的文档是否都被召回了（0-1）。

    需要 ground_truth_docs 作为参照。未提供则返回 -1 表示无法计算。
    """
    if ground_truth_docs is None:
        return -1.0  # 无法计算

    if not ground_truth_docs:
        return 1.0  # 没有需要的文档 = 完美

    retrieved_ids = set(doc.metadata.get("chunk_id", "") for doc in docs)
    ground_truth_ids = set(doc.metadata.get("chunk_id", "") for doc in ground_truth_docs)

    if not ground_truth_ids:
        return 1.0

    recalled = len(retrieved_ids & ground_truth_ids)
    return recalled / len(ground_truth_ids)


def evaluate(question: str, answer: str, retrieved_docs: list, ground_truth_docs: list = None) -> dict:
    """
    一站式评估，单次 judge 调用返回三个生成质量指标，降低批量实验成本。

    返回格式：
    {
        "faithfulness": float,
        "answer_relevancy": float,
        "context_precision": float,
        "context_recall": float or "N/A",
    }
    """
    context_text = "\n\n".join([
        f"[{doc.metadata.get('chunk_id', 'unknown')}] {doc.page_content}"
        for doc in retrieved_docs
    ]) if retrieved_docs else ""

    if not answer or not context_text:
        quality_scores = {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
        }
    else:
        prompt = (
            "你是严格的 RAG 评估器。根据问题、答案和召回证据，一次性给出三个 0 到 1 的分数。\n"
            "- faithfulness: 答案陈述是否都能被证据支持。\n"
            "- answer_relevancy: 答案是否直接回答问题；证据不足时的明确拒答也可以相关。\n"
            "- context_precision: 召回证据中与问题直接相关的比例。\n"
            "只输出 JSON，不要解释，例如："
            '{"faithfulness": 0.8, "answer_relevancy": 0.9, "context_precision": 0.6}\n\n'
            f"【问题】\n{question}\n\n【答案】\n{answer}\n\n【召回证据】\n{context_text}"
        )
        llm = get_chat_llm()
        response = llm.invoke(prompt)
        text = response.content.strip() if hasattr(response, "content") else str(response).strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:-1])
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = {}
        quality_scores = {}
        for name in ("faithfulness", "answer_relevancy", "context_precision"):
            try:
                quality_scores[name] = round(max(0.0, min(1.0, float(payload[name]))), 2)
            except (KeyError, TypeError, ValueError):
                quality_scores[name] = 0.0

    return {
        **quality_scores,
        "context_recall": (
            round(context_recall_score(question, retrieved_docs, ground_truth_docs), 2)
            if ground_truth_docs else "N/A（需要 ground truth）"
        ),
    }


def evaluate_agent_behavior(
    result: dict,
    expected_answerable: bool | None = None,
) -> dict:
    """对可确定的 Agent 行为做非 LLM 评估。"""
    history = result.get("history", [])
    search_steps = [step for step in history if step.get("action") == "search_docs"]
    queries = [step.get("query") for step in search_steps if step.get("query")]
    retrieved_ids = {
        chunk_id
        for step in search_steps
        for chunk_id in step.get("retrieved_chunk_ids", [])
    }
    citations = result.get("citations", [])
    actions = [step.get("action") for step in history]
    valid_actions = {"search_docs", "rewrite_query", "generate_answer", "route_to_hybrid"}
    routed_to_hybrid = result.get("routed_to") == "hybrid_rag"
    metrics = {
        "action_count": len(history),
        "search_count": len(search_steps),
        "duplicate_query_count": len(queries) - len(set(queries)),
        "no_progress_search_count": sum(
            1 for step in search_steps if not step.get("new_chunk_ids", [])
        ),
        "citations_valid": set(citations).issubset(retrieved_ids),
        "citation_count": len(citations),
        "abstained": result.get("abstained", False),
        "llm_calls": result.get("llm_calls", 0),
        "routed_to": result.get("routed_to"),
        "valid_action_sequence": (
            bool(actions)
            and all(action in valid_actions for action in actions)
            and (
                (routed_to_hybrid and actions[0] == "route_to_hybrid")
                or (
                    actions[0] == "search_docs"
                    and (result.get("abstained", False) or "generate_answer" in actions)
                )
            )
        ),
    }
    if expected_answerable is not None:
        metrics["unable_to_answer_correct"] = (
            result.get("abstained", False) is (not expected_answerable)
        )
    return metrics
