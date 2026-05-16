"""
RAGAs 四指标评估模块（轻量实现，LLM-as-Judge）

指标：
- Faithfulness：答案是否忠于检索到的文档
- Answer Relevancy：答案是否回答原始问题
- Context Precision：召回文档中有多少是真正有用的
- Context Recall：需要的文档都召回了没有（需 ground truth）
"""

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
    一站式评估，返回四指标分数。

    返回格式：
    {
        "faithfulness": float,
        "answer_relevancy": float,
        "context_precision": float,
        "context_recall": float or "N/A",
    }
    """
    context_text = "\n\n".join([
        doc.page_content for doc in retrieved_docs
    ]) if retrieved_docs else ""

    return {
        "faithfulness": round(faithfulness_score(question, answer, context_text), 2),
        "answer_relevancy": round(answer_relevancy_score(question, answer), 2),
        "context_precision": round(context_precision_score(question, retrieved_docs), 2),
        "context_recall": (
            round(context_recall_score(question, retrieved_docs, ground_truth_docs), 2)
            if ground_truth_docs else "N/A（需要 ground truth）"
        ),
    }
