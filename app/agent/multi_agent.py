"""
Multi-Agent 编排层：Researcher + Writer 协作。

Researcher 负责搜索证据，Writer 负责生成答案并自检。
Writer 不满意时把需求退回 Researcher 重新搜，最多 3 轮。
"""

from app.services.retriever import search_docs
from app.agent.tools import RewriteQueryTool
from app.agent.writer import write_answer


def research(question: str, extra_hint: str = "", max_rounds: int = 3) -> list:
    """
    Researcher Agent：搜索知识库收集证据。

    参数：
        question: 用户原始问题
        extra_hint: Writer 退回时的补充说明（如"缺定义、缺时间"）
        max_rounds: 最多搜几轮

    返回：Document 列表
    """
    all_docs = []
    seen_ids = set()
    query = question

    rewrite_tool = RewriteQueryTool()

    for round_num in range(max_rounds):
        # 如果有额外提示，拼到 query 里
        if extra_hint and round_num == 0:
            query = f"{question} {extra_hint}"

        # 执行混合检索
        docs = search_docs(query, top_k=5)

        # 去重收集
        new_count = 0
        for doc in docs:
            chunk_id = doc.metadata.get("chunk_id", "")
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                all_docs.append(doc)
                new_count += 1

        # 本轮有新收获就继续，没有就改写重试
        if new_count > 0:
            extra_hint = ""  # 首次提示只用一次
            if round_num < max_rounds - 1:
                # 用 rewrite 换角度再搜一轮
                rewritten = rewrite_tool.execute(query)
                query = rewritten["rewritten_query"]
        else:
            break  # 搜不到了，歇

    return all_docs


def orchestrate(question: str) -> dict:
    """
    编排 Researcher + Writer 协作。

    流程：
    1. Researcher 搜索证据
    2. Writer 生成答案 + 自检
    3. 自信 → 返回；不自信 → 把 missing 退回 Researcher 重新搜
    4. 最多 3 轮

    返回：{"answer": str, "rounds": int, "confident": bool}
    """
    extra_hint = ""

    for round_num in range(1, 4):
        # 1. Researcher 搜索
        context = research(question, extra_hint=extra_hint)

        # 2. Writer 写答案 + 自检
        result = write_answer(question, context)

        # 3. 自信或最后一轮 → 返回
        if result["confident"] or round_num >= 3:
            return {
                "answer": result["answer"],
                "rounds": round_num,
                "confident": result["confident"],
                "citations": result.get("citations", []),
                "sources": result.get("sources", []),
                "abstained": result.get("abstained", not result["confident"]),
            }

        # 4. 不自信：把需求退回 Researcher
        extra_hint = result["missing"]

    return {
        "answer": "抱歉，当前知识库证据不足，无法可靠回答这个问题。",
        "rounds": 3,
        "confident": False,
        "citations": [],
        "sources": [],
        "abstained": True,
    }
