"""Writer Agent：复用统一的证据引用门控生成答案。"""

from app.agent.tools import GenerateAnswerTool


def write_answer(question: str, context: list) -> dict:
    """
    输入：用户问题 + Researcher 搜来的证据块
    输出：{"answer": str, "confident": bool, "missing": str}

    answer：生成的答案文本
    confident：True=证据足够，False=需要 Researcher 重新搜
    missing：当 confident=False 时，说明缺什么
    """
    result = GenerateAnswerTool().execute(question, context)
    return {
        "answer": result["answer"],
        "confident": result.get("supported", False),
        "missing": result.get("missing", ""),
        "citations": result.get("citations", []),
        "sources": result.get("sources", []),
        "abstained": result.get("abstained", False),
    }
