"""Writer Agent —— 基于 Researcher 提供的证据生成答案，同时判断证据是否充分。"""
import json
from app.core.llm_client import get_chat_llm


def write_answer(question: str, context: list) -> dict:
    """
    输入：用户问题 + Researcher 搜来的证据块
    输出：{"answer": str, "confident": bool, "missing": str}

    answer：生成的答案文本
    confident：True=证据足够，False=需要 Researcher 重新搜
    missing：当 confident=False 时，说明缺什么
    """
    if not context:
        return {
            "answer": "",
            "confident": False,
            "missing": f"知识库中没有找到与 '{question}' 相关的任何内容",
        }

    # 构建证据文本
    context_text = "\n\n".join([
        d.page_content for d in context
    ])

    # prompt 同时让 LLM 做两件事：生成答案 + 自检证据是否充分
    prompt = (
        "你是一个专业问答助手。请根据参考资料回答用户问题。\n\n"
        f"【参考资料】\n{context_text}\n\n"
        f"【用户问题】\n{question}\n\n"
        "【要求】\n"
        "1. 基于参考资料生成答案，不要编造\n"
        "2. 判断参考资料是否足以回答这个问题\n"
        "3. 输出 JSON 格式（不要加引号或解释）：\n"
        '{{"answer": "你的答案",'
        ' "confident": true/false,'
        ' "missing": "如果证据不足，说明缺什么；否则为空字符串"}}'
    )

    # 调用 LLM
    llm = get_chat_llm()
    response = llm.invoke(prompt)
    text = response.content.strip() if hasattr(response, "content") else str(response).strip()

    # 解析 LLM 返回的 JSON
    try:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
        result = json.loads(text)
        return {
            "answer": result.get("answer", ""),
            "confident": bool(result.get("confident", False)),
            "missing": result.get("missing", ""),
        }
    except json.JSONDecodeError:
        # JSON 解析失败：把原文当答案，保守判断为 confident
        return {
            "answer": text,
            "confident": True,
            "missing": "",
        }
