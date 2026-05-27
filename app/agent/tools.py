import hashlib
import json

from app.services.retriever import search_docs
from app.core.llm_client import get_chat_llm
from app.core.cache import get as cache_get, set as cache_set

REFUSAL_ANSWER = "抱歉，当前知识库证据不足，无法可靠回答这个问题。"


def document_source(doc) -> dict:
    """将证据文档转换为可序列化的来源记录。"""
    return {
        "chunk_id": doc.metadata.get("chunk_id", "unknown"),
        "source": doc.metadata.get("source"),
        "content": doc.page_content,
    }


class SearchDocsTool:
    """搜索文档 Tool"""

    name = "search_docs"
    description = "根据问题从知识库检索最相关的 top-k 文档块。返回检索结果列表，每个元素包含 chunk 内容和相关性信息。"

    def execute(self, query: str, top_k: int = 5, strategy: str = "enhanced") -> dict:
        """
        执行文档检索。

        返回格式：
        {
            "query": str,           # 本次检索用的 query
            "count": int,          # 返回了多少个 chunk
            "docs": list[Document]  # 检索到的文档块
        }
        """
        docs = search_docs(query, top_k=top_k, strategy=strategy)
        return {
            "query": query,
            "strategy": strategy,
            "count": len(docs),
            "docs": docs,
        }


class RewriteQueryTool:
    """改写查询问题 Tool（LLM 驱动）"""

    name = "rewrite_query"
    description = "将用户问题改写为更适合检索的形式，可以同义词扩展、具体化、分解问题。返回改写后的查询字符串。"

    def __init__(self):
        # 实例级别：每次创建新实例时重置
        self._used_rewrites: set = set()  # 记录已用过的改写版本（用于去重）

    def reset(self):
        """重置状态（每个请求开始时调用）"""
        self._used_rewrites.clear()

    def execute(self, original_query: str) -> dict:
        """
        改写 query（LLM 驱动 + 缓存）。

        相同 query 直接从缓存返回，省去 LLM 调用。
        """
        # 检查 LLM 缓存
        cache_key = f"llm:rewrite:{original_query}"
        cached = cache_get(cache_key)
        if cached is not None:
            self._used_rewrites.add(cached["rewritten_query"])
            return cached

        # 构建改写 prompt
        rewrite_prompt = (
            "你是一个查询改写专家。你的任务是对给定的用户问题进行改写，使其更适合知识库检索。\n\n"
            "改写策略包括：\n"
            "1. 同义词替换（如'RAG'→'Retrieval-Augmented Generation'）\n"
            "2. 问题分解（复杂问题拆成多个子问题）\n"
            "3. 具体化（泛化问题加上领域限定）\n"
            "4. 表述转换（问句→关键词组合）\n\n"
            "【重要】直接输出改写后的查询语句，不要解释，不要加引号或前缀。\n\n"
            f"原始问题：{original_query}\n\n"
            "改写后："
        )

        # 调用 LLM
        llm = get_chat_llm()
        response = llm.invoke(rewrite_prompt)
        rewritten = response.content.strip() if hasattr(response, "content") else str(response).strip()

        # 记录本次改写（用于后续去重提示）
        self._used_rewrites.add(rewritten)

        result = {
            "original_query": original_query,
            "rewritten_query": rewritten,
        }

        # 存入缓存
        cache_set(cache_key, result)

        return result


class GenerateAnswerTool:
    """生成答案 Tool"""

    name = "generate_answer"
    description = "基于收集到的证据块和问题生成最终答案。答案必须有证据支撑。"

    def execute(self, question: str, context: list) -> dict:
        """
        生成答案（LLM + 缓存）。

        相同 question + 相同 context（按 chunk_id 判断）直接从缓存返回。
        """
        if not context:
            return {
                "answer": REFUSAL_ANSWER,
                "source_count": 0,
                "sources": [],
                "citations": [],
                "supported": False,
                "abstained": True,
            }

        # 用 chunk_id 列表生成缓存 key（只比对证据是否相同，不比对完整文本）
        chunk_ids = sorted([
            doc.metadata.get("chunk_id", "") for doc in context
        ])
        key_raw = f"{question}|{'|'.join(chunk_ids)}"
        cache_key = f"llm:answer:v2:{hashlib.md5(key_raw.encode()).hexdigest()}"

        cached = cache_get(cache_key)
        if cached is not None:
            return cached

        valid_docs = {
            doc.metadata.get("chunk_id"): doc
            for doc in context
            if doc.metadata.get("chunk_id")
        }
        context_text = "\n\n".join(
            f"[{chunk_id}] source={doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
            for chunk_id, doc in valid_docs.items()
        )

        # 构建 prompt
        prompt = (
            "你是一个严格的有据问答助手。只能使用以下证据回答，不能补充常识或猜测。\n\n"
            f"【参考资料】\n{context_text}\n\n"
            f"【用户问题】{question}\n\n"
            "判断证据能否直接支撑答案。如果不能，supported 必须为 false。\n"
            "若能回答，answer 中的事实陈述必须以 [chunk_id] 标注，citations 只能列出提供过的 chunk_id。\n"
            "只输出 JSON："
            '{"supported": true/false, "answer": "答案或空字符串", '
            '"citations": ["chunk_id"], "missing": "证据不足时说明缺口"}'
        )

        # 调用 LLM
        llm = get_chat_llm()
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        try:
            text = text.strip()
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:-1])
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            payload = {"supported": False, "citations": [], "missing": "生成结果无法验证"}

        requested_citations = payload.get("citations", [])
        citations = [
            chunk_id for chunk_id in requested_citations
            if chunk_id in valid_docs
        ]
        answer_text = str(payload.get("answer", "")).strip()
        supported = bool(payload.get("supported")) and bool(answer_text) and bool(citations)
        if not supported:
            result = {
                "answer": REFUSAL_ANSWER,
                "source_count": 0,
                "sources": [],
                "citations": [],
                "supported": False,
                "abstained": True,
                "missing": payload.get("missing", "没有可验证引用"),
            }
        else:
            if not any(f"[{chunk_id}]" in answer_text for chunk_id in citations):
                answer_text = f"{answer_text}\n\n引用：" + " ".join(f"[{chunk_id}]" for chunk_id in citations)
            result = {
                "answer": answer_text,
                "source_count": len(citations),
                "sources": [document_source(valid_docs[chunk_id]) for chunk_id in citations],
                "citations": citations,
                "supported": True,
                "abstained": False,
            }

        # 存入缓存
        cache_set(cache_key, result)

        return result


# 模块级别：持久化 tool 实例，同一 name 返回同一个实例
_tool_instances: dict = {}


def get_tools():
    """返回所有可用的 Tool 列表（单例）"""
    global _tool_instances
    if not _tool_instances:
        _tool_instances = {
            "search_docs": SearchDocsTool(),
            "rewrite_query": RewriteQueryTool(),
            "generate_answer": GenerateAnswerTool(),
        }
    return list(_tool_instances.values())


def get_tool_by_name(name: str):
    """根据 name 获取 Tool 实例（单例）"""
    if not _tool_instances:
        get_tools()  # 初始化
    return _tool_instances.get(name)
