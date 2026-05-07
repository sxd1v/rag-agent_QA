import hashlib
from app.services.retriever import search_docs
from app.core.llm_client import get_chat_llm
from app.core.cache import get as cache_get, set as cache_set


class SearchDocsTool:
    """搜索文档 Tool"""

    name = "search_docs"
    description = "根据问题从知识库检索最相关的 top-k 文档块。返回检索结果列表，每个元素包含 chunk 内容和相关性信息。"

    def execute(self, query: str, top_k: int = 5) -> dict:
        """
        执行文档检索。

        返回格式：
        {
            "query": str,           # 本次检索用的 query
            "count": int,          # 返回了多少个 chunk
            "docs": list[Document]  # 检索到的文档块
        }
        """
        docs = search_docs(query, top_k=top_k)
        return {
            "query": query,
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
                "answer": "抱歉，知识库中没有找到足够的证据来回答这个问题。",
                "source_count": 0,
            }

        # 用 chunk_id 列表生成缓存 key（只比对证据是否相同，不比对完整文本）
        chunk_ids = sorted([
            doc.metadata.get("chunk_id", "") for doc in context
        ])
        key_raw = f"{question}|{'|'.join(chunk_ids)}"
        cache_key = f"llm:answer:{hashlib.md5(key_raw.encode()).hexdigest()}"

        cached = cache_get(cache_key)
        if cached is not None:
            return cached

        # 构建 context 文本
        context_text = "\n\n".join([doc.page_content for doc in context])

        # 构建 prompt
        prompt = (
            f"你是一个专业的问答助手。请根据以下参考资料回答用户问题。\n\n"
            f"【参考资料】\n{context_text}\n\n"
            f"【用户问题】{question}\n\n"
            f"请结合参考资料给出准确、完整的回答。如果资料不足以回答，请如实说明。"
        )

        # 调用 LLM
        llm = get_chat_llm()
        answer = llm.invoke(prompt)
        answer_text = answer.content if hasattr(answer, "content") else str(answer)

        result = {
            "answer": answer_text,
            "source_count": len(context),
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
