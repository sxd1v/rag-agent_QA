import hashlib
import json
import re

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

    def execute(
        self,
        query: str,
        top_k: int = 5,
        strategy: str = "enhanced",
        enable_rerank: bool | None = None,
    ) -> dict:
        """
        执行文档检索。

        返回格式：
        {
            "query": str,           # 本次检索用的 query
            "count": int,          # 返回了多少个 chunk
            "docs": list[Document]  # 检索到的文档块
        }
        """
        docs = search_docs(
            query,
            top_k=top_k,
            strategy=strategy,
            enable_rerank=enable_rerank,
        )
        return {
            "query": query,
            "strategy": strategy,
            "enable_rerank": enable_rerank,
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

    def _parse_payload(self, text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:-1]).strip()
        try:
            return json.loads(text)
        except (TypeError, json.JSONDecodeError):
            match = re.search(r"\{.*\}", text, flags=re.S)
            if match:
                return json.loads(match.group(0))
            raise

    def _valid_citations(self, payload: dict, answer_text: str, valid_docs: dict) -> list[str]:
        raw = payload.get("citations", [])
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            raw = []
        from_payload = [chunk_id for chunk_id in raw if chunk_id in valid_docs]
        from_answer = [
            chunk_id for chunk_id in re.findall(r"\[(chunk-[a-f0-9]+)\]", answer_text)
            if chunk_id in valid_docs
        ]
        citations = []
        for chunk_id in from_payload + from_answer:
            if chunk_id not in citations:
                citations.append(chunk_id)
        return citations

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
        cache_key = f"llm:answer:v6:{hashlib.md5(key_raw.encode()).hexdigest()}"

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

        valid_chunk_ids = list(valid_docs)

        # 构建 prompt
        prompt = (
            "你是一个严格的有据问答助手。只能使用以下证据回答，不能补充常识或猜测。\n\n"
            f"【参考资料】\n{context_text}\n\n"
            f"【用户问题】{question}\n\n"
            f"【允许引用的 chunk_id】{valid_chunk_ids}\n\n"
            "判断证据能否直接支撑答案。如果不能，supported 必须为 false。\n"
            "若能回答，answer 中的事实陈述必须以 [chunk_id] 标注，citations 只能列出允许引用的 chunk_id。\n"
            "只要证据中有直接相关信息，就应基于证据简洁回答，不要因为表述不完全相同而拒答。\n"
            "回答必须直接回应用户问题中的核心术语和比较维度；如果问题问“是什么/用途/区别/为什么/如何”，答案要显式覆盖这些问法。\n"
            "定义类问题不要只给一句同义改写，应说明它在 RAG 链路中的位置、工作方式或用途。\n"
            "对于列举或对比问题，使用分号或短列表列出证据中的所有相关项，不要只回答其中一项。\n"
            "优先复用证据中的原文短语和描述，不要改写成更宽泛的说法。\n"
            "每个列表项或分号分隔的事实都要单独标注 [chunk_id]，不要只在整段末尾放一个引用。\n"
            "如果问题问“分别是哪两个例子/有哪些方式/如何组合”，答案要逐项给出，不要概括成一句。\n"
            "尽量保留证据或问题中的关键术语原文，例如：召回、Context、top-k、Hybrid Retrieval、Rerank、Query Rewrite。\n"
            "不要添加证据中没有直接出现的效果外推；例如证据只说 Rerank 优化排序时，不要扩展成提升生成质量或提升准确率。\n"
            "只输出 JSON："
            '{"supported": true/false, "answer": "答案或空字符串", '
            '"citations": ["chunk_id"], "missing": "证据不足时说明缺口"}'
        )

        payload = {}
        answer_text = ""
        citations = []
        supported = False
        missing = "生成结果无法验证"
        llm = get_chat_llm()
        for attempt in range(2):
            response = llm.invoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            try:
                payload = self._parse_payload(text)
            except (TypeError, ValueError, json.JSONDecodeError):
                payload = {"supported": False, "citations": [], "missing": "生成结果无法解析"}

            answer_text = str(payload.get("answer", "")).strip()
            citations = self._valid_citations(payload, answer_text, valid_docs)
            supported = bool(payload.get("supported")) and bool(answer_text) and bool(citations)
            missing = payload.get("missing", "没有可验证引用")
            if supported:
                break
            if attempt == 0:
                prompt = (
                    "上一次输出无法通过证据校验。请重新检查证据并修复 JSON。\n"
                    "如果证据能回答，必须使用允许引用的 chunk_id；如果不能回答，仍输出 supported=false。\n\n"
                    f"【允许引用的 chunk_id】{valid_chunk_ids}\n\n"
                    f"【参考资料】\n{context_text}\n\n"
                    f"【用户问题】{question}\n\n"
                    "回答必须直接覆盖用户问题中的核心术语和比较维度；定义类问题要说明位置、工作方式或用途；列举/对比/组合题逐项回答；优先复用证据原文短语；保留召回、Context、top-k、Rerank 等关键术语；不要添加证据外推，并在每个事实后标注 chunk_id。\n"
                    "只输出 JSON："
                    '{"supported": true/false, "answer": "答案或空字符串", '
                    '"citations": ["chunk_id"], "missing": "证据不足时说明缺口"}'
                )
        if not supported:
            result = {
                "answer": REFUSAL_ANSWER,
                "source_count": 0,
                "sources": [],
                "citations": [],
                "supported": False,
                "abstained": True,
                "missing": missing,
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
