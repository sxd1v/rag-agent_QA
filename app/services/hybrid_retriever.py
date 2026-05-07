"""
两阶段检索服务：Multi-Query + 混合检索 + RRF融合 + Rerank精排

流程：
1. Multi-Query: LLM 生成多个检索表达
2. 混合检索: 同时跑向量检索 + BM25
3. RRF 融合: 把所有 query 的结果按排名合并
4. Rerank: Cross-Encoder 精排，输出最终 top-k
"""

import os
import pickle
import jieba
from typing import List, Tuple
from langchain_core.documents import Document

from app.db.vector_store import get_vector_store
from app.core.llm_client import get_chat_llm

# BM25 索引持久化路径
_BM25_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "bm25_index.pkl"
)


# ============== BM25 辅助 ==============

class BM25:
    """
    轻量 BM25 实现（中文分词版）。
    不依赖 rank_bm25 的默认英文 tokenizer。
    """

    def __init__(self, corpus: List[Document]):
        self.doc_texts = [doc.page_content for doc in corpus]
        self.doc_ids = [doc.metadata.get("chunk_id", str(i)) for i, doc in enumerate(corpus)]
        # 手动分词（中文用 jieba）
        self.tokenized_corpus = [self._tokenize(text) for text in self.doc_texts]
        # 计算 IDF
        self._compute_idf()
        # 计算文档长度和平均长度
        self.doc_len = [len(t) for t in self.tokenized_corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0

    def _tokenize(self, text: str) -> List[str]:
        """中文分词"""
        return [w for w in jieba.cut(text) if w.strip()]

    def _compute_idf(self):
        """计算每个词的 IDF 值"""
        import math
        N = len(self.tokenized_corpus)
        df = {}
        for doc_tokens in self.tokenized_corpus:
            unique_tokens = set(doc_tokens)
            for token in unique_tokens:
                df[token] = df.get(token, 0) + 1
        self.idf = {}
        for token, freq in df.items():
            self.idf[token] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        """对单个 query token 列表计算每个文档的 BM25 分数"""
        import math
        scores = []
        k1, b = 1.5, 0.75
        for i, doc_tokens in enumerate(self.tokenized_corpus):
            score = 0.0
            doc_len = self.doc_len[i]
            for token in query_tokens:
                if token not in self.idf:
                    continue
                tf = doc_tokens.count(token)
                idf = self.idf[token]
                # BM25 公式
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_len / max(self.avgdl, 1))
                score += idf * numerator / denominator
            scores.append(score)
        return scores

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        BM25 检索，返回 (doc_index, score) 列表，按分数从高到低。
        """
        query_tokens = self._tokenize(query)
        scores = self.get_scores(query_tokens)
        # 排序
        doc_scores = list(enumerate(scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return doc_scores[:top_k]


# ============== Multi-Query ==============

def generate_multi_queries(original_query: str, num_queries: int = 3) -> List[str]:
    """
    用 LLM 生成多个不同表达形式的 query。

    策略：
    - 同义词替换
    - 问题分解
    - 表述转换（问句→关键词）
    - 具体化/泛化

    返回包含原始 query 在内的 num_queries 个检索表达。
    """
    prompt = (
        "你是一个查询改写专家。请将用户问题改写为 {n} 个不同的检索表达形式。\n\n"
        "改写策略：\n"
        "1. 同义词替换（如'RAG'→'Retrieval-Augmented Generation'）\n"
        "2. 问题分解（复杂问题拆成多个子问题）\n"
        "3. 表述转换（问句→关键词组合）\n"
        "4. 具体化（泛化问题加上领域限定）\n\n"
        "【重要】输出格式：每行一个改写后的查询，不要加编号、不要加引号、不要解释。\n\n"
        f"原始问题：{original_query}\n\n"
        "改写后："
    ).format(n=num_queries - 1)  # 因为包含原始 query

    llm = get_chat_llm()
    response = llm.invoke(prompt)
    text = response.content.strip() if hasattr(response, "content") else str(response).strip()

    # 解析多行输出
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    # 过滤掉可能的引号或编号
    queries = []
    for line in lines:
        line = line.strip("•-*1234567890.、　 ")
        if line and line != original_query:
            queries.append(line)
        if len(queries) >= num_queries - 1:
            break

    # 确保包含原始 query
    all_queries = [original_query] + queries[: num_queries - 1]
    return all_queries


# ============== 向量检索 ==============

def vector_search(query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
    """
    向量检索，返回 (Document, score) 列表。
    注意：Chroma 的 similarity_search 返回 Document，没有分数。
    这里返回 1.0 作为默认分数（RRF 只看排名，不看分数）。
    """
    vector_store = get_vector_store()
    docs = vector_store.similarity_search(query=query, k=top_k)
    # Chroma 不返回分数，我们用固定分数 1.0 让 RRF 按排名计算
    return [(doc, 1.0) for doc in docs]


# ============== BM25 检索 ==============

# 模块级别缓存（避免每次检索都重新建索引）
_bm25_index: BM25 = None


def get_bm25_index() -> BM25:
    """获取 BM25 索引（优先从缓存加载，缓存不存在则构建并持久化）"""
    global _bm25_index
    if _bm25_index is not None:
        return _bm25_index

    # 尝试从缓存加载
    if os.path.exists(_BM25_CACHE_PATH):
        try:
            with open(_BM25_CACHE_PATH, "rb") as f:
                _bm25_index = pickle.load(f)
            print(f"[BM25] 从缓存加载，共 {len(_bm25_index.doc_texts)} 条文档")
            return _bm25_index
        except Exception:
            print("[BM25] 缓存加载失败，重新构建")

    # 从 Chroma 获取全部文档构建 BM25
    vector_store = get_vector_store()
    all_docs = vector_store.get(limit=10000, include=["documents", "metadatas"])
    docs = [
        Document(page_content=doc, metadata=meta or {})
        for doc, meta in zip(all_docs["documents"], all_docs["metadatas"])
    ]
    _bm25_index = BM25(docs)
    print(f"[BM25] 构建完成，共 {len(docs)} 条文档")

    # 持久化到缓存
    os.makedirs(os.path.dirname(_BM25_CACHE_PATH), exist_ok=True)
    with open(_BM25_CACHE_PATH, "wb") as f:
        pickle.dump(_bm25_index, f)
    print(f"[BM25] 已缓存到 {_BM25_CACHE_PATH}")

    return _bm25_index


def bm25_search(query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
    """BM25 检索"""
    bm25 = get_bm25_index()
    results = bm25.search(query, top_k=top_k)
    return [
        (Document(page_content=bm25.doc_texts[i], metadata={"chunk_id": bm25.doc_ids[i]}), score)
        for i, score in results if score > 0
    ]


# ============== RRF 融合 ==============

def rrf_fusion(
    results_list: List[List[Tuple[Document, float]]],
    k: int = 60,
) -> List[Tuple[Document, float]]:
    """
    Reciprocal Rank Fusion：把多组检索结果按排名融合。

    公式：score(q, d) = Σ(1 / (k + rank(d)))

    参数：
    - results_list: 每个元素是一组 (Document, score) 列表
    - k: 经验值 60，避免排名差异被过度放大
    """
    doc_scores = {}

    for results in results_list:
        for rank, (doc, _) in enumerate(results, start=1):
            chunk_id = doc.metadata.get("chunk_id", doc.page_content[:50])
            if chunk_id not in doc_scores:
                doc_scores[chunk_id] = {"doc": doc, "score": 0.0}
            doc_scores[chunk_id]["score"] += 1.0 / (k + rank)

    # 排序
    sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
    return [(item["doc"], item["score"]) for item in sorted_docs]


# ============== Rerank ==============

def rerank_documents(
    query: str,
    candidates: List[Document],
    top_k: int = 5,
) -> List[Document]:
    """
    LLM-as-Reranker：用 LLM 对候选文档批量打分精排。

    一次性传入所有候选，让 LLM 输出最相关的文档编号，
    避免逐个打分刷爆 API。

    参数：
    - query: 用户问题
    - candidates: 候选文档列表
    - top_k: 返回前几名

    返回精排后的 Document 列表。
    """
    if not candidates:
        return []
    if len(candidates) <= top_k:
        return candidates

    # 构造候选文档摘要
    doc_list = []
    for i, doc in enumerate(candidates):
        cid = doc.metadata.get("chunk_id", f"doc-{i}")
        preview = doc.page_content[:200].replace("\n", " ")
        doc_list.append(f"[{i}] {cid}: {preview}")

    prompt = (
        "你是一个检索排序专家。请根据用户问题，从以下候选文档中选出最相关的 N 篇。\n\n"
        f"【用户问题】\n{query}\n\n"
        f"【候选文档】\n" + "\n".join(doc_list) + "\n\n"
        f"请选出最相关的 {top_k} 篇文档，只输出文档编号（如 0,3,5），不要解释：\n"
    )

    llm = get_chat_llm()
    response = llm.invoke(prompt)
    text = response.content.strip() if hasattr(response, "content") else str(response).strip()

    # 解析 LLM 返回的编号
    try:
        # 提取数字
        import re
        indices = [int(x) for x in re.findall(r'\d+', text) if 0 <= int(x) < len(candidates)]
        # 去重保持顺序
        seen = set()
        indices = [i for i in indices if not (i in seen or seen.add(i))]
        result = [candidates[i] for i in indices[:top_k]]
        # 补足不足的部分
        for i, doc in enumerate(candidates):
            if len(result) >= top_k:
                break
            if i not in indices:
                result.append(doc)
        return result[:top_k]
    except Exception:
        print("[Rerank] LLM 输出解析失败，返回 RRF 原始顺序")
        return candidates[:top_k]


# ============== 顶层接口 ==============

def hybrid_search(query: str, top_k: int = 5, num_queries: int = 3) -> List[Document]:
    """
    完整的两阶段检索流程。

    参数：
    - query: 用户问题
    - top_k: 最终返回几个文档
    - num_queries: Multi-Query 生成几个表达（包含原始）

    返回最终精排后的 Document 列表。
    """
    print(f"\n[Hybrid Search] 原始 query: {query}")

    # Step 1: Multi-Query
    queries = generate_multi_queries(query, num_queries=num_queries)
    print(f"[Multi-Query] 生成了 {len(queries)} 个检索表达: {queries}")

    # Step 2: 每个 query 同时跑向量 + BM25，用 RRF 融合
    all_results = []
    for q in queries:
        # 向量检索 top-10
        vector_results = vector_search(q, top_k=10)
        # BM25 检索 top-10
        bm25_results = bm25_search(q, top_k=10)
        # 合并这两组结果（不做 RRF，每个 query 各自做一次融合）
        # 或者直接 concat 交给上层的 RRF
        all_results.append(vector_results + bm25_results)

    # RRF 融合所有 query 的结果
    fused = rrf_fusion(all_results, k=60)
    # 取前 top_k * 2 个候选（给 Rerank 留空间）
    candidates = [doc for doc, _ in fused[: top_k * 2]]
    print(f"[RRF 融合] 候选文档数: {len(candidates)}")

    # Step 3: Rerank 精排
    final_docs = rerank_documents(query, candidates, top_k=top_k)
    print(f"[Rerank] 最终输出: {len(final_docs)} 个文档")

    return final_docs


# ============== 调试接口 ==============

def debug_hybrid_search(query: str, top_k: int = 5) -> dict:
    """
    返回完整调试信息的字典，用于对比两阶段 vs 直接向量检索。
    """
    # 两阶段结果
    two_stage_results = hybrid_search(query, top_k=top_k)

    # 直接向量检索（对比用）
    direct_results = vector_search(query, top_k=top_k)
    direct_docs = [doc for doc, _ in direct_results]

    return {
        "query": query,
        "two_stage_docs": [
            {"chunk_id": d.metadata.get("chunk_id"), "content": d.page_content[:200]}
            for d in two_stage_results
        ],
        "direct_vector_docs": [
            {"chunk_id": d.metadata.get("chunk_id"), "content": d.page_content[:200]}
            for d in direct_docs
        ],
        "two_stage_ids": [d.metadata.get("chunk_id") for d in two_stage_results],
        "direct_ids": [d.metadata.get("chunk_id") for d in direct_docs],
    }