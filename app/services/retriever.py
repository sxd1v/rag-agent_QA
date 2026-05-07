from concurrent.futures import ThreadPoolExecutor
from app.services.hybrid_retriever import bm25_search, vector_search, rrf_fusion
from app.core.cache import get as cache_get, set as cache_set, clear as cache_clear

# 线程池：向量和 BM25 并行检索共用
_executor = ThreadPoolExecutor(max_workers=2)


def clear_retrieval_cache():
    """清空检索缓存（知识库更新后调用）"""
    cache_clear()


def search_docs(query: str, top_k: int = 5) -> list:
    """
    混合检索（Redis缓存 + 线程池并行）：向量 + BM25 同时发起 → RRF 融合 → top-k。

    Redis 可用时缓存持久化、跨进程共享；不可用时自动降级到内存字典。

    Args:
        query: 检索 query
        top_k: 返回多少个结果

    Returns:
        Document 对象列表，按相关性从高到低排序
    """
    # 检查缓存（Redis 优先，内存兜底）
    cache_key = f"retrieval:{query}:{top_k}"
    cached = cache_get(cache_key)
    if cached is not None:
        print(f"[Cache] 命中: {cache_key}")
        return cached

    # 两路并行检索：线程池同时提交，等两者都返回
    future_vec = _executor.submit(vector_search, query, 10)
    future_bm25 = _executor.submit(bm25_search, query, 10)
    vector_results = future_vec.result()
    bm25_results = future_bm25.result()

    # RRF 按排名融合
    fused = rrf_fusion([vector_results, bm25_results], k=60)

    # 取 top-k + 去重
    seen = set()
    final_docs = []
    for doc, score in fused:
        chunk_id = doc.metadata.get("chunk_id", doc.page_content[:50])
        if chunk_id not in seen:
            seen.add(chunk_id)
            final_docs.append(doc)
        if len(final_docs) >= top_k:
            break

    # 存入缓存（Redis 优先，内存兜底）
    cache_set(cache_key, final_docs)

    return final_docs


def retrieve_debug(question: str, top_k: int = 5) -> list:
    """
    调试用：只做检索，不做生成。
    返回检索到的文档块，供排查召回问题使用。
    """
    return search_docs(question, top_k=top_k)
