"""
两阶段检索实验脚本
对比: 直接向量检索 vs Multi-Query + 混合检索 + RRF + Rerank

运行方式: python test_hybrid_search.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.hybrid_retriever import (
    hybrid_search, debug_hybrid_search, vector_search, rrf_fusion,
    generate_multi_queries, bm25_search, get_bm25_index
)
from app.services.retriever import search_docs


def pretty_docs(docs, max_len=120):
    for i, d in enumerate(docs, 1):
        cid = d.metadata.get("chunk_id", "?")
        content = d.page_content[:max_len].replace("\n", " ")
        print(f"  [{i}] {cid}: {content}...")


def main():
    questions = [
        "什么是 RAG?",
        "RAG 的检索增强是什么原理？",
        "Query Rewrite 有哪些方法？",
    ]

    for question in questions:
        print(f"\n{'=' * 65}")
        print(f"  Question: {question}")
        print(f"{'=' * 65}")

        # 直接向量检索
        direct_docs = search_docs(question, top_k=5)
        print(f"\n[Direct Vector Search] top-5:")
        pretty_docs(direct_docs)

        # 两阶段检索
        print(f"\n[Two-Stage: Multi-Query + Hybrid + RRF]")
        two_stage_docs = hybrid_search(question, top_k=5, num_queries=3)
        print(f"Result: {len(two_stage_docs)} docs")
        pretty_docs(two_stage_docs)

        # 重叠分析
        direct_ids = set(d.metadata.get("chunk_id", "") for d in direct_docs)
        two_stage_ids = set(d.metadata.get("chunk_id", "") for d in two_stage_docs)
        overlap = direct_ids & two_stage_ids
        print(f"\n  Direct IDs:      {list(direct_ids)}")
        print(f"  Two-Stage IDs:   {list(two_stage_ids)}")
        print(f"  Overlap:         {overlap if overlap else '(none)'}")
        print(f"  Only in Two-Stage: {two_stage_ids - direct_ids}")

    print(f"\n\n{'=' * 65}")
    print("  [Done] 两阶段检索实验完成")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()