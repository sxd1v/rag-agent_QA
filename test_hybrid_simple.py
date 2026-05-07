"""
跳过 Rerank 模型加载，测试 Multi-Query + 混合检索 + RRF
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 强制跳过 Rerank 模型
import app.services.hybrid_retriever as hr
hr._cross_encoder = None  # 预设为 None，跳过模型加载

from app.services.hybrid_retriever import hybrid_search


def main():
    print("=" * 60)
    print("  两阶段检索实验（Multi-Query + Hybrid + RRF）")
    print("  Rerank 模型因网络原因跳过")
    print("=" * 60)

    # 单条测试
    question = "什么是 RAG？"

    print(f"\n原始问题: {question}")
    print("-" * 40)

    result = hybrid_search(question, top_k=5, num_queries=3)

    print(f"\n[OK] Final returned {len(result)} document chunks:")
    for i, doc in enumerate(result, 1):
        cid = doc.metadata.get("chunk_id", "?")
        content = doc.page_content[:120].replace("\n", " ")
        print(f"  [{i}] {cid}: {content}...")


if __name__ == "__main__":
    main()