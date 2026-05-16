"""测试 Multi-Agent：Researcher + Writer 协作"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.agent.multi_agent import orchestrate

questions = [
    "什么是 RAG？",
    "RAG 的缺点有哪些？",
    "向量数据库有哪些选择？",
]

for q in questions:
    print(f"\n{'='*60}")
    print(f"问题: {q}")
    print(f"{'='*60}")

    result = orchestrate(q)

    print(f"\n答案: {result['answer']}")
    print(f"协作轮数: {result['rounds']}")
    print(f"Writer 自信: {result['confident']}")
