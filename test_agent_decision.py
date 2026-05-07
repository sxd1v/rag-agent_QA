"""测试升级后的 LLM 决策 Agent"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.agent.react_loop import run_react_loop

# 测试问题
test_questions = [
    "什么是 RAG？",
    "RAG 的缺点有哪些？",
]

for q in test_questions:
    print(f"\n{'='*60}")
    print(f"问题: {q}")
    print(f"{'='*60}")

    result = run_react_loop(q)

    print(f"\n最终答案:\n{result['answer']}")
    print(f"\n检索次数: {result['retrieval_attempts']}")
    print(f"最终 query: {result['final_query']}")
    print(f"\n决策历史 ({len(result['history'])} 步):")
    for step in result['history']:
        print(f"  Step {step['step']}:")
        print(f"    Thought: {step['thought'][:100]}...")
        print(f"    Action: {step['action']}")
        print(f"    Args: {step['action_args']}")
        print(f"    Observation: {step['observation'][:100]}")
