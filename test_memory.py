"""测试多轮记忆：同一 session 下 Agent 能记住上一轮对话"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.agent.react_loop import run_react_loop

session_id = "test-memory-session"

# 第一轮
print("=" * 50)
print("[第1轮] 用户: 什么是 RAG？")
r1 = run_react_loop("什么是 RAG？", session_id=session_id)
print(f"Agent: {r1['answer'][:100]}...\n")

# 第二轮：利用记忆，省略主语
print("[第2轮] 用户: 它的缺点是什么？")
r2 = run_react_loop("它的缺点是什么？", session_id=session_id)
print(f"Agent: {r2['answer'][:100]}...\n")

# 第三轮：继续追问
print("[第3轮] 用户: 怎么解决这些缺点？")
r3 = run_react_loop("怎么解决这些缺点？", session_id=session_id)
print(f"Agent: {r3['answer'][:100]}...\n")

print("=" * 50)
print("多轮记忆测试完成。检查第2、3轮回答是否关联了之前的上下文。")
