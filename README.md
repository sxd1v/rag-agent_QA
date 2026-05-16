# ReAct Agent 智能问答系统

基于 ReAct（Reasoning + Acting）模式的智能问答 Agent，融合 Advanced RAG 检索技术与 LLM 自主决策，让模型在回答问题时能够自主判断"搜什么、搜几次、何时结束"，而非走固定流程。

**技术栈**：Python / FastAPI / LangChain / Chroma 

---

## 项目架构

```
rag_api/
├── app/
│   ├── agent/                     # Agent 核心
│   │   ├── react_loop.py          # ReAct 循环 + LLM 决策引擎
│   │   ├── react_state.py         # 8 字段 State 状态机
│   │   └── tools.py               # 3 个 Tool（search_docs/rewrite_query/generate_answer）
│   ├── api/
│   │   └── routes.py              # 路由层
│   ├── core/
│   │   ├── config.py              # 集中配置
│   │   └── llm_client.py          # LLM 工厂（公共模块）
│   ├── services/
│   │   ├── retriever.py           # 混合检索入口（向量+BM25+RRF）
│   │   ├── hybrid_retriever.py    # BM25 + RRF + Multi-Query 算法
│   │   └── qa_service.py          # 普通 RAG 模式问答
│   ├── db/
│   │   └── vector_store.py        # Chroma 向量库接入
│   └── schemas.py                 # Pydantic 数据模型
├── test_agent_decision.py
└── test_hybrid_search.py
```

### 模块职责边界

| 模块 | 负责 | 不负责 |
|------|------|--------|
| `agent/` | ReAct 循环、LLM 决策、Tool 管理 | 检索实现、API 格式 |
| `services/retriever.py` | 混合检索编排 | Agent 决策 |
| `services/hybrid_retriever.py` | BM25/RRF 算法 | State 管理 |
| `api/routes.py` | 接请求、调模块、回响应 | 检索、生成 |

---

## 核心特性

### 1. ReAct Agent 决策引擎

传统 RAG 是固定流水线，Agent 每步根据状态动态决策：

```
用户问题 → search_docs → LLM 评估检索结果质量 → 不够→改写重试 / 够了→生成答案
```

- **LLM 驱动决策**：LLM 根据检索结果的内容质量（非数量）判断下一步，非硬编码 if-else
- **8 字段 State 状态机**：query、retrieval_attempts、context、retrieval_history、step、done
- **keep_chunks 机制**：LLM 显式选择哪些检索结果进入 context
- **错误反馈闭环**：上一步结果回传 LLM，避免重复调用错误工具
- **防死循环保护**：检索次数耗尽强制兜底

### 2. 混合检索系统

```
query → 向量检索 top-10（语义）──┐
                                ├→ RRF 按排名融合 → 去重 → top-k
       → BM25 检索 top-10（精确）─┘
```

- **自实现 BM25**：jieba 中文分词 + 完整 BM25 公式
- **RRF**：只比排名不比分数，解决向量和 BM25 分数尺度不可比
- **Multi-Query**：LLM 生成多种问法同时检索

### 3. 三个 Tool

| Tool | 作用 |
|------|------|
| `search_docs` | 混合检索（向量+BM25+RRF） |
| `rewrite_query` | LLM 改写查询（同义词、分解、具体化） |
| `generate_answer` | 基于证据块生成答案 |

### 4. Rerank 精排（LLM-as-Reranker）

向量检索召回的 top-k 候选，语义相似不代表"真正回答问题的相关"。Rerank 精排这一步：

- 用 LLM 对所有候选批量打分，选出最相关的 top-n
- 相比扩大 top_k，Rerank 在有限 context 窗口内放入更高质量的证据
- 不依赖 BGE/Cohere 等外部模型，用现有 LLM 即可

### 5. RAGAs 评估（LLM-as-Judge）

`/evaluate` 端点自动跑 Agent 问答 + 四指标评估：

| 指标 | 衡量什么 | 实现方式 |
|------|---------|---------|
| Faithfulness | 答案是否忠于文档 | LLM 逐句检查 |
| Answer Relevancy | 答案是否回答原问题 | LLM 判断 |
| Context Precision | 召回文档相关比例 | LLM 逐文档判断 |
| Context Recall | 相关文档是否全召回 | 需 ground truth |

### 6. BM25 索引持久化

首次构建后 pickle 缓存到 `data/bm25_index.pkl`，后续启动直接加载，省去 Chroma 全量导出和分词的耗时。

### 7. API 双模式 + 评估

| 端点 | 模式 | 说明 |
|------|------|------|
| `POST /ask` | 普通 RAG | 固定流程：检索→生成 |
| `POST /agent_ask` | ReAct Agent | LLM 自主决策检索策略 |
| `GET /health` | 健康检查 | — |
| `POST /retrieve_debug` | 调试 | 仅检索，排查召回问题 |
| `POST /evaluate` | 评估 | Agent 问答 + RAGAs 四指标 |

---

## 快速启动

```bash
cd rag_api
pip install -r requirements.txt
```

配置 `.env`：
```env
SILICONFLOW_API_KEY=your_key
MINIMAX_API_KEY=your_key
EMBEDDING_PROVIDER=minimax
```

```bash
# 启动服务
uvicorn app.main:app --reload

# 调用 Agent
curl -X POST http://localhost:8000/agent_ask \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是 RAG？"}'
```

### 响应示例

```json
{
  "answer": "RAG 是一种结合检索和生成的技术架构...",
  "retrieval_attempts": 1,
  "history": [
    {"step": 1, "thought": "首次检索", "action": "search_docs",
     "observation": "检索返回了 3 个相关 chunk。"},
    {"step": 2, "thought": "LLM 决策：证据充足", "action": "generate_answer",
     "observation": "生成了答案，共参考了 3 个证据块。"}
  ]
}
```

---

## 测试

```bash
python test_agent_decision.py   # Agent 决策测试
python test_hybrid_search.py    # 混合检索对比实验
```

---

## 技术决策

| 问题 | 决策 | 原因 |
|------|------|------|
| 手写 ReAct vs LangGraph？ | 手写 | 展示底层理解 |
| LLM 决策 vs 硬编码规则？ | LLM | 规则只看数量，LLM 懂内容 |
| RRF vs 直接拼接分数？ | RRF | 向量和 BM25 分数分布不可比 |
| BM25 自实现 vs rank_bm25？ | 自实现 | 中文分词可控，展示算法理解 |

## 待完成

- [x] Rerank 集成（LLM-as-Reranker）
- [x] RAGAs 评估集成（LLM-as-Judge）
- [x] BM25 索引持久化
- [ ] Redis 缓存
- [ ] FastAPI 异步改造
- [ ] 部署
