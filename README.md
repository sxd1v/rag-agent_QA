# ReAct Agent 智能问答系统

基于 ReAct（Reasoning + Acting）模式的智能问答 Agent，融合 Advanced RAG 检索技术与 LLM 决策，并用工程约束限制无证据回答和无效循环。

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

- **LLM + 硬约束**：LLM 决定检索动作；最终答案必须引用当前 context 中真实存在的 `chunk_id`，否则返回拒答
- **8 字段 State 状态机**：query、retrieval_attempts、context、retrieval_history、step、done
- **检索轨迹**：逐轮保存 action、query、召回 chunk 与新增 chunk，响应中返回最终 citations
- **错误反馈闭环**：上一步结果回传 LLM，避免重复调用错误工具
- **防死循环保护**：最大检索/动作次数、重复召回无新增即收敛、错误熔断

### 2. 混合检索系统

```
query → 向量检索 top-10（语义）──┐
                                ├→ RRF 按排名融合 → 去重 → top-k
       → BM25 检索 top-10（精确）─┘
```

- **自实现 BM25**：jieba 中文分词 + 完整 BM25 公式
- **RRF**：只比排名不比分数，解决向量和 BM25 分数尺度不可比
- **公平融合**：增强检索中每个 query 的向量列表与 BM25 列表分别进入 RRF，两个检索器的同名次权重一致
- **Multi-Query**：LLM 生成多种问法同时检索

### 3. 三个 Tool

| Tool | 作用 |
|------|------|
| `search_docs` | Agent 默认走增强检索（Multi-Query + 向量/BM25 + RRF + Rerank）；普通模式可作为基线 |
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

索引缓存到 `data/bm25_index.pkl`。每次 `ingest.py` 写入完成后会用稳定 `chunk_id` upsert 向量、重建 BM25 并只清除召回缓存；服务进程也会检测缓存文件更新时间并自动重载，避免向量库与 BM25 使用不同版本的数据。

### 7. API 双模式 + 评估

| 端点 | 模式 | 说明 |
|------|------|------|
| `POST /ask` | 普通 RAG | 固定流程：检索→生成 |
| `POST /agent_ask` | ReAct Agent | LLM 自主决策检索策略 |
| `GET /health` | 健康检查 | — |
| `POST /retrieve_debug` | 调试 | 仅检索，排查召回问题 |
| `POST /evaluate` | 评估 | Agent 问答 + RAGAs 四指标 |

`/ask` 请求可传 `retrieval_strategy: "vector" | "hybrid" | "enhanced"`；`/agent_ask` 默认使用 `enhanced`。Agent 响应包含 `sources`、`citations`、`abstained`、`history[].retrieved_chunk_ids` 与 `trace_metrics`。

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
  "answer": "RAG 是一种结合检索和生成的技术架构。[chunk-a1b2]",
  "retrieval_attempts": 1,
  "final_query": "什么是 RAG？",
  "citations": ["chunk-a1b2"],
  "abstained": false,
  "sources": [{"chunk_id": "chunk-a1b2", "source": "data/test_knowledge.md", "content": "..."}],
  "history": [
    {"step": 1, "thought": "首次检索", "action": "search_docs",
     "action_args": {"query": "什么是 RAG？", "top_k": 3},
     "retrieved_chunk_ids": ["chunk-a1b2"], "new_chunk_ids": ["chunk-a1b2"],
     "observation": "检索返回了 3 个相关 chunk。"},
    {"step": 2, "thought": "LLM 决策：证据充足", "action": "generate_answer",
     "action_args": {},
     "observation": "生成了答案，共参考了 1 个证据块。"}
  ],
  "trace_metrics": {"citations_valid": true, "llm_calls": 3}
}
```

---

## 测试

```bash
python test_agent_decision.py   # Agent 决策测试
python test_hybrid_search.py    # 混合检索对比实验
python -m unittest discover -s tests -v  # 不调用外部 API 的回归测试
python eval_batch.py             # vector / hybrid / ReAct 三组批量对比
python eval_batch.py --cold-cache # 清除各 pipeline 间缓存的受控对比
```

批量评估集包含 30 个定义、流程、对比、跨段信息和无法回答问题；报告输出 Faithfulness、Answer Relevancy、Context Precision、无法回答准确率、响应耗时、LLM 调用次数与失败率。Agent 指标包含重复/无新增检索和动作序列合规度；要判断动作在语义上是否最优，仍需补人工标注。需要完整 Context Recall 时，请在 `/evaluate` 提供人工标注的 `ground_truth_chunk_ids`。

### 实测结果

2026-05-25 使用已配置的 MiniMax Embedding 与 SiliconFlow LLM 运行 `python eval_batch.py --workers 8`，结果保存在 `data/eval_report.json`。该次运行启用了服务缓存且按 `vector_rag`、`hybrid_rag`、`react_agent` 顺序执行，因此耗时与 LLM 调用次数是缓存可用时的实际表现，不作为冷缓存严格对照。

| Pipeline | 通过 | Faithfulness | Answer Relevancy | Context Precision | 无法回答准确率 | 平均耗时 | 平均 LLM 调用 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vector RAG | 24/30 | 0.80 | 0.80 | 0.22 | 1.00 | 109.3 s | 1.00 |
| Hybrid RAG | 28/30 | 0.97 | 0.97 | 0.21 | 0.80 | 93.8 s | 0.53 |
| ReAct Agent (enhanced) | 28/30 | 0.96 | 0.97 | 0.32 | 0.80 | 411.8 s | 5.73 |

ReAct Agent 的 30 条结果中，引用校验与动作序列合规度均为 `30/30`，平均检索次数为 `1.97`；同时发生 `22` 次无新增检索，说明增强链路在当前小型知识库上的额外检索成本明显，不能仅凭通过率声称优于 Hybrid RAG。

### Docker 验证

2026-05-25 已实际构建并运行镜像 `rag-api:evidence-grounded`，挂载本地 Chroma 数据目录验证：

| 验证项 | 结果 |
|---|---|
| `docker build -t rag-api:evidence-grounded .` | 成功 |
| `GET /health` | `200`, `{"status":"ok"}` |
| `POST /retrieve_debug`，`hybrid` | `200`，返回知识文档 chunk |
| `POST /agent_ask`，`enhanced` | `200`，`citations_valid=true`，引用 `chunk-104d6214b9dc2797` |

容器验证中发现并修复了混合检索并发初始化同一 Chroma client 的异常：向量存储现按 collection 加锁复用实例。

---

## 技术决策

| 问题 | 决策 | 原因 |
|------|------|------|
| 手写 ReAct vs LangGraph？ | 手写 | 展示底层理解 |
| LLM 决策 vs 硬编码规则？ | 混合约束 | LLM 选择检索动作，代码强制引用、拒答和收敛边界 |
| RRF vs 直接拼接分数？ | RRF | 向量和 BM25 分数分布不可比 |
| BM25 自实现 vs rank_bm25？ | 自实现 | 中文分词可控，展示算法理解 |

## 能力边界

- 引用校验、拒答和循环约束能降低无证据自信回答的风险，但不会彻底消除模型幻觉；效果结论应以标注测试集结果为准。
- 路由为异步接口，阻塞的模型/向量调用在线程池中执行；这不是高并发压测结论，生产部署仍需做容量与超时验证。
- session 在 Redis 可用时可跨进程共享，Redis 不可用会降级为单进程内存；长期记忆写入独立 `memory` collection，不参与 `knowledge` 文档检索与 BM25 建库。
- 本地单机知识库是当前默认部署形态。

## 待完成

- [x] Rerank 集成（LLM-as-Reranker）
- [x] RAGAs 评估集成（LLM-as-Judge）
- [x] BM25 索引持久化
- [x] Redis 可用时的缓存与 session 共享（无 Redis 时内存降级）
- [x] FastAPI 异步路由与阻塞调用线程池卸载
- [ ] 基于人工标注 chunk 的正式召回实验与并发压测
- [ ] 部署
