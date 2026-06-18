# 编排模型说明：多范式平台（Fixed Graph + Supervisor）

`agent-platform` 是 **多编排范式** 的多智能体平台 SDK，并可选 **Router Engine（Phase 10）** 作为企业默认入口。默认 **Workflow Profile**（原 Fixed Graph）；`profile=auto` 时先 classification 再选执行后端。

## 平台入口

```python
from agent_framework.bootstrap.platform import create_runtime

await create_runtime("travel", mode="fixed_graph").process_request("北京天气")
await create_runtime("travel", mode="supervisor").process_request("北京天气")
await create_runtime("travel", mode="supervisor", transport="mixed").process_request("订上海酒店")
await create_runtime("customer_service", profile="auto").process_request("投诉并咨询退货")
```

HTTP：`profile=auto|workflow|adaptive`（推荐 `auto`）；遗留 `mode` 仍可用。

---

## Fixed Graph（默认）

```
用户请求
   ↓
[pre_survey?] → [retrieve_memory?] → build_plan → execute_layer (分层) → aggregate → [save_memory?]
```

| 特点 | 说明 |
|------|------|
| 拓扑 | **编译期固定** StateGraph，节点集合由 `PipelineConfig` 开关裁剪 |
| 规划 | 中心 LLM 一次性（或分层）拆解子任务，写入 `execution_plan` |
| 执行 | 按层并行调用子 Agent，结果写入 `subtask_results` |
| 聚合 | 中心 LLM 合并子任务输出为 `final_response` |
| 扩展点 | **DomainPlugin**（子 Agent 集 + prompts + 路由策略） |
| 适用 | 任务可分解、子 Agent 职责清晰：客服分流、行程规划、金融问答流水线 |

**推荐入口**：`create_runtime(domain, mode="fixed_graph")`。

## Supervisor（Phase 7B）

```
用户请求 → Supervisor LLM 动态 handoff → 子 Agent 子图 → 循环直至结束
```

| 特点 | 说明 |
|------|------|
| 实现 | `agent_framework/orchestration/supervisor/` + `langgraph-supervisor` |
| 依赖 | `pip install "agent-platform[supervisor]"` |
| 领域配置 | `DomainPrompts.supervisor_system`（空则按 registry 自动生成） |
| 适用 | 工具链不确定、需多轮试探、对话式调度 |

**推荐入口**：`create_runtime(domain, mode="supervisor")`。

## A2A Transport（Phase 7C）

```
Supervisor handoff → 本地子图 和/或  A2A 远程 Agent（HTTP）
```

| 特点 | 说明 |
|------|------|
| 传输 | `transport=local`（默认）全本地；`a2a` 仅远程；`mixed` 远程替代同名本地 Agent |
| 配置 | 领域插件 `A2AEndpoint` / `create_a2a_endpoints`；travel 示例：`TRAVEL_A2A_HOTEL_URL` |
| 依赖 | `pip install "agent-platform[a2a]"`（`a2a-sdk`） |
| 范围 | **仅 Supervisor**；Fixed Graph 不走 A2A |

书稿参考：`Chapter-7/A2A/a2a_agents.py`（hotel @ `http://127.0.0.1:9012/`）。

**暴露本地子 Agent 为 A2A Server**（Phase 7D）：见 [`docs/a2a_server.md`](a2a_server.md)。

**推荐入口**：`create_runtime(domain, mode="supervisor", transport="mixed")`。

## 可观测性（Phase 8 Tracing）

| Span / Event | 路径 |
|--------------|------|
| `agent.invoke` | Supervisor 本地 handoff（`invoke_traced.py`） |
| `a2a.call` | A2A Client 远程调用（含 W3C traceparent inject） |
| `a2a.server.invoke` | A2A Server 处理入站请求 |
| `handoff.completed` | Supervisor 根 span 上的 handoff 汇总 event |
| `sub_agent_conversation` / `a2a.error` | 子 Agent 对话与 A2A 失败语义 |

### Prometheus（Phase 8B）

| 指标 | Labels |
|------|--------|
| `agent_platform_chat_requests_total` | `domain`, `mode`, `transport`, `status` |
| `agent_platform_job_requests_total` | `domain`, `mode`, `transport` |
| `agent_platform_job_outcomes_total` | `domain`, `mode`, `transport`, `status` |
| `agent_platform_a2a_calls_total` | `domain`, `endpoint`, `status` |
| `agent_platform_a2a_call_duration_seconds` | `domain`, `endpoint` |
| `agent_platform_handoffs_total` | `domain`, `target`, `transport` |

`GET /metrics` 暴露；`endpoint` 标签为 `host:port` 低基数形式。

## Supervisor（书稿 Chapter-6 原型）

```
用户请求 → Supervisor LLM 动态选择下一个 Agent → 可能循环多轮 → 结束
```

| 特点 | 说明 |
|------|------|
| 拓扑 | **运行时动态**，由 Supervisor 每步决定路由 |
| 规划 | 无全局 execution_plan，逐步决策 |
| 适用 | 工具链不确定、需要反复试探、人机协作切换 |

书稿参考：`Chapter-6/supervisor/`、`Chapter-7/A2A/`。

## 对比摘要

| 维度 | agent-platform (Fixed Graph) | Supervisor |
|------|------------------------------|------------|
| 图结构 | 固定，可预测 | 动态 |
| 可观测性 | 节点级 span 稳定 | 路径因请求而异 |
| 延迟 | 规划一次 + 分层执行 | 可能多轮 Supervisor LLM |
| 领域扩展 | DomainPlugin | 换 Agent 池 + Supervisor prompt |
| 误用风险 | 强行做「无限循环工具调用」 | 强行做「复杂 DAG 行程规划」 |

## 何时不要用本 SDK

- 需要 **纯跨服务** 编排且不用本平台 Supervisor → 直接用 Chapter-7 A2A 栈
- 需要 **运行时改图** 或 **人工审批插入任意节点** → 考虑 LangGraph 自定义图或 Temporal
- 单 Agent + 工具即可 → 直接用 LangChain Agent，无需本平台

## 何时适合本 SDK

- 多子 Agent **职责稳定**，中心负责拆任务与汇总
- 希望 **同一套编排** 复用到 travel / 客服 / 金融等多个领域
- 需要 **记忆、追踪、多租户、HTTP API** 等平台能力

## 版本与边界

- 包名：`agent-platform`
- 编排实现：`agent_framework/orchestration/fixed_graph/`
- 领域示例：`domains/travel`（书稿）、`domains/demo`（最小模板）

若 Supervisor 与 Fixed Graph 需要共存，建议在**应用层**按场景选不同入口，而非在同一 `LangGraphOrchestrator` 内混用两种范式。
