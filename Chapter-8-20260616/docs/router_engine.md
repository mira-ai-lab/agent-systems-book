# Router Engine（Phase 10–11）



企业通用路由引擎 L1：在编排执行前完成 **Agent classification（name + score）**、**多轮对话改写** 与 **子 Agent 指令构建**，并由 `profile=auto` 选择执行 Profile。



## 执行 Profile



| profile | 编排后端 | 说明 |

|---------|----------|------|

| `auto` | Router → 动态选择 | 企业推荐；先路由再执行 |

| `workflow` | Router → Fixed Graph | 强制 workflow；Router 预填 execution_plan |

| `adaptive` | Supervisor | 原 `mode=supervisor` |



### auto 决策规则（MVP）



- classification 结果中 **≥2 个 Agent score ≥ 0.5** → `workflow`（Fixed Graph）

- 否则 → `adaptive`（Supervisor）



## 路由阶段（Phase 11）



```

query + optional history

  → history_gate（0/1，不相关则丢弃 history）

  → interaction_rewrite（槽位补全 / 指代消解）

  → extraction（事件短语 → routing_plan.events）

  → classification（name + score）

  → instruction_build（adaptive 单主候选时，为 handoff 构建指令）

  → RoutingPlan.execution_query → 编排后端

```



| 阶段 | router-sdk 对齐 | 开关 |

|------|-----------------|------|

| `history_gate` | `flow_builder.history_prompt` | `RouterConfig.enable_history_gate` |

| `interaction_rewrite` | `response_handler.handle_interaction.prompt_rewrite` | `RouterConfig.enable_interaction_rewrite` |

| `instruction_build` | `response_handler.build_instruction.context_rebuild` | `RouterConfig.enable_instruction_build` |
| `extraction` | `selection.extraction` | `RouterConfig.enable_extraction` |



平台话术：



- Router 阶段：`agent_framework/router/prompts/locales/zh.json`

- 领域空字段回退：`agent_framework/prompts/locales/zh.json` → `DomainPrompts.with_platform_defaults()`



## API



```bash

curl -X POST http://127.0.0.1:8780/v1/chat \

  -H "Content-Type: application/json" \

  -d '{

    "domain": "customer_service",

    "profile": "auto",

    "query": "订单号是 12345",

    "conversation_history": "用户: 我要退货\n助手: 请提供订单号"

  }'

```



响应含 `routing_plan`（`rewritten_query`、`history_relevant`、`agent_instruction`、`execution_query` 等）与 `resolved_profile`。



## SDK



```python

from agent_framework.bootstrap.platform import create_runtime



runtime = create_runtime("customer_service", profile="auto")

result = await runtime.process_request(

    "订单号是 12345",

    conversation_history="用户: 我要退货\n助手: 请提供订单号",

)

print(result["routing_plan"])

```



## 架构



```

query [+ history] → RouterEngine.route() → RoutingPlan

      → profile=workflow  → RouterOrchestrator → LangGraphOrchestrator（prefilled plan）

      → profile=adaptive  → RouterOrchestrator → SupervisorOrchestrator（execution_query）

```

> **Phase 19**：`profile=workflow` 与 `profile=auto`（resolved=workflow）均经 `RouterOrchestrator`；  
> `allow_task_planner_decomposition=False`，无预填 plan 时 `build_plan` 拒绝 TaskPlanner fallback。



## Phase 11D — step_summary

Supervisor 每次 handoff 完成后，可按 `PipelineConfig.enable_step_summary` 对子 Agent 长回复做 step 级压缩（默认关闭，阈值 `step_summary_min_chars=200`）。

## Phase 13 — task.stage

多 step handoff 完成后，可按 `PipelineConfig.enable_stage_summary` 生成阶段级累计摘要（默认关闭，`stage_summary_min_steps=2`）。响应字段：`stage_summary`。

## Phase 14 — extraction

`RouterEngine` 在 classification 前抽取核心事件短语，写入 `routing_plan.events`。有历史时自动使用 `selection.extraction.multi` 模板。

## Phase 12 — 动态 Agent Registry

运行时向领域叠加 Agent 元数据或 A2A 端点；默认持久化到 `data/dynamic_agents.json`。

| 环境变量 | 说明 |
|----------|------|
| `DYNAMIC_AGENTS_PATH` | JSON 文件路径（默认 `data/dynamic_agents.json`） |
| `DYNAMIC_AGENTS_PERSIST=false` | 关闭持久化，仅内存 |

```bash
curl -X POST http://127.0.0.1:8780/v1/domains/demo/agents \
  -H "Content-Type: application/json" \
  -d '{"name":"RemoteFAQ","description":"远程 FAQ","source":"a2a","a2a_url":"http://127.0.0.1:9100/"}'
```

## 后续

- 跨租户 stage summary 隔离策略
- 知识路由与向量检索联动

## Phase 15 — thread 累积 + knowledge 路由

### Thread `last_stage_summary`

- 持久化：`data/thread_stage_context.json`（`THREAD_STAGE_CONTEXT_PATH` / `THREAD_STAGE_CONTEXT_PERSIST`）
- Supervisor 请求注入【先前阶段累计进度】；stage 压缩后写回
- `RouterOrchestrator` 将其作为 `instruction_build.previous_step_info`

### Knowledge 路由

- classification catalog 展开 `知识支持` skill 的 tags/keywords
- `knowledge_routing` 阶段对 query/events 做关键词匹配并合并 classification 分数

## Phase 16 — 跨域推断 + task_decomposition + hybrid

### 跨域推断（只传 query）

- `POST /v1/chat` 的 `domain` 可选
- 优先级：显式 `domain` > `DEFAULT_DOMAIN` > LLM 跨域 classification（`platform_domain_router.py`）
- 响应含 `resolved_domain` / `domain_candidates`（自动推断时）

### Task decomposition

- `RouterEngine` 在 `profile=workflow` 时运行 `task_decomposition`
- 产出写入 `RoutingPlan.steps`（`step_id` / `description` / `agent` / `depends_on`）
- 开关：`RouterConfig.enable_task_decomposition`（默认开启）

### Phase 27 — travel 语义路由（semantic_routing）

- **`domain=travel`** 且 **`RouterConfig.semantic_task_routing=true`**（默认）时：
  1. 使用领域 **`decomposition_prompt`** 拆解（替代通用 Router 拆解话术）
  2. 调用 TaskPlanner **`run_dependency_analysis` + `route_to_agents`**（`agent_routing` prompt）
  3. 产出带语义 Agent 与 **`depends_on`** 的 `RoutingStep`，预填 Fixed Graph `execution_plan`
- 阶段名：`semantic_routing`（出现在 `metadata.stages` 与 SSE `router.semantic_routing`）
- 关闭：`RouterConfig(semantic_task_routing=False)` 回退为 classification 分数顺位绑 Agent
- 书稿全链路（含 pre_survey LLM）：仍可用 CLI **`--legacy-graph`**

### Hybrid Profile

- `profile=hybrid` → Supervisor 编排，默认 `transport=mixed`（local + A2A 混部）
- 适用于 travel 等配置了 A2A 端点的领域

### 产品域

- `travel`、`customer_service` 均为正式产品域（`is_sample=false`）
- 推荐：`profile=auto`；复杂行程可用 `domain=travel`

## Phase 17 — en locale + Router 预填 execution_plan

### English locale

- Router：`agent_framework/router/prompts/locales/en.json`
- 领域默认话术：`agent_framework/prompts/locales/en.json`
- 用法：`create_runtime(..., locale="en")` 或 `RouterOrchestrator(..., locale="en")`
- 缺失 key 时 loader 回落 `zh`

### Router → FixedGraph 计划预填

- `profile=auto` 且 resolved_profile=`workflow` 时，`RoutingPlan.steps` 转为 `execution_plan`
- FixedGraph `build_plan` 检测到 `prefilled_execution_plan` 后跳过 TaskPlanner LLM 拆解
- `execution_plan.source = "router_engine"`

```python
from agent_framework.router.execution_plan_bridge import execution_plan_from_routing_plan
prefilled = execution_plan_from_routing_plan(plan, user_query=query)
await orchestrator.process_request(query, prefilled_execution_plan=prefilled)
```

## Phase 18 — API locale + pre_survey 联动 + 领域 en

### API `locale`

```bash
curl -X POST http://127.0.0.1:8780/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"What is your return policy?","domain":"customer_service","locale":"en"}'
```

- 请求/响应字段：`locale`（默认 `zh`）
- 租户池按 `(domain, profile, mode, transport, locale, user_id)` 缓存 runtime

### Router → pre_survey

- `pre_survey_from_routing_plan()` 将 `routing_plan.events`、候选 Agent、stages 写入四段式 `pre_survey`
- **`PipelineConfig.pre_survey_mode`**（Phase 28）：
  - `router_prefill`（默认）：有 Router 预填时跳过 Ch2 LLM
  - `full_ch2`：Fixed Graph 仍跑 `run_pre_survey()`，并回写 `execution_plan`（**travel 插件默认**）
  - `off`：不运行 pre_survey 节点
- travel **semantic_routing** 拆解时注入 Router bridge `pre_survey`（不再传空 `{}`）
- `execution_plan` 含 `pre_survey_mode` / `pre_survey_source` 便于 trace 观测

### 领域 prompt 英文化

- `domains/customer_service/prompts_en.py`
- `domains/travel/prompts_en.py`
- `CustomerServicePrompts.build(locale="en")` / `TravelPrompts.build(locale="en")`
- `demo` 插件内置 en 话术

## Phase 19 — workflow 与 Router 完全合一

### 统一入口

- `create_runtime(..., profile="workflow")` 与 `create_orchestrator(domain)` 均返回 **`RouterOrchestrator`**
- 不再从平台工厂直连 `LangGraphOrchestrator`；Fixed Graph 作为 Router 内部执行后端

### Router 预填 + 禁止二次拆解

- `ensure_execution_plan_from_routing_plan()`：有 `steps` 时转换，否则合成单步 plan（主候选 Agent）
- `RouterOrchestrator._router_unified_pipeline()` 设置 `allow_task_planner_decomposition=False`
- `build_plan` 无 `prefilled_execution_plan` 且禁止拆解时 **raise**，避免静默回落 TaskPlanner

### workflow 强制路由

- `entry_profile=workflow` 时 `RouterEngine.route(..., force_profile="workflow")`
- 单 Agent 场景也会运行 `task_decomposition`（不再要求多 candidate）

```python
from agent_framework.bootstrap.platform import create_orchestrator

orch = create_orchestrator("customer_service")  # RouterOrchestrator, entry_profile=workflow
result = await orch.process_request("退货政策是什么？")
# result["routing_plan"], result["prefilled via backend kwargs"]
```

## Phase 20 — SSE + Router stage events + FixedGraph 进度流

### 事件 schema（MVP）

| type | 说明 |
|------|------|
| `router.extraction` | `data.events` 为 extraction 短语列表 |
| `router.classification` | `data.candidates` 为 Agent 候选 |
| `router.task_decomposition` | `data.steps` 为拆解步骤 |
| `router.plan` | 完整 `routing_plan` |
| `graph.progress` | FixedGraph 阶段进度文本 |
| `graph.node` | LangGraph 节点更新摘要 |
| `graph.subtask.token` | 子 Agent LLM 逐 token（`task_id` / `agent` / `token`） |
| `graph.subtask.completed` | 子 Agent 完成（已流式时 CLI 仅打 `done`；否则含 `summary`） |
| `graph.token` | 聚合阶段 token（可选） |
| `final` | 与 `/v1/chat` 等价的结果 payload |
| `error` | 流内错误（超时 / 500 等） |

### SDK

```python
from agent_framework.stream.events import public_event

async for event in runtime.iter_request_stream("退货政策是什么？", thread_id="t1"):
    print(public_event(event)["type"])
```

### HTTP SSE

```bash
curl -N -X POST http://127.0.0.1:8780/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"退货政策是什么？","domain":"customer_service","profile":"auto"}'
```

响应 `Content-Type: text/event-stream`；每条事件形如：

```
id: 1
event: router.extraction
data: {"type":"router.extraction","stage":"extraction","data":{"events":["咨询退货"]}}
```

## Phase 21 — 向量知识库 knowledge_routing

- 文档路径：`domains/{domain}/knowledge/documents.json`（`id` / `agent` / `text` / `tags`）
- `RouterConfig.knowledge_backend`：`keyword` | `vector` | `hybrid`（默认 hybrid）
- `knowledge_vector_min_score`（默认 0.15）与 keyword 阈值（0.65）分离
- `metadata.knowledge_matches[].source`：`keyword` | `vector`

## Phase 22 — Agent Registry 产品层

```bash
curl http://127.0.0.1:8780/v1/agents
curl -X POST .../v1/domains/demo/agents -d '{"name":"GlobalFAQ","scope":"shared","skills":[...]}'
```

- `scope=shared` 的 Agent 写入 `__shared__`，对所有 domain runtime 可见
- `GET /v1/agents` 汇总 static + dynamic + shared

## Phase 23 — 领域 prompt locales

- 路径：`domains/{domain}/locales/zh.json` / `en.json`
- 加载：`domain_prompts_from_locale(domain, locale)` 或 `TravelPrompts.build(locale=...)`
- 导出：`python scripts/export_domain_locales.py`

## Phase 24 — route(query) + KB Embedding 后端

### 统一 SDK 入口

```python
from agent_framework.bootstrap import route

result = await route("退货政策是什么？", domain="customer_service")
# result["domain"], result["routing_plan"], result["final_response"]
```

### Embedding 后端

| 值 | 说明 |
|----|------|
| `hashing` | 默认；jieba + hashing，无 API |
| `embedding` | DashScope/OpenAI 兼容 `text-embedding-v3` |

```python
RouterConfig(knowledge_embedding_backend="embedding")
# 或环境变量 KNOWLEDGE_EMBEDDING_BACKEND=embedding
```

### HTTP 默认路径（24.2）

- `POST /v1/chat`：**仅 `query` 必填**；`profile` 默认 `auto`；`domain` 可省略
- `GET /v1/domains`：顶层 `recommended_profile: "auto"`，各 domain 条目同字段

### 可观测字段（24.3）

SDK / HTTP 响应 / SSE `final` 事件统一顶层字段：

| 字段 | 说明 |
|------|------|
| `resolved_domain` | 实际执行领域 |
| `resolved_profile` | Router 选定的 workflow / adaptive |
| `routing_plan` | 完整 L1 路由计划 |
| `knowledge_matches` | 从 `routing_plan.metadata` 提升，便于 Debug 面板 |

实现：`agent_framework/router/observability.py` → `enrich_routing_observability()`

### Demo 分层（24.4）

- **产品默认**：`python scripts/run_demo.py` → `customer_service` + `profile=auto`
- **能力展示**：`python scripts/run_demo.py --domain travel --profile workflow --show-graph`
- **MCP**：`travel_agent_mcp_server.py` 为 travel 书稿域；产品路径见 `run_demo.py`

### Chroma 持久化 KB（24.6）

- 路径：`data/knowledge/{domain}/documents.json` + `chroma/`（与 `chroma_memory/` 隔离）
- Ingest：`python scripts/ingest_knowledge.py --domain customer_service` 或 `--all`
- `RouterConfig.knowledge_storage`：`auto`（有 persisted 则 chroma）| `memory` | `chroma`

### KB 管理 API（24.7）

```bash
GET  /v1/domains/customer_service/knowledge
POST /v1/domains/customer_service/knowledge
# body: { "documents": [...], "replace": false, "embedding_backend": "hashing" }
```

写入后自动 sync Chroma 并 `tenant_pool.invalidate(domain)`。

### hybrid 分数归一化（24.8）

| source | raw 范围 | 归一化 |
|--------|----------|--------|
| `keyword` | 0.65–1.0（命中阈值） | `(raw - min) / (1 - min)` |
| `vector` | cosine ~0.15–1.0 | `(raw - vector_min) / (1 - vector_min)` |

`knowledge_matches[]` 同时含 `raw_score`、`normalized_score`；合并候选使用 `normalized_score`。

### Registry 产品化（24.9–24.12）

#### 跨域 catalog（24.9）

- `build_domain_catalog()`（`agent_catalog.py`）汇总各 domain 静态 + 动态 Agent，并追加 `[跨域共享 Agent]` 段
- LLM 跨域推断（`resolve_request_domain`）使用该 catalog

#### Agent 别名（24.10）

- `DynamicAgentRecord.alias_of`：shared/metadata Agent 引用静态 Agent（如 `FAQAgent`）
- `merge_dynamic_agents()` 继承被引用 Agent 的 `skills` / `description`

#### Registry 变更事件（24.11）

- 事件类型：`registry.updated`（`stream/events.py`）
- API 注册/注销动态 Agent 响应含 `registry_event`
- 可选 webhook：环境变量 `REGISTRY_WEBHOOK_URL`

#### Agent 目录 API（24.12）

```bash
GET /v1/agents?domain=customer_service&scope=shared&source=dynamic
```

### i18n parity（24.13–24.16）

- 子 Agent：`domains/{domain}/agents/locales/{zh,en}.json`
- 加载：`agent_system_prompt(domain, agent_name, locale)`；运行时 `agent_locale_context(locale)`
- 缺失 key：`locale.missing_keys` structured log + fallback `zh`
- `prompts_en.py` 已 deprecated，请使用 `locales/en.json`

### 编排可解释 + travel 叙事（24.17–24.20）

- `routing_plan.metadata.profile_reason`：如 `strong_agents=2>=0.5: FAQAgent=0.90, TicketAgent=0.85`
- adaptive 流式：`SupervisorOrchestrator.iter_request_stream` 产出 `handoff.completed` 事件
- 领域定位：见 [domains.md](domains.md)
- 发版回归：`python scripts/product_readiness_check.py`

### 多租户 KB（25.1–25.4）

- Shared：`data/knowledge/{domain}/`；Tenant overlay：`tenants/{user_id}/`
- 路由检索：`routing_plan.metadata.kb_tenant_id` = 请求 `user_id`
- API：`GET/POST /v1/domains/{domain}/knowledge?user_id=alice`
- 环境变量：`KNOWLEDGE_TENANT_ISOLATION=true`（默认开启）

### Registry 联邦（25.5–25.8）

- 环境变量：`REGISTRY_FEDERATION_URLS=http://peer-a:8780,http://peer-b:8780`
- 可选：`REGISTRY_FEDERATION_API_KEY` 访问远程 `/v1/agents`
- 联邦目录：`GET /v1/agents?federated=true&health=true`
- Cluster 状态：`GET /v1/registry/federation`
- 远程 Agent 条目含 `origin=federated`、`federation_cluster`；A2A 动态 Agent 可附 `a2a_health`

### Embedding 召回评测（25.9–25.10）

- Fixture：`data/knowledge/benchmark/fixtures.json`
- 脚本：`python scripts/benchmark_knowledge_recall.py --backends hashing,embedding --json`
- 报告字段：`hit@1/3/5`、`raw_score`、`normalized_score`、A/B `comparison`

### 前端 SDK（25.11–25.12）

- 包路径：`packages/router-client/`（`@agent-platform/router-client`）
- 同步：`client.route(query, { profile: 'auto' })` → `POST /v1/chat`
- 流式：`for await (const event of client.routeStream(query))` → SSE
- 构建：`cd packages/router-client && npm install && npm test`

### SDK 联调（26.1）

自动化（pytest 内嵌 uvicorn + Node）：

```bash
pytest tests/test_phase26_router_client_integration.py -q
```

对已运行 API 手动冒烟：

```bash
python scripts/run_api.py          # 终端 1
python scripts/smoke_router_client.py
# 或
cd packages/router-client
ROUTER_CLIENT_BASE_URL=http://127.0.0.1:8780 npm run test:integration
```

### Demo Web（26.5）

```bash
python scripts/run_api.py              # 终端 1 :8780
cd packages/demo-web && npm install && npm run dev   # 终端 2 :5173
```

Vite dev 将 `/v1/*` 代理到 API；页面可切换同步 / SSE 并查看 Router 时间线。详见 [packages/demo-web/README.md](../packages/demo-web/README.md)。

完整联调流程见 [sdk_integration.md](sdk_integration.md)（Phase 26.6）。

### Jobs SDK（26.7）

```typescript
const job = await client.submitJob("规划七日游", { domain: "travel" });
const status = await client.getJob(job.job_id);
```

### 重试 / 超时（26.8）

- 默认 `fetchPolicy`：120s 超时、2 次重试、500ms 起指数退避
- 自动重试 429/502/503/504 与网络/超时；4xx 业务错误不重试
- 错误类型：`RouterClientError` / `RouterClientTimeoutError` / `RouterClientNetworkError`

### 版本同步（26.4）

```bash
python scripts/sync_package_versions.py --check
python scripts/sync_package_versions.py --sync 0.22.0
```

`agent-platform`、`router-client`、`demo-web` 共用 semver；详见 [sdk_integration.md](sdk_integration.md#发版与版本同步264)。
