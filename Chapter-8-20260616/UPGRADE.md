# Chapter-8 架构升级（20260616 分支）



本目录为 **生产化改造工作区**。



| 目录 | 用途 |

|------|------|

| `Chapter-8/` | 稳定主线 |

| `Chapter-8-backup-20260616/` | 只读快照 |

| `Chapter-8-20260616/` | **本目录** |



## 升级阶段



### Phase 1 — 基础生产化（已完成）



- [x] 旅行 demo 工厂、`configure_agent_llm`、请求超时、HTTP API



### Phase 2 — 稳定性（已完成）



- [x] LLM/HTTP 重试、并发限流、`depends_on` 解析、checkpoint 可配置、memory namespace



### Phase 3 — 规模化（已完成）



- [x] **Docker** + `docker-compose.yml`

- [x] **GitHub Actions CI**（`.github/workflows/chapter8-upgrade-ci.yml`）

- [x] **多租户** `user_id` → `TenantOrchestratorPool` + 记忆隔离

- [x] **异步任务队列** `POST /v1/jobs` + `GET /v1/jobs/{id}` + SQLite `JobStore`

- [x] `travel_api` 统一接入 `async_http_request`



### Phase 4 — 领域插件化 / 通用 SDK（已完成）



- [x] **`DomainPlugin` 协议**（`agent_framework/domain/plugin.py`）

- [x] **插件注册表** `register_domain` / `get_domain_plugin` / `list_domains`

- [x] **框架零 travel 默认回落**：`LangGraphOrchestrator` / `GraphContext` / `SubAgentFactory` 必须显式注入或通过 `domain=` 解析

- [x] **`configure_agent_llm` / `build_agent`** 迁至 `agent_framework/infra/agent_runtime.py`

- [x] **平台入口** `create_orchestrator(domain)`（`agent_framework/bootstrap/platform.py`）

- [x] **内置插件**：`domains/travel/plugin.py`、`domains/customer_service/plugin.py`（客服示例）

- [x] **多领域租户池** `(domain, user_id)` LRU 缓存

- [x] **HTTP API**：`domain` 请求字段、`GET /v1/domains`、Job 携带 `domain`

- [x] 版本 **0.4.0**



### Phase 5 — 平台品牌化（已完成）



- [x] PyPI 包名 **`agent-platform`** v0.5.0（原 `travel-multi-agent` 弃用）

- [x] OTEL 默认服务名 **`multi-agent-platform`**

- [x] API **`domain` 必填**（无默认回落；可选 `DEFAULT_DOMAIN` 环境变量兼容旧客户端）

- [x] `/ready` 仅检查插件注册表；可选 `READY_DOMAIN` 预热

- [x] `travel` 标注为**书稿示例**；MCP 服务名改为 `travel-example`

- [x] Docker 服务名 `platform-api`



### Phase 6 — 正式对外 SDK（已完成）



- [x] **拆包**：`agent-platform`（`agent_framework` + `services`）与 `agent-platform-domains-builtin`（`domains/pyproject.toml`）

- [x] **entry_points** 插件发现：`agent_platform.domains` 组，无框架硬编码 import

- [x] **Plugin 开发文档** [`docs/plugin_development.md`](docs/plugin_development.md) + `domains/demo` 最小模板

- [x] **API 鉴权**：`API_KEYS` + `X-API-Key` 头（未配置则放行）

- [x] **Prometheus**：`GET /metrics`（`prometheus-client`，`[api]` extra）

- [x] **编排边界文档** [`docs/orchestration_model.md`](docs/orchestration_model.md)（Fixed Graph vs Supervisor）

- [x] 版本 **0.6.0**；开发安装：`pip install -e ".[api,dev]" && pip install -e domains/`



### Phase 7A/7B — 多编排范式：Supervisor（已完成）



- [x] **`OrchestrationBackend` 协议**（`agent_framework/orchestration/protocol.py`）

- [x] **`create_runtime(domain, mode=...)`** — `fixed_graph`（默认）| `supervisor`

- [x] **Supervisor 后端**（`agent_framework/orchestration/supervisor/`，基于 `langgraph-supervisor`）

- [x] **`DomainPlugin.supported_modes`** + `DomainPrompts.supervisor_system`

- [x] 内置领域均支持双模式：`travel` / `customer_service` / `demo`

- [x] API / Job / 租户池缓存键增加 **`mode`**

- [x] 可选依赖：`pip install "agent-platform[supervisor]"`

- [x] 版本 **0.7.0**



### Phase 7C — A2A Transport（Supervisor 远程 handoff）



- [x] **`AgentTransport`**：`local` | `a2a` | `mixed`（仅 `mode=supervisor`）

- [x] **`A2AEndpoint`** + `agent_framework/transport/a2a/`（复用 Chapter-7 A2A 协议）

- [x] **`create_runtime(..., transport=...)`** + 租户池缓存键含 `transport`

- [x] **`DomainPlugin.create_a2a_endpoints`** / `resolved_a2a_endpoints()`；`travel` 支持 `TRAVEL_A2A_HOTEL_URL`

- [x] API / Job 请求体增加 **`transport`**（默认 `local`）

- [x] 可选依赖：`pip install "agent-platform[a2a]"`

- [x] 版本 **0.7.1**



### Phase 8 — 可观测性：Supervisor / A2A Tracing（已完成）



- [x] **`agent.invoke`** span：Supervisor 本地子 Agent（`invoke_traced.py`）

- [x] **`a2a.call`** span + W3C **traceparent inject**（`call_traced.py`）

- [x] **`handoff.completed`** event + 根 span `orchestration.mode` / `agent.transport`

- [x] **`a2a.error`** / **`sub_agent_conversation`** events

- [x] **`inject_trace_context` / `extract_trace_context`** 工具函数

- [x] 版本 **0.8.0**（tracing 部分；Prometheus 维度扩展留待 0.8.1）



### Phase 8B — Prometheus：mode / transport / a2a_*（已完成）



- [x] **`agent_framework/observability/metrics.py`** — 框架层指标（API 与编排共用）

- [x] **`agent_platform_chat_requests_total{domain,mode,transport,status}`**

- [x] **`agent_platform_job_requests_total{domain,mode,transport}`** + **`job_outcomes_total`**

- [x] **`agent_platform_a2a_calls_total`** + **`a2a_call_duration_seconds`**

- [x] **`agent_platform_handoffs_total{domain,target,transport}`**

- [x] **`request_metrics_context`** — 请求级 domain/mode/transport 上下文

- [x] 版本 **0.8.1**



### Phase 10 — Router Engine + profile=auto（已完成）



- [x] **`RoutingPlan`** / **`AgentCandidate`**（`agent_framework/router/plan.py`）

- [x] **`RouterEngine`** + **classification** 阶段（`router/stages/classification.py`）

- [x] **`RouterOrchestrator`**（`profile=auto`：路由 → workflow / adaptive 委托执行）

- [x] **执行 Profile**：`auto` | `workflow` | `adaptive`（`create_runtime(profile=...)`）

- [x] 平台 locale：`agent_framework/router/prompts/locales/zh.json`

- [x] API：`POST /v1/chat` 增加 `profile`；响应含 `routing_plan`

- [x] 文档 [`docs/router_engine.md`](docs/router_engine.md)

- [x] 版本 **0.9.0**



### Phase 11 — 多轮改写 + InstructionBuilder + 平台 locale（已完成）



- [x] **`history_gate`**（`router/stages/history_gate.py`）— 对话历史相关性 0/1

- [x] **`interaction_rewrite`**（`router/stages/interaction_rewrite.py`）— 多轮槽位补全

- [x] **`InstructionBuilder`**（`router/instruction_builder.py`）— adaptive 主候选 handoff 前指令构建

- [x] **`RouterEngine.route(history=...)`** 串联上述阶段；`RoutingPlan.execution_query`

- [x] 平台 locale 扩展：`router/prompts/locales/zh.json` + `prompts/locales/zh.json`

- [x] **`DomainPrompts.with_platform_defaults()`** 空字段回退

- [x] API：`conversation_history` 可选字段（`profile=auto`）

- [x] 文档 [`docs/router_engine.md`](docs/router_engine.md)

- [x] 版本 **0.10.0**



### Phase 11D — step_summary（Supervisor 步骤压缩）（已完成）



- [x] **`StepSummarizer`**（`orchestration/supervisor/step_summary.py`）

- [x] 话术 `task.summary` → `router/prompts/locales/zh.json`

- [x] **`PipelineConfig.enable_step_summary`** / `step_summary_min_chars`（默认关闭）

- [x] Supervisor 执行后压缩 `subtask_results.agent_summary`



### Phase 12 — 动态 Agent Registry（已完成）



- [x] **`DynamicAgentStore`** + `merge_dynamic_agents`（`domain/dynamic_registry.py`）

- [x] **`SubAgentRegistry.register_metadata`** / `unregister` / `is_metadata_only`

- [x] **`resolve_domain_registry_and_a2a`** — `create_runtime` 自动合并动态 Agent

- [x] **`TenantOrchestratorPool.invalidate(domain)`** — 注册变更后失效缓存

- [x] API：`GET/POST/DELETE /v1/domains/{domain}/agents`

- [x] **JSON 持久化**：`PersistedDynamicAgentStore`（`data/dynamic_agents.json`，`DYNAMIC_AGENTS_PERSIST` / `DYNAMIC_AGENTS_PATH`）

- [x] 版本 **0.11.0** → **0.12.0**（含 Phase 13/14）



### Phase 13 — task.stage 阶段级压缩（已完成）



- [x] **`StageSummarizer`**（`orchestration/supervisor/stage_summary.py`）

- [x] 话术 `task.stage` → `router/prompts/locales/zh.json`

- [x] **`PipelineConfig.enable_stage_summary`** / `stage_summary_min_steps`（默认关闭）

- [x] Supervisor 响应含 `stage_summary`



### Phase 14 — selection.extraction 事件抽取（已完成）



- [x] **`run_extraction`**（`router/stages/extraction.py`）

- [x] **`RouterConfig.enable_extraction`**（默认开启）

- [x] `RoutingPlan.events` 写入 API `routing_plan.events`

- [x] 版本 **0.12.0**



### Phase 15 — thread 累积 + knowledge 路由（已完成）



- [x] **`ThreadStageContextStore`** + JSON 持久化（`thread_stage_context.py`）

- [x] Supervisor / Router 跨请求 `last_stage_summary` 注入与写回

- [x] **`knowledge_routing`** + classification catalog 知识点展开

- [x] API 默认 `profile=auto`

- [x] 版本 **0.13.0**



### Phase 16 — 产品域 + 跨域推断 + task_decomposition + hybrid（已完成）



- [x] **`travel`** 作为正式产品域（`is_sample=False`）

- [x] **`PlatformDomainRouter`**：`POST /v1/chat` 可省略 `domain`，LLM 跨域推断

- [x] **`task_decomposition`** 并入 `RouterEngine`（workflow 时写入 `RoutingPlan.steps`）

- [x] 显式 **`profile=hybrid`** → Supervisor + 默认 `transport=mixed`

- [x] 版本 **0.14.0**



### Phase 17 — en locale + Router 预填 execution_plan（已完成）



- [x] **`agent_framework/router/prompts/locales/en.json`** + **`prompts/locales/en.json`**

- [x] locale loader 缺失时回落 `zh`

- [x] **`execution_plan_bridge`**：`RoutingPlan.steps` → `prefilled_execution_plan`

- [x] FixedGraph `build_plan` 跳过 TaskPlanner（Router workflow 路径）

- [x] 版本 **0.15.0**



### Phase 18 — API locale + pre_survey 联动 + 领域 prompt en（已完成）



- [x] **`POST /v1/chat`** / **`POST /v1/jobs`** 增加 `locale`（`zh` \| `en`）

- [x] **`pre_survey_bridge`**：Router `events` → FixedGraph `pre_survey`；预填时跳过 LLM 预调查

- [x] **`domains/*/prompts_en.py`** + `prompt_bundle.build(locale=...)`

- [x] 租户池 cache key 含 `locale`

- [x] 版本 **0.16.0**



### Phase 19 — README 产品化 + workflow 与 Router 完全合一（已完成）



- [x] **README 产品化**：travel / customer_service 均为产品域；推荐 `profile=auto`；API `domain` 可选

- [x] **`profile=workflow`** 统一走 `RouterOrchestrator`（不再直连 `LangGraphOrchestrator`）

- [x] **`create_orchestrator`** 返回 `RouterOrchestrator`（workflow 别名）

- [x] **`PipelineConfig.allow_task_planner_decomposition=False`**：Router 路径禁止 TaskPlanner fallback

- [x] **`ensure_execution_plan_from_routing_plan`**：无 steps 时合成单步 plan

- [x] **`RouterEngine.force_profile=workflow`**：workflow 入口强制拆解阶段

- [x] 版本 **0.17.0**



### Phase 20 — SSE + Router stage events + FixedGraph 进度流（已完成）



- [x] **`RouterEngine.route_stream()`**：逐阶段 yield `router.*` 事件（extraction / classification / task_decomposition / plan）

- [x] **`RouterOrchestrator.iter_request_stream()`**：Router 事件 → FixedGraph `graph.node` / `graph.progress` / `graph.token` → `final`

- [x] **`LangGraphOrchestrator.iter_request_stream()`**：结构化 graph 事件 + StreamSink 进度/token

- [x] **`POST /v1/chat/stream`**：`text/event-stream`（SSE），前端可逐条消费

- [x] **`agent_framework/stream/`**：事件 schema + `format_sse()`

- [x] 版本 **0.18.0**



### Phase 21 — 向量知识库 + knowledge_routing hybrid（已完成）



- [x] **`domains/{domain}/knowledge/documents.json`** + `DomainKnowledgeStore`（jieba + hashing embedding）

- [x] **`RouterConfig.knowledge_backend`**：`keyword` \| `vector` \| `hybrid`

- [x] **`resolve_knowledge_candidates()`**：合并 keyword + vector；`metadata.knowledge_matches` 含 `source`

- [x] **`RouterEngine(domain=...)`** 传入领域名以加载 KB

- [x] 版本 **0.19.0**（与 Phase 22/23 同批）



### Phase 22 — Agent Registry 产品层 + 跨 domain shared（已完成）



- [x] **`GET /v1/agents`**：平台级 Agent 目录（static + dynamic + shared）

- [x] **`DynamicAgentRecord.scope=shared`** → `__shared__` 桶，合并进所有 domain registry

- [x] **`agent_catalog.py`**：`list_platform_agent_entries()` / `summarize_domain_agents()`

- [x] API `POST /v1/domains/{domain}/agents` 支持 `scope` + `skills`

- [x] 跨域推断 catalog 含 dynamic/shared Agent 摘要



### Phase 23 — 领域 prompt locales JSON（已完成）



- [x] **`domains/*/locales/{zh,en}.json`**（travel / customer_service / demo）

- [x] **`domain_prompts_from_locale()`** + `prompt_bundle.build()` 统一读 JSON

- [x] **`scripts/export_domain_locales.py`** 从 legacy prompts.py 导出



### Phase 24 — route(query) + KB Embedding 后端（P0 进行中）



- [x] **`await route(query)`** 统一 SDK 入口（`agent_framework/bootstrap/entry.py`）

- [x] **`KnowledgeEmbeddingBackend`**：`hashing`（默认）| `embedding`（DashScope/OpenAI 兼容）

- [x] **`RouterConfig.knowledge_embedding_backend`** + `KNOWLEDGE_EMBEDDING_BACKEND` 环境变量

- [x] 向量命中 metadata 含 `embedding_backend` / `raw_score`

- [x] **HTTP 默认路径对齐**：`POST /v1/chat` 仅 `query`；OpenAPI examples；`GET /v1/domains` → `recommended_profile=auto`

- [x] **routing 可观测字段**：`resolved_domain` / `resolved_profile` / `routing_plan` / `knowledge_matches`（SDK + HTTP + SSE）

- [x] **`run_demo.py`** 默认 `customer_service` + `profile=auto`；travel MCP 标注为能力展示域

- [x] **Chroma 持久化 KB**：`data/knowledge/{domain}/` + `scripts/ingest_knowledge.py`

- [x] **KB 管理 API**：`GET/POST /v1/domains/{domain}/knowledge`（热更新 + `tenant_pool.invalidate`）

- [x] **hybrid 分数归一化**：`knowledge_matches` 含 `raw_score` + `normalized_score`（统一 `[0,1]`）

### Phase 24 P1 — Registry 产品化（24.9–24.12）



- [x] **`build_domain_catalog()`** 含 dynamic + shared + alias 摘要

- [x] **`DynamicAgentRecord.alias_of`** + `registry_agent` 跨域引用静态 Agent 能力

- [x] **`registry.updated`** SSE 事件 schema + `REGISTRY_WEBHOOK_URL` 占位

- [x] **`GET /v1/agents`** 支持 `?domain=`、`?scope=`、`?source=`

### Phase 24 P1 — i18n parity（24.13–24.16）



- [x] **`domains/*/agents/locales/{zh,en}.json`**（travel 五 Agent + CS 两 Agent）

- [x] **`agent_system_prompt()`** + `agent_locale_context` 请求级 locale

- [x] **locale 缺失 structured log** + fallback zh（`locale.missing_keys`）

- [x] **`prompts_en.py`** 标记 deprecated，re-export 自 `locales/en.json`

- [x] **`tests/test_phase24_i18n.py`** 三 domain + API locale 端到端

- [x] 版本 **0.20.0**

### Phase 24 P2 — 编排与 travel 叙事收尾（24.17–24.20）



- [x] **`routing_plan.metadata.profile_reason`** — auto/forced profile 可解释

- [x] **Supervisor `iter_request_stream`** — yield `handoff.completed` + `final`

- [x] **`docs/domains.md`** — travel=能力展示，CS=默认产品域

- [x] **`scripts/product_readiness_check.py`** — 六维度 % 自检

- [x] **`tests/test_phase24_p2.py`**

### Phase 25 — 生产深度（延后项，进行中）

**目标**：多租户 KB、Registry 联邦、Embedding 评测、前端 SDK。

#### P0 — 多租户 KB（25.1–25.4）

- [x] **`data/knowledge/{domain}/tenants/{user_id}/`** overlay + shared 合并
- [x] **`KNOWLEDGE_TENANT_ISOLATION`** 环境变量（默认开启）
- [x] **KB API / Router** 传递 `user_id` / `kb_tenant_id`
- [x] **Chroma 增量 upsert**（`replace=false` 按 doc_id）
- [x] **`tests/test_phase25_kb_tenant.py`**

#### P1 — Registry 联邦（25.5–25.8）

- [x] **`REGISTRY_FEDERATION_URLS`** + `REGISTRY_FEDERATION_API_KEY` 拉取远程 `/v1/agents`
- [x] **`list_federated_agents()`** 合并 local + federated catalog
- [x] **`GET /v1/agents?federated=true`**（可选 `&health=true` A2A 探测）
- [x] **`GET /v1/registry/federation`** cluster 健康占位
- [x] **`tests/test_phase25_registry_federation.py`**

#### P2 — Embedding 评测（25.9–25.10）

- [x] **`data/knowledge/benchmark/fixtures.json`** 评测 query + expected doc/agent
- [x] **`agent_framework/router/kb/benchmark.py`** hit@k + raw/normalized score 报告
- [x] **`scripts/benchmark_knowledge_recall.py`** hashing vs embedding A/B
- [x] **`tests/test_phase25_kb_benchmark.py`**

#### P3 — 前端 SDK（25.11–25.12）

- [x] **`packages/router-client/`** — `@agent-platform/router-client`
- [x] **`route()` + `routeStream()`** 对齐 `POST /v1/chat` / `/v1/chat/stream`
- [x] **SSE parser** + Node unit tests
- [x] **npm 包占位 README** + `tests/test_phase25_router_client.py`

- [x] **Phase 25 全部完成**（版本 **0.21.0**）

### Phase 26 — SDK 产品化与 Demo（已完成）

**目标**：router-client 联调验证、CI、最小 Demo UI、异步 jobs SDK、semver 同步。

#### P0 — SDK 联调与 CI（26.1–26.4）

- [x] **`packages/router-client/tests/integration.test.mjs`** — `route()` + `routeStream()` 对 live API
- [x] **`tests/test_phase26_router_client_integration.py`** — uvicorn 临时服务 + Node 联调
- [x] **`scripts/smoke_router_client.py`** — 对已运行 API 的手动冒烟
- [x] **GitHub Actions**：`packages/router-client` npm test + integration（26.3）
- [x] **`scripts/sync_package_versions.py`** — semver 校验 / 同步（26.4）
- [x] **CI `--check`** + `tests/test_phase26_version_sync.py`

#### P1 — Demo Web UI（26.5–26.6）

- [x] **`packages/demo-web/`** — Vite + router-client 最小 Chat + SSE 时间线（26.5）
- [x] **`docs/sdk_integration.md`** — 本地联调三步（API + smoke + demo）（26.6）

#### P2 — SDK 能力补齐（26.7–26.8）

- [x] **`submitJob()` / `getJob()`** 对齐 `POST/GET /v1/jobs`（26.7）
- [x] **重试 / 超时 / 错误类型** 统一封装（26.8）

- [x] **Phase 26 全部完成**（版本 **0.21.0**）

### Phase 27 — travel 语义路由 + 运维/安全文档（已完成）

**目标**：Router `profile=workflow` + `domain=travel` 与书稿 Ch4 TaskPlanner 对齐；补齐运维/安全文档。

#### P0 — Router travel 语义拆解（27.1–27.3）

- [x] **`agent_framework/router/stages/semantic_routing.py`** — 领域 `decomposition_prompt` + `agent_routing`（依赖分析保留 `depends_on`）
- [x] **`RouterConfig.semantic_task_routing`**（默认 `true`）；`SEMANTIC_ROUTING_DOMAINS={travel}`
- [x] **`RoutingStep.depends_on`** + `execution_plan_bridge` 透传
- [x] Router 阶段 **`semantic_routing`**（metadata.stages）
- [x] **`tests/test_phase27_travel_router.py`**

#### P1 — 运维 / 安全文档（27.4–27.5）

- [x] **[docs/operations.md](docs/operations.md)** — Docker、探针、env、多租户、CI、发版、故障排查
- [x] **[docs/security.md](docs/security.md)** — API_KEYS、密钥、租户隔离、传输与 trace

- [x] **Phase 27 全部完成**（版本 **0.22.0**）

### Phase 28 — pre_survey_mode + bridge pre_survey（已完成）

**目标**：缩小 Router workflow 与 legacy-graph 在 travel 上的质量差距。

- [x] **`PipelineConfig.pre_survey_mode`**：`router_prefill` \| `full_ch2` \| `off`
- [x] **`full_ch2`**：Fixed Graph 跑完整 Ch2 LLM，并回写 `execution_plan.pre_survey`
- [x] **Router semantic_routing**：`run_domain_decomposition` 注入 `pre_survey_bridge` 结果
- [x] **travel 插件默认 `full_ch2`**；`execution_plan.pre_survey_mode` / `pre_survey_source` 可观测
- [x] **`tests/test_phase28_pre_survey_mode.py`**

- [x] **Phase 28 全部完成**（版本 **0.23.0**）

### Phase 29 — 子任务流式摘要（已完成）

**目标**：子 Agent 完成后即推送 `graph.subtask.completed`，改善长链路 UX。

- [x] **`graph.subtask.completed`** + `build_subtask_summary()` + `StreamSink.emit_subtask_completed`
- [x] **`execute_layer`** 流式模式下子任务完成即 emit
- [x] **`run_demo.py --stream`** 打印 progress / subtask / aggregate token
- [x] **`tests/test_phase29_subtask_stream.py`**

- [x] **Phase 29 全部完成**（版本 **0.23.0**）



### Phase 7D — A2A Server 暴露子 Agent（已完成）



- [x] **`agent_framework/transport/a2a/server/`** — Executor + Starlette app + CLI

- [x] **`serve_sub_agent(domain, registry_agent=...)`** / `create_sub_agent_a2a_app`

- [x] **Server 侧 trace 提取**（`TraceContextMiddleware`）

- [x] CLI：`python -m agent_framework.transport.a2a.server --domain demo --agent EchoAgent`

- [x] 文档 [`docs/a2a_server.md`](docs/a2a_server.md)



## 运行



```bash

cd Chapter-8-20260616

pip install -e ".[api,dev]"

pip install -e domains/

pytest

python scripts/run_api.py

```



### Docker



```bash

cd Chapter-8-20260616

docker compose up --build

# http://localhost:8780/health

```



## API



| 方法 | 路径 | 说明 |

|------|------|------|

| GET | `/health` | 存活探针 |

| GET | `/ready` | 就绪探针 |

| GET | `/v1/domains` | 已注册领域列表 |

| POST | `/v1/chat` | 同步问答（`domain` + `user_id`） |

| POST | `/v1/jobs` | 提交异步任务（`domain` + `user_id`） |

| GET | `/v1/jobs/{job_id}` | 查询任务状态/结果 |



### 示例



```bash

# 同步（多租户）

curl -X POST http://127.0.0.1:8780/v1/chat \

  -H "Content-Type: application/json" \

  -d '{"query":"北京明天天气","user_id":"alice"}'



# 异步（长行程）

curl -X POST http://127.0.0.1:8780/v1/jobs \

  -H "Content-Type: application/json" \

  -d '{"query":"规划上海苏州杭州七日游","domain":"travel","user_id":"bob"}'



curl http://127.0.0.1:8780/v1/jobs/job-xxxxxxxxxxxx

```



## 环境变量



| 变量 | 默认 | 说明 |

|------|------|------|

| `REQUEST_TIMEOUT_SEC` | `120` | 编排超时 |

| `MAX_CONCURRENT_REQUESTS` | `4` | 并发上限 |

| `TENANT_ORCHESTRATOR_CACHE_SIZE` | `32` | 租户编排器 LRU 缓存 |

| `JOB_DB_PATH` | `data/jobs.db` | 异步任务库 |

| `CHECKPOINT_BACKEND` | `memory` | `memory` \| `sqlite` |

| `DEFAULT_DOMAIN` | *(空)* | 可选：仅当设置时才作为 API 未传 `domain` 的回落 |
| `READY_DOMAIN` | *(空)* | 可选：`/ready` 探针预热某一领域（如 `travel`） |
| `MEMORY_NAMESPACE_PREFIX` | `chapter8_memories` | Store 命名空间 |
| `OTEL_SERVICE_NAME` | `multi-agent-platform` | 平台 OTEL 服务名 |
| `API_KEYS` | *(空)* | 逗号分隔 API Key；设置后 `/v1/*` 需 `X-API-Key` |
| `REGISTRY_FEDERATION_URLS` | *(空)* | 逗号分隔远程 platform 基址，用于 Registry 联邦 |
| `REGISTRY_FEDERATION_API_KEY` | *(空)* | 拉取远程 `/v1/agents` 时使用的 API Key |
| `KNOWLEDGE_TENANT_ISOLATION` | `true` | 多租户 KB overlay 开关 |
| `TRAVEL_A2A_HOTEL_URL` | *(空)* | travel 领域 Supervisor mixed/a2a 时酒店 Agent 的 A2A 基址（如 `http://127.0.0.1:9012/`） |

