# Chapter-8 Tracing 改造设计：trace_provider + 埋点清单

> 状态：设计文档（未实施）  
> 对齐规范：latc 系列 `@trace_span` / `current_trace_add_event` / `latc.` 采样前缀  
> 现有实现：`travel_multi_agent/tracing/`（`span()` 上下文管理器 + OTel）

---

## 1. 改造目标

| 维度 | 现状 | 目标 |
|------|------|------|
| 业务 API | `with span(...)` 散落在 orchestrator / nodes | `@trace_span` 装饰 async 函数 / async generator |
| Span 命名 | `travel.request`、`orchestration.*`、`agent.*` | `latc.travel-multi-agent.*`（参与采样） |
| 参数落盘 | 手动 `**attributes` | `attrs_args` 自动序列化 |
| 调用链 | OTel context 隐式传播 | 默认隐式；并行 / 跨边界时用 `parent_arg` |
| 业务事件 | `record_tool_event` → `tool.completed` | `current_trace_add_event` + 统一 event 名 |
| 自动事件 | 无 | 每个 span 自动 `request` / `result`（或 `error`） |

---

## 2. 模块分层

```
travel_multi_agent/tracing/
├── setup.py              # 已有：TracerProvider、导出器（console/file/otlp）
├── trace_provider.py     # 【新增】Sampler、@trace_span、current_trace_add_event
├── spans.py              # 【过渡期】span() 委托 trace_provider；最终可删除或仅保留测试兼容
├── logging_config.py     # 已有：trace_id / span_id 注入日志
├── file_exporter.py      # 已有：spans.jsonl
└── __init__.py           # 导出 trace_span、current_trace_add_event、setup_observability
```

**原则**：业务代码（`orchestrator.py`、`nodes.py`、`domain/task_planner.py`）只 import `trace_span` / `current_trace_add_event`，不直接调用 `opentelemetry`。

---

## 3. trace_provider 接口设计

### 3.1 配置与初始化

```python
# trace_provider.py（设计签名，非最终实现）

LATC_PREFIX = "latc."
DEFAULT_ATTR_MAX_LEN = 500
DEFAULT_RESULT_MAX_LEN = 2000

def configure_trace_provider(
    *,
    service_name: str = "travel-multi-agent",
    sample_latc_only: bool = True,       # True：仅 latc.* 前缀 span 被 ParentBased + 规则采样
    always_sample_roots: bool = True,      # 根 span 是否始终采样（便于联调）
) -> None:
    """在 configure_tracing() 内调用；与现有 setup.py 合并或由其委托。"""
```

**采样规则（与 agent-router 对齐）**：

- Span `name` 以 `latc.` 开头 → 参与采样并导出
- 非 `latc.` 前缀 → `DROP`（或 `RECORD_ONLY` 仅本地调试）
- 开发环境可通过 `OTEL_TRACES_SAMPLE_ALL=1` 关闭前缀过滤

### 3.2 核心装饰器

```python
def trace_span(
    name: str,
    *,
    attrs_args: list[str] | None = None,
    parent_arg: str | None = None,
    record_result: bool = True,
    result_max_len: int = DEFAULT_RESULT_MAX_LEN,
) -> Callable:
    """
    装饰 async 函数或 async generator。

    行为：
    1. 进入时创建 span（name 建议 latc.travel-multi-agent.*）
    2. 若 parent_arg 指定且调用方传入非 None，用其绑定父 context（见 3.4）
    3. attrs_args 所列参数名 → 序列化为 span attributes + request event
    4. 正常返回 → result event（可截断）
    5. 异常 → record_exception + error event，原样 re-raise
    6. async generator：在 __anext__ 循环外包 span，迭代结束或异常时关闭
    """
```

**使用约束**：

- `name` 必须以 `latc.` 开头（否则打 WARNING，且可能被 Sampler 丢弃）
- `attrs_args` 中的名字必须是被装饰函数的参数名
- 不支持 `*args` 位置参数名；仅关键字可序列化参数

### 3.3 手动 Event

```python
def current_trace_add_event(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> None:
    """在当前 active span 上追加业务 event；无 active span 时静默或打 debug 日志。"""

def get_current_span_context() -> tuple[str | None, str | None]:
    """返回 (trace_id, span_id)，供 API 响应与日志关联。"""

def attach_parent_context(trace_parent: Any) -> contextvars.Context | None:
    """
    parent_arg 的载体类型（择一实现）：
    - OTel SpanContext
    - (trace_id, span_id) 元组
    - 含 traceparent 的 dict
    返回 attach 后的 token，span 结束时 detach。
    """
```

### 3.4 parent_arg 机制

```python
async def example(
    task: dict,
    trace_parent: Optional[Any] = None,  # parent_arg="trace_parent"
):
    ...
```

**绑定顺序**：

1. 若 `trace_parent` 非 None → `attach_parent_context(trace_parent)` 后再 `start_span`
2. 否则 → 使用当前 OTel context（`context.attach` 已存在的父 span）

**Chapter-8 默认策略**：

- 第一层（orchestrator、图节点）：不传 `parent_arg`，依赖 `travel.request` 根 span
- `_invoke_sub_agent`：第一阶段不传；若 `asyncio.gather` 后 spans.jsonl 显示子 Agent span 挂到错误父节点，再增加 `parent_arg` 并由 `execute_layer_node` 传入 `get_current_span()`

### 3.5 参数序列化（attrs_args）

```python
def serialize_for_trace(value: Any, *, max_len: int = DEFAULT_ATTR_MAX_LEN) -> Any:
    """
    统一序列化规则：
    - str：截断 max_len
    - dict / list：json.dumps(ensure_ascii=False)，总长截断
    - Pydantic / dataclass：model_dump() 或 asdict()
    - CentralAgentState：仅提取白名单字段（见 4.2）
    - 不可序列化：str(value) 后截断
    - 敏感字段过滤：api_key, token, password 等（键名黑名单）
    """
```

**Attribute 键名规范**：点分命名，与 OTel 语义约定一致，例如 `thread.id`、`task.id`、`user.query_preview`。

### 3.6 自动 request / result event

每个被 `@trace_span` 装饰的调用：

| Event name | 时机 | 内容 |
|------------|------|------|
| `request` | span 开始后 | `{param_name: serialized_value}` 来自 `attrs_args` |
| `result` | 正常返回前 | `{status: "ok", preview: ...}` 返回值摘要 |
| `error` | 异常时 | `{status: "error", error_type, error_message}` + `record_exception` |

`record_result=False` 时跳过 `result` event（适用于返回超大 state 的节点）。

### 3.7 与现有 API 的兼容

| 现有 | 迁移后 |
|------|--------|
| `span(name, **attrs)` | `trace_provider.start_span_context(name, attrs)` 或保留 `span()` 委托 |
| `record_tool_event(...)` | `current_trace_add_event("tool.completed", {...})` |
| `record_exception(...)` | 装饰器内部调用；业务侧一般不再手写 |

---

## 4. Span 命名规范

**前缀**：`latc.travel-multi-agent`

| 层级 | 命名模式 | 示例 |
|------|----------|------|
| 请求根 | `.request` | `latc.travel-multi-agent.request` |
| 编排节点 | `.orchestration.{step}` | `latc.travel-multi-agent.orchestration.pre_survey` |
| 执行层 | `.orchestration.execute_layer` | layer.index 作 attribute |
| 子 Agent | `.agent.invoke` | agent.name 作 attribute |
| 领域规划 | `.planner.{step}` | 可选二级 span |
| 工具 | 不单独开 span | 用 event `tool.completed` |

**与旧名对照**：

| 旧名 | 新名 |
|------|------|
| `travel.request` | `latc.travel-multi-agent.request` |
| `orchestration.pre_survey` | `latc.travel-multi-agent.orchestration.pre_survey` |
| `orchestration.execute_layer.1` | `latc.travel-multi-agent.orchestration.execute_layer` + `layer.index=1` |
| `agent.WeatherAgent` | `latc.travel-multi-agent.agent.invoke` + `agent.name=WeatherAgent` |

---

## 5. 业务 Event 清单

| Event name | 触发位置 | attributes |
|------------|----------|------------|
| `request` | 所有 `@trace_span` | 自动 |
| `result` | 所有 `@trace_span` | 自动 |
| `tool.completed` | `_invoke_sub_agent` 内 tool message | `tool.name`, `task.id`, `agent.name`, `tool.has_error`, `tool.output_preview` |
| `sub_agent_conversation` | `_invoke_sub_agent` 返回前 | `query`, `agent`, `response`, `status` |
| `layer.partial_failure` | `execute_layer_node` 层内部分失败 | `layer.index`, `failed_tasks` |
| `memory.retrieved` | `retrieve_memory_node` | `memory.count` |
| `memory.saved` | `save_memory_node` | `memory.type=preference` |
| `plan.built` | `build_plan_node` | `subtask.count`, `layer.count`, `execution.order` |

---

## 6. 完整埋点清单（按模块）

### 6.1 orchestrator.py — `LangGraphOrchestrator`

| 函数 | 是否埋点 | span name | attrs_args | parent_arg | 手动 event | 优先级 |
|------|----------|-----------|------------|------------|------------|--------|
| `process_request` | ✅ | `latc.travel-multi-agent.request` | `user_query`, `thread_id` | — | — | P0 |
| `process_request_stream` | ✅ | `latc.travel-multi-agent.request` | `user_query`, `thread_id` | — | attribute `stream=true` | P0 |
| `iter_request_stream` | ✅（async gen） | `latc.travel-multi-agent.request.stream` | `user_query`, `thread_id` | — | — | P1 |
| `_build_initial_state` | ❌ | — | — | — | — | — |
| `_attach_stdout_stream_handlers` | ❌ | — | — | — | — | — |
| `_result_from_state` | ❌ | — | — | — | — | — |
| `get_visualizer` / `show_graph` / `save_graph` | ❌ | — | — | — | — | — |

**state 序列化白名单**（若 attrs_args 含 `state` 时使用）：

```python
STATE_TRACE_KEYS = ("thread_id", "user_query", "enable_memory", "enable_stream", "current_layer_index")
# user_query 截断 200 字符
```

### 6.2 nodes.py — 图节点与内部函数

| 函数 | 是否埋点 | span name | attrs_args | parent_arg | 手动 event | 优先级 |
|------|----------|-----------|------------|------------|------------|--------|
| `pre_survey_node` | ✅ | `latc.travel-multi-agent.orchestration.pre_survey` | `state`（白名单） | — | — | P0 |
| `retrieve_memory_node` | ✅ | `latc.travel-multi-agent.orchestration.retrieve_memory` | `state` | — | `memory.retrieved` | P0 |
| `build_plan_node` | ✅ | `latc.travel-multi-agent.orchestration.build_plan` | `state` | — | `plan.built` | P0 |
| `execute_layer_node` | ✅ | `latc.travel-multi-agent.orchestration.execute_layer` | `state` | — | `layer.partial_failure` | P0 |
| `aggregate_node` | ✅ | `latc.travel-multi-agent.orchestration.aggregate` | `state` | — | result 含 `final_response.length` | P0 |
| `save_memory_node` | ✅ | `latc.travel-multi-agent.orchestration.save_memory` | `state` | — | `memory.saved` | P0 |
| `_invoke_sub_agent` | ✅ | `latc.travel-multi-agent.agent.invoke` | `task`, `thread_id` | `trace_parent`（二期） | `tool.completed`, `sub_agent_conversation` | P0 |
| `_stream_llm_text` | ⚠️ 可选 | `latc.travel-multi-agent.llm.stream` | — | — | token 数 event | P2 |
| `_traced_node` | 🗑️ 删除 | 由节点直接 `@trace_span` 替代 | — | — | — | P0 |
| `_streaming` / `_append_log` / `_format_result` 等 | ❌ | — | — | — | — | — |
| `has_more_layers` | ❌ | 同步条件函数，无 span | — | — | — | — |

**execute_layer_node attributes（除 attrs_args 外固定写入）**：

- `layer.index`：当前层序号（1-based）
- `layer.tasks`：逗号分隔 task_id
- `thread.id`

**_invoke_sub_agent task 序列化字段**：

- `task.id`, `task.agent`, `task.description`（截断 120）, `task.depends_on`

### 6.3 domain/task_planner.py — 二级 span（可选，P1）

在节点 span 下增加 planner 子 span，便于区分「节点总耗时」与「LLM 规划耗时」。

| 函数 | span name | attrs_args | 说明 |
|------|-----------|------------|------|
| `run_pre_survey` | `latc.travel-multi-agent.planner.pre_survey` | `user_query` | 被 pre_survey_node 调用 |
| `run_decomposition` | `latc.travel-multi-agent.planner.decomposition` | `user_query` | |
| `run_dependency_analysis` | `latc.travel-multi-agent.planner.dependency` | `sub_steps` | |
| `route_to_agents` | `latc.travel-multi-agent.planner.routing` | — | 对标 agent-router `do_routing` |
| `build_execution_plan` | `latc.travel-multi-agent.planner.build_plan` | `user_query` | 或仅装饰此函数覆盖子步骤 |

**建议**：首期只在 `build_execution_plan` 打一个 planner span；细粒度拆分放二期。

### 6.4 agents/*.py — 工具函数（P2，可选）

| 函数 | span name | 说明 |
|------|-----------|------|
| `get_weather` | `latc.travel-multi-agent.tool.weather` | 外部 API 耗时 |
| `recommend_hotel` | `latc.travel-multi-agent.tool.hotel` | |
| `search_flights` | `latc.travel-multi-agent.tool.flight` | |
| 其他 `@tool` | `latc.travel-multi-agent.tool.{name}` | |

首期通过 `tool.completed` event 已能观测工具层；独立 tool span 仅在需要区分「Agent 思考 vs 工具 IO」时再加。

### 6.5 infra/memory — 不单独埋点（P3）

`search_memories` / `ingest` 耗时较短，由节点 span + `memory.*` event 覆盖即可。

### 6.6 book/ — 不埋点

`central_agent_demo_short.py` 等为离线演示，默认 `OTEL_TRACES_EXPORTER=none`。

---

## 7. 目标 Span 树（一次完整请求）

```
latc.travel-multi-agent.request
│  attributes: thread.id, user.query_preview, stream?
│  events: request, result
│
├─ latc.travel-multi-agent.orchestration.pre_survey
│    └─ [P1] latc.travel-multi-agent.planner.pre_survey
│
├─ latc.travel-multi-agent.orchestration.retrieve_memory
│    event: memory.retrieved
│
├─ latc.travel-multi-agent.orchestration.build_plan
│    ├─ [P1] latc.travel-multi-agent.planner.build_plan
│    event: plan.built
│
├─ latc.travel-multi-agent.orchestration.execute_layer  (layer.index=1)
│    ├─ latc.travel-multi-agent.agent.invoke  (agent.name=WeatherAgent)
│    │    events: tool.completed, sub_agent_conversation
│    └─ latc.travel-multi-agent.agent.invoke  (agent.name=AttractionAgent)  [并行]
│
├─ latc.travel-multi-agent.orchestration.execute_layer  (layer.index=2)
│    └─ latc.travel-multi-agent.agent.invoke  (agent.name=ItineraryAgent)
│
├─ latc.travel-multi-agent.orchestration.aggregate
│
└─ latc.travel-multi-agent.orchestration.save_memory
     event: memory.saved
```

---

## 8. 代码改造示意（非最终实现）

### 8.1 orchestrator

```python
@trace_span(
    name="latc.travel-multi-agent.request",
    attrs_args=["user_query", "thread_id"],
)
async def process_request(self, user_query: str, thread_id: str = "default") -> Dict[str, Any]:
    ...
    trace_id, span_id = get_current_span_context()
    return self._result_from_state(final_state, trace_id, span_id)
```

### 8.2 图节点（替代 _traced_node）

```python
@trace_span(
    name="latc.travel-multi-agent.orchestration.pre_survey",
    attrs_args=["state"],
)
async def pre_survey_node(state: CentralAgentState) -> Dict[str, Any]:
    ...
```

### 8.3 子 Agent

```python
@trace_span(
    name="latc.travel-multi-agent.agent.invoke",
    attrs_args=["task", "thread_id"],
    parent_arg="trace_parent",  # 二期按需启用
)
async def _invoke_sub_agent(
    task: Dict[str, Any],
    prior_results: Dict[str, Any],
    thread_id: str,
    trace_parent: Optional[Any] = None,
) -> Dict[str, Any]:
  ...
  current_trace_add_event(
      name="sub_agent_conversation",
      attributes={
          "query": query_preview,
          "agent": agent_name,
          "response": agent_summary[:500],
          "status": status,
      },
  )
```

---

## 9. 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `OTEL_SERVICE_NAME` | `travel-multi-agent` | 已有 |
| `OTEL_TRACES_EXPORTER` | `console` | 已有 |
| `OTEL_TRACES_SAMPLE_ALL` | `0` | `1` 时所有 span 采样（开发用） |
| `OTEL_TRACE_ATTR_MAX_LEN` | `500` | attrs_args 截断 |
| `OTEL_TRACE_RESULT_MAX_LEN` | `2000` | result event 截断 |
| `LOG_LEVEL` / `LOG_JSON` | 已有 | 日志仍带 trace_id |

---

## 10. 测试计划

| 测试项 | 文件 | 断言 |
|--------|------|------|
| `@trace_span` 设置 context | `tests/test_trace_provider.py` | trace_id / span_id 非空 |
| `latc.` 前缀采样 | 同上 | 非 latc span 被 drop（可 mock Sampler） |
| attrs_args 截断 | 同上 | 超长 user_query attribute ≤ max_len |
| 异常传播 | 同上 | span status ERROR + error event |
| async generator | 同上 | `iter_request_stream` span 正确关闭 |
| 并行层父子关系 | `tests/test_tracing.py` | execute_layer 下 agent.invoke 的 parent_span_id 正确 |
| 文件导出 | 已有 `test_file_exporter` | 新 span name 写入 spans.jsonl |

---

## 11. 实施顺序

| 阶段 | 内容 | 风险 |
|------|------|------|
| **Phase 0** | 本文档评审 + span 名定稿 | 低 |
| **Phase 1** | 新增 `trace_provider.py`，`span()` 委托，不改业务 | 低 |
| **Phase 2** | orchestrator + 6 个图节点改 `@trace_span`，删 `_traced_node` | 中 |
| **Phase 3** | `_invoke_sub_agent` + 业务 event | 中 |
| **Phase 4** | 验证 `asyncio.gather` 父子链，按需 `parent_arg` | 中 |
| **Phase 5** | task_planner 二级 span（P1） | 低 |
| **Phase 6** | 更新 README / Chapter-8/README span 树示例 | 低 |

---

## 12. 与 agent-router 对照

| agent-router | Chapter-8 |
|--------------|-----------|
| `latc.agent-router.invoker` | `latc.travel-multi-agent.request` |
| `latc.agent-router.invoder.do_routing` | `latc.travel-multi-agent.orchestration.build_plan` + planner |
| `latc.agent-router.router.routing` | `latc.travel-multi-agent.planner.routing` |
| `sub_agent_conversation` event | `_invoke_sub_agent` 内同名 event |
| `trace_parent` 参数 | `_invoke_sub_agent` 二期 `parent_arg` |

---

## 13. 不在本期范围

- W3C `traceparent` HTTP 透传（Chapter-7 A2A 集成时再做）
- Jaeger UI 仪表盘与告警规则
- Metrics（仅 Traces；Metrics 另立文档）
- `book/` 离线案例埋点

---

*文档版本：v0.1 | 对应代码基线：Chapter-8 `travel_multi_agent` 固定图编排*
