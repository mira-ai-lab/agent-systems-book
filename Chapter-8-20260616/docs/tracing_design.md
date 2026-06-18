# Tracing 设计说明（Chapter-8）

Chapter-8 使用 **OpenTelemetry Traces + 结构化日志**，实现一次请求从编排到子 Agent 的全链路可观测。

实现位于 `agent_framework/tracing/`；业务代码（`orchestration/`、`domain/`）只依赖 `@trace_span` 与 `current_trace_add_event`，不直接调用 OTel SDK。

---

## 1. 模块一览

| 文件 | 职责 |
|------|------|
| `setup.py` | `TracerProvider`、导出器（console / file / otlp / none） |
| `trace_provider.py` | `@trace_span`、`span_name()`、采样、参数序列化、业务 event |
| `spans.py` | `span()` / `record_tool_event` 兼容层 |
| `logging_config.py` | 日志注入 `trace_id` / `span_id` |
| `file_exporter.py` | 写入 `traces/*.jsonl` |

入口（编排器启动时调用一次）：

```python
from agent_framework.tracing import setup_observability

setup_observability()  # configure_logging + configure_tracing
```

---

## 2. 核心 API

### `@trace_span`

装饰 **async 函数**（及 async generator），自动：

- 创建 span，写入 `request` event（来自 `attrs_args`）
- 正常返回 → `result` event；异常 → `error` event + `record_exception`
- `record_result=False` 时跳过 result（大图节点避免写入超大 state）

```python
from agent_framework.tracing import trace_span, span_name

@trace_span(
    name=span_name("orchestration.pre_survey"),
    attrs_args=["state"],
    record_result=False,
)
async def pre_survey_node(state): ...
```

| 参数 | 说明 |
|------|------|
| `name` | 完整 span 名，推荐 `span_name("后缀")` |
| `attrs_args` | 从函数参数自动序列化为 attribute / request event |
| `parent_arg` | 并行子 Agent 时传入父 span（如 `trace_parent`） |

### `span_name(suffix)`

根据 `OTEL_SERVICE_NAME` 生成 `latc.{service}.{suffix}`，默认：

`latc.multi-agent-platform.orchestration.build_plan`

### `current_trace_add_event(name, attributes)`

在当前 active span 上追加业务 event（无 span 时静默跳过）。

### `get_current_span_context()`

返回 `(trace_id, span_id)`，用于 API 响应与日志关联。

---

## 3. Span 树（一次完整请求）

```
latc.multi-agent-platform.request
├── orchestration.pre_survey
│   └── planner.pre_survey
├── orchestration.retrieve_memory          event: memory.retrieved
├── orchestration.build_plan
│   └── planner.decomposition / dependency / routing / build_plan
│   event: plan.built
├── orchestration.execute_layer              attr: layer.index, layer.tasks
│   └── agent.invoke                         attr: agent.name, task.id
│       events: tool.completed, sub_agent_conversation
├── orchestration.aggregate                  attr: final_response.length
└── orchestration.save_memory                event: memory.saved
```

`execute_layer` 按依赖分层循环；同层多个 `agent.invoke` 通过 `asyncio.gather` 并行，并用 `parent_arg` 挂到当前层 span 下。

---

## 4. 业务 Event 清单

| Event | 触发位置 | 主要字段 |
|-------|----------|----------|
| `request` / `result` / `error` | 所有 `@trace_span` | 自动 |
| `tool.completed` | `_invoke_sub_agent` | `tool.name`, `task.id`, `agent.name`, `tool.has_error` |
| `sub_agent_conversation` | `_invoke_sub_agent` | `query`, `agent`, `response`, `status` |
| `plan.built` | `build_plan_node` | `subtask.count`, `layer.count`, `execution.order` |
| `layer.partial_failure` | `execute_layer_node` | `layer.index`, `failed_tasks` |
| `memory.retrieved` | `retrieve_memory_node` | `memory.count` |
| `memory.saved` | `save_memory_node` | `memory.type` |

---

## 5. 采样与序列化

**采样**（`LatcPrefixSampler`）：

- 默认只采样 `latc.*` 根 span；已采样父 span 的子 span 跟随
- `OTEL_TRACES_SAMPLE_ALL=1` → 全量采样（开发 / 测试）

**参数序列化**（`attrs_args`）：

- 字符串截断（默认 500 字符，`OTEL_TRACE_ATTR_MAX_LEN`）
- `CentralAgentState` 仅提取白名单：`thread_id`, `user_query`, `enable_memory`, `enable_stream`, `current_layer_index`
- 键名含 `api_key` / `token` / `password` 等 → 过滤

---

## 6. 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `OTEL_SERVICE_NAME` | `multi-agent-platform` | 服务名，span 前缀来源 |
| `OTEL_TRACES_EXPORTER` | `console` | `console` / `file` / `otlp` / `none` |
| `OTEL_TRACES_DIR` | `Chapter-8/traces/` | file 模式输出目录 |
| `OTEL_TRACES_FILE_MODE` | `timestamp` | `timestamp` 每次启动新文件；`append` 追加同一文件 |
| `OTEL_TRACES_FILENAME` | — | 显式指定文件名（优先级最高） |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | OTLP 地址 |
| `OTEL_TRACES_SAMPLE_ALL` | `0` | `1` 全量采样 |
| `OTEL_TRACE_ATTR_MAX_LEN` | `500` | attribute 截断 |
| `OTEL_TRACE_RESULT_MAX_LEN` | `2000` | result event 截断 |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `LOG_JSON` | `false` | JSON 行日志 |

**本地落盘示例：**

```bash
OTEL_TRACES_EXPORTER=file
python scripts/run_demo.py
# → traces/spans_YYYYMMDD_HHMMSS.jsonl
```

用 `result["trace_id"]` 或日志中的 `trace_id` 过滤单次请求。

---

## 7. 扩展新埋点

1. 在 async 函数上加 `@trace_span(name=span_name("your.step"), attrs_args=[...])`
2. 需要额外业务语义时，在函数内调用 `current_trace_add_event("your.event", {...})`
3. 新 span 后缀建议沿用现有命名：`orchestration.*` / `planner.*` / `agent.invoke`
4. 补充断言：`tests/test_trace_provider.py`、`tests/test_tracing.py`

---

## 8. 测试

```bash
cd Chapter-8
pytest tests/test_trace_provider.py tests/test_tracing.py -q
```

覆盖：`@trace_span` 上下文、`latc.` 采样、attrs 截断、异常传播、file exporter。

---

*对应代码：`agent_framework/tracing/` · 编排埋点：`orchestrator.py` / `nodes.py` / `domain/task_planner.py`*
