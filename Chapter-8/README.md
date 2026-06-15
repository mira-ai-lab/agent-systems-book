# Chapter-8: 旅行多智能体系统（LangGraph 固定图）

整合 **Chapter-2 / 3 / 4 / 5** 的能力，通过 **LangGraph StateGraph** 实现中心调度 → 子智能体分层执行 → 汇聚结果。

代码分为 **通用框架** 与 **旅行领域插件** 两层，便于换领域时只替换 `domains/<name>/`，编排与追踪逻辑复用。

## 目录结构

```
Chapter-8/
├── pyproject.toml
├── requirements.txt
├── agent_framework/              # 通用：编排 + 追踪 + domain 层
│   ├── config.py                 # paths + dotenv + create_llm
│   ├── domain/                   # registry / planner / pipeline / parsing …
│   ├── orchestration/fixed_graph/
│   ├── tracing/
│   └── infra/memory/
├── domains/
│   └── travel/                   # 旅行领域唯一实现
│       ├── agents/               # Weather / Hotel / Restaurant / Flight / Itinerary
│       ├── prompts.py
│       ├── specs.py
│       ├── registry.py           # create_travel_registry()
│       └── infra/                # travel_api / weather_mcp
├── travel_multi_agent/           # 向后兼容 re-export（旧 import 仍可用）
├── book/                         # 书稿示例与 Agent 定义说明
├── scripts/
│   ├── run_demo.py
│   ├── show_graph.py
│   └── test_weather_agent.py
├── docs/
│   └── tracing_design.md
└── tests/
    ├── test_planner.py
    ├── test_tracing.py
    └── test_trace_provider.py
```

## 安装（必做）

在 **IDEA 使用的同一 Python 解释器**（如 conda `agent-systems-book`）下执行：

```bash
cd Chapter-8
pip install -e .
# 开发依赖（pytest）
pip install -e ".[dev]"
```

未执行 `pip install -e .` 时，直接运行 `scripts/run_demo.py` 会报 `ModuleNotFoundError`（找不到 `agent_framework` / `domains`）。

**IntelliJ IDEA**：将 `Chapter-8` 标记为 Sources Root（仓库 `.idea` 已配置），并确认 Project SDK 为上述 conda 环境。修改后可在 IDEA 中 **File → Invalidate Caches** 刷新索引。

## 运行

| 用途 | 命令 |
|------|------|
| 查看图结构（无需 API Key） | `python scripts/show_graph.py` |
| 完整演示 | `python scripts/run_demo.py` |
| 单元测试 | `pytest` |

## 环境配置

书仓库根目录或 `Chapter-8/.env`：

- `DASHSCOPE_API_KEY` — 百炼大模型
- `AMAP_KEY` / `BAIDU_MAP_AK` — 地图 POI（可选）

Chroma 向量库：`Chapter-8/chroma_memory/`（`agent_framework.config.CHROMA_DIR`）

### 可观测性（OpenTelemetry + 结构化日志）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OTEL_SERVICE_NAME` | `travel-multi-agent` | 服务名（span 前缀由此派生） |
| `OTEL_TRACES_EXPORTER` | `console` | `console` / `file` / `otlp` / `none` |
| `OTEL_TRACES_DIR` | `Chapter-8/traces/` | `file` 模式下 span 写入目录 |
| `OTEL_TRACES_FILE_MODE` | `timestamp` | `timestamp` 按启动时间分文件；`append` 追加 `spans.jsonl` |
| `OTEL_TRACES_FILENAME` | — | 显式指定文件名（优先级最高） |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | Jaeger/Tempo OTLP 地址 |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `LOG_JSON` | `false` | `true` 时输出 JSON 行 |
| `OTEL_TRACES_SAMPLE_ALL` | `0` | `1` 时所有 span 采样（开发 / 测试） |
| `OTEL_TRACE_ATTR_MAX_LEN` | `500` | attrs 截断长度 |
| `OTEL_TRACE_RESULT_MAX_LEN` | `2000` | result event 截断 |

Span 层级（前缀由 `OTEL_SERVICE_NAME` 派生，默认类似 `latc.travel-multi-agent`）：

```
latc.travel-multi-agent.request
├── orchestration.pre_survey
│   └── planner.pre_survey
├── orchestration.retrieve_memory
├── orchestration.build_plan
│   └── planner.build_plan / planner.decomposition / planner.routing ...
├── orchestration.execute_layer
│   └── agent.invoke (agent.name=WeatherAgent)
│       └── event: tool.completed / sub_agent_conversation
├── orchestration.aggregate
└── orchestration.save_memory
```

日志每行带 `trace_id` / `span_id`，可用 `result["trace_id"]` 关联一次请求的全链路。

**写入本地目录（无需 Jaeger）：**

在 `.env` 或运行前设置：

```bash
OTEL_TRACES_EXPORTER=file
# 可选，默认 Chapter-8/traces/
OTEL_TRACES_DIR=D:/myproject/mira-ai-lab/agent-systems-book/Chapter-8/traces
```

运行 `python scripts/run_demo.py` 后，span 会写入 `{OTEL_TRACES_DIR}/spans_YYYYMMDD_HHMMSS.jsonl`（每次进程启动一个新文件）。启动日志会打印 `output_file` 路径。用 `trace_id` 过滤即可查看单次请求：

```powershell
# 查看最新一次 run 的文件（按修改时间排序）
Get-ChildItem traces/spans_*.jsonl | Sort-Object LastWriteTime -Descending | Select-Object -First 1

Select-String -Path traces/spans_20260612_143052.jsonl -Pattern "你的trace_id"
```

若希望恢复旧行为（所有 run 追加到同一文件），在 `.env` 中设置 `OTEL_TRACES_FILE_MODE=append`。

## 架构

**运行时流水线：**

```
用户请求
  → [Ch2] 思维链预调查
  → [Ch3] 长期记忆检索
  → [Ch4] 任务拆解 → 依赖排序 → 子 Agent 路由
  → [Ch5+] 5 个子智能体按层并行执行（execute_layer 循环）
  → 聚合 → [Ch3] 写入记忆
```

**包职责：**

```
agent_framework     orchestration + tracing + 通用 domain（与业务无关）
domains/travel      agents / prompts / specs / infra（旅行领域实现）
travel_multi_agent  兼容层 re-export（历史 import，新代码可不使用）
```

## 代码中使用

**推荐（新代码）：**

```python
from agent_framework.config import load_project_dotenv
from agent_framework.domain.pipeline import PipelineConfig
from agent_framework.orchestration.fixed_graph import LangGraphOrchestrator
from domains.travel import TravelPrompts, create_travel_registry, travel_domain_config

load_project_dotenv()

orchestrator = LangGraphOrchestrator(
    registry=create_travel_registry(),
    prompts=TravelPrompts.build(),
    domain_config=travel_domain_config(enable_guess_agent=True),
    pipeline=PipelineConfig(enable_pre_survey=True, enable_memory=True),
)
result = await orchestrator.process_request("查询上海明天天气")
print(result["final_response"])
```

**简化入口（使用默认旅行 demo 配置）：**

```python
from agent_framework.config import load_project_dotenv
from agent_framework.orchestration.fixed_graph import LangGraphOrchestrator

load_project_dotenv()
orchestrator = LangGraphOrchestrator(enable_memory=True)
result = await orchestrator.process_request("查询上海明天天气")
```

**向后兼容（旧 import 仍可用）：**

```python
from travel_multi_agent.orchestration.fixed_graph import LangGraphOrchestrator
```

## 子智能体

WeatherAgent · HotelAgent · RestaurantAgent · FlightAgent · ItineraryAgent

详细设计见 [docs/tracing_design.md](docs/tracing_design.md)；书稿侧 Agent 契约说明见 [book/agent_definitions.py](book/agent_definitions.py)。
