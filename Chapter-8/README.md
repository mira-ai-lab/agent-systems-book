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
├── book/                         # 书稿示例与 Agent 定义说明
├── scripts/
│   ├── run_demo.py
│   ├── show_graph.py
│   ├── test_weather_agent.py
│   └── travel_agent_mcp_server.py   # Claude Code / MCP 客户端接入
├── docs/
│   └── tracing_design.md
└── tests/
    ├── test_planner.py              # 旅行 registry / prompts / 时间锚点
    ├── test_domain_parsing.py       # parsing + 聚合辅助
    ├── test_domain_registry.py      # SubAgentRegistry / DomainConfig
    ├── test_domain_task_planner.py  # TaskPlanner（Mock LLM）
    ├── test_orchestration_graph.py  # StateGraph 构建 / 条件边
    ├── test_orchestration_orchestrator.py  # LangGraphOrchestrator
    ├── test_mcp_server.py           # travel_agent_mcp_server 工具
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

## Claude Code MCP

`scripts/travel_agent_mcp_server.py` 将本项目的 LangGraph 旅行多智能体以 **MCP stdio** 形式暴露给 [Claude Code](https://code.claude.com/)，可在对话中直接调用天气、酒店、行程等能力。

### 前置条件

1. 已完成上文 **安装**（`pip install -e .`），并额外安装 MCP SDK：

   ```bash
   cd Chapter-8
   pip install mcp
   ```

2. 书仓库根目录或 `Chapter-8/.env` 中已配置 `DASHSCOPE_API_KEY`（服务启动时会通过 `load_project_dotenv()` 自动加载，**无需**在 `.mcp.json` 里重复写 `env`）。

3. 使用与 IDE / `run_demo.py` **同一 Python 解释器**（Windows 建议在配置里写绝对路径，避免 Claude Code 调到别的 `python`）。

### 方式一：项目根 `.mcp.json`（推荐）

在书仓库根目录创建或编辑 `.mcp.json`（Claude Code 会在**项目根**启动时自动读取）：

```json
{
  "mcpServers": {
    "travel-agent": {
      "type": "stdio",
      "command": "D:\\conda\\python.exe",
      "args": [
        "Chapter-8/scripts/travel_agent_mcp_server.py"
      ]
    }
  }
}
```

将 `command` 换成你本机 conda / venv 的 `python.exe` 绝对路径。

**必须在书仓库根目录启动 Claude Code：**

```powershell
cd D:\myproject\mira-ai-lab\agent-systems-book
claude
```

在 Claude Code 内执行 `/mcp`，应看到 `travel-agent` 为 **connected**。然后可提问，例如：

> 用 travel-agent 查一下北京明天天气

首次调用工具时选择 **Yes, and don't ask again** 可免重复授权。

### 方式二：`claude mcp add`（CLI 注册）

在 `Chapter-8` 目录下，将 MCP 注册为**当前项目** scope：

```bash
cd Chapter-8
claude mcp add travel-agent --scope project -- D:\conda\python.exe scripts/travel_agent_mcp_server.py
```

等价于在 `Chapter-8/.mcp.json` 中写入 stdio 配置（`args` 为 `scripts/travel_agent_mcp_server.py`）。若从 `Chapter-8` 子目录启动 `claude`，则读取此文件。

### 暴露的工具

| 工具 | 说明 |
|------|------|
| `ask_travel_agent` | 自然语言问答，返回最终文本答复 |
| `ask_travel_agent_detailed` | 同上，额外返回 `execution_plan`、`subtask_results`、`trace_id` 等结构化字段 |

可选参数 `thread_id`：同一 ID 可复用长期记忆上下文。

### 可选：HTTP/SSE 模式

供非 stdio 的 MCP 客户端使用（Claude Code 默认用 stdio，一般不需要）：

```powershell
$env:TRAVEL_MCP_TRANSPORT = "sse"
python scripts/travel_agent_mcp_server.py
# 默认 http://127.0.0.1:8766/sse
```

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TRAVEL_MCP_TRANSPORT` | `stdio` | `stdio` \| `sse` |
| `TRAVEL_MCP_HOST` | `127.0.0.1` | SSE 监听地址 |
| `TRAVEL_MCP_PORT` | `8766` | SSE 端口 |
| `TRAVEL_MCP_ENABLE_MEMORY` | `1` | `0` 关闭长期记忆 |

### 排障

| 现象 | 处理 |
|------|------|
| `/mcp` 无服务器 | 确认在书仓库根（或 `Chapter-8`）启动 `claude`，且对应目录存在 `.mcp.json` |
| `travel-agent` **failed** | 在相同 `command` 下执行：`python -c "from agent_framework.orchestration.fixed_graph.orchestrator import LangGraphOrchestrator; print('ok')"` |
| `ImportError` / `numpy` 相关 | 用户 site-packages 中 numpy 可能损坏：`pip install --force-reinstall numpy` |
| 缺 API Key | 检查根目录 `.env` 中 `DASHSCOPE_API_KEY` |
| 查看详细日志 | `claude --debug` |

本地快速验证 MCP 进程能否拉起（应无报错并挂起等待 stdio）：

```powershell
cd D:\myproject\mira-ai-lab\agent-systems-book
python Chapter-8/scripts/travel_agent_mcp_server.py
```

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

## 子智能体

WeatherAgent · HotelAgent · RestaurantAgent · FlightAgent · ItineraryAgent

详细设计见 [docs/tracing_design.md](docs/tracing_design.md)；书稿侧 Agent 契约说明见 [book/agent_definitions.py](book/agent_definitions.py)。
