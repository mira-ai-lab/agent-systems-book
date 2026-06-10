# Chapter-8: 旅行多智能体系统（LangGraph 固定图）

整合 **Chapter-2 / 3 / 4 / 5** 的能力，通过 **LangGraph StateGraph** 实现中心调度 → 子智能体分层执行 → 汇聚结果。

## 目录结构

```
Chapter-8/
├── pyproject.toml
├── requirements.txt
├── travel_multi_agent/                   # Python 包（唯一代码根）
│   ├── config.py               # paths + dotenv + create_llm
│   ├── domain/
│   ├── agents/
│   ├── infra/
│   ├── orchestration/fixed_graph/
│   └── tracing/
├── scripts/
│   ├── run_demo.py
│   └── show_graph.py
└── tests/
    ├── test_planner.py
    └── test_tracing.py
```

## 安装（必做）

在 **IDEA 使用的同一 Python 解释器**（如 conda `agent-systems-book`）下执行：

```bash
cd Chapter-8
pip install -e .
# 开发依赖（pytest）
pip install -e ".[dev]"
```

未执行 `pip install -e .` 时，直接运行 `scripts/run_demo.py` 会报 `ModuleNotFoundError: travel_multi_agent`。

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

Chroma 向量库：`Chapter-8/chroma_memory/`（`travel_multi_agent.config.CHROMA_DIR`）

### 可观测性（OpenTelemetry + 结构化日志）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OTEL_SERVICE_NAME` | `travel-multi-agent` | 服务名 |
| `OTEL_TRACES_EXPORTER` | `console` | `console` / `file` / `otlp` / `none` |
| `OTEL_TRACES_DIR` | `Chapter-8/traces/` | `file` 模式下 span 写入目录 |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | Jaeger/Tempo OTLP 地址 |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `LOG_JSON` | `false` | `true` 时输出 JSON 行 |

Span 层级：

```
travel.request
├── orchestration.pre_survey
├── orchestration.retrieve_memory
├── orchestration.build_plan
├── orchestration.execute_layer.1
│   └── agent.WeatherAgent
│       └── event: tool.completed (tool.name=get_weather)
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

运行 `python scripts/run_demo.py` 后，span 会追加写入 `{OTEL_TRACES_DIR}/spans.jsonl`（每行一个 JSON）。用 `trace_id` 过滤即可查看单次请求：

```powershell
Select-String -Path traces/spans.jsonl -Pattern "你的trace_id"
```

## 架构

```
用户请求
  → [Ch2] 思维链预调查
  → [Ch3] 长期记忆检索
  → [Ch4] 任务拆解 → 依赖排序
  → [Ch5+] 6 个子智能体执行（execute_layer 按层循环）
  → 聚合 → [Ch3] 写入记忆
```

## 代码中使用

```python
from travel_multi_agent.config import load_project_dotenv
from travel_multi_agent.orchestration.fixed_graph import LangGraphOrchestrator

load_project_dotenv()
orchestrator = LangGraphOrchestrator(enable_memory=True)
result = await orchestrator.process_request("查询上海明天天气")
print(result["final_response"])
```

## 子智能体

WeatherAgent · AttractionAgent · HotelAgent · RestaurantAgent · FlightAgent · ItineraryAgent

详细说明见 `travel_multi_agent/orchestration/fixed_graph/` 内各模块 docstring。
