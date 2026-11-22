# Chapter-8：旅行多智能体系统（LangGraph 固定图 + A2A + MCP）

本章把第 2~7 章的能力整合成一个旅行多智能体系统，重点演示三件事：

1. **中心编排用 LangGraph 固定图**：`pre_survey → retrieve_memory → build_plan → route → execute_layer → aggregate → save_memory`，依次对应第 2、3、4、6、5+ 章。
2. **子智能体先以 A2A 服务启动**：5 个领域智能体各自是独立进程，通过 Agent Card 注册到中心智能体（复用第 7 章的 A2A 基础设施）。
3. **工具走第三方 MCP 且保持常驻会话**：天气（weatherapi-mcp）、地图/酒店/餐厅（百度地图 MCP）、航班（飞常准 Aviation MCP）在子智能体服务启动时绑定，stdio 会话在服务生命周期内复用。

## 目录结构

```
Chapter-8/
├── pyproject.toml
├── requirements.txt
├── travel_multi_agent.ipynb           # 本章入口 Notebook（一步步讲解）
├── travel_multi_agent/                # 唯一 Python 包
│   ├── config.py                      # 路径 + sys.path + .env 加载
│   ├── mcp_tools.py                   # MCP 统一注册 + 常驻会话
│   ├── planning.py                    # 复用 Ch6：事实调查 + 任务拆解（不含路由）
│   ├── memory.py                      # 复用 Ch6 MemoryAdapter + LangGraph Store 后端
│   ├── agents/
│   │   ├── registry.py                # DomainAgentName 枚举 + 能力目录 + A2A agent 工厂
│   │   └── agent_cards/*.json         # 5 张 A2A Agent Card（name → endpoint）
│   ├── a2a/
│   │   ├── server.py                  # 把领域智能体发布为 A2A 服务（复用 Ch7）
│   │   ├── client.py                  # Agent Card 注册 + A2AClient（复用 Ch7）
│   │   └── settings.py                # 枚举键控的 host / ports
│   ├── orchestration/
│   │   ├── pipeline.py                # PipelineConfig 开关
│   │   ├── aggregation_helpers.py     # 代码式聚合（无聚合 LLM）
│   │   └── fixed_graph/
│   │       ├── orchestrator.py        # LangGraphOrchestrator
│   │       ├── graph.py               # 图装配（含 route 节点）
│   │       ├── nodes.py               # 各阶段节点实现
│   │       ├── state.py               # CentralAgentState
│   │       ├── stream_sink.py
│   │       └── visualize.py
│   └── tracing/                       # OpenTelemetry + 结构化日志
└── tests/
```

## 架构

```
用户请求
  → [Ch2] 思维链预调查 pre_survey
  → [Ch3] 长期记忆检索 retrieve_memory
  → [Ch4] 任务拆解 + 依赖分析 build_plan
  → [Ch6] 领域智能体路由 route          ← 独立节点，复用 Ch6 agent_routing
  → [Ch5+] 按层并行调用子智能体 execute_layer
       ├── WeatherAgent     ── A2A ──→ weatherapi-mcp
       ├── HotelAgent       ── A2A ──→ 百度地图 MCP
       ├── RestaurantAgent  ── A2A ──→ 百度地图 MCP
       ├── FlightAgent      ── A2A ──→ 飞常准 Aviation MCP
       └── ItineraryAgent   ── A2A ──→ （无工具，综合依赖结果）
  → 聚合 aggregate
  → [Ch3] 写入记忆 save_memory
```

中心智能体不直接实例化子智能体：它从 `agents/agent_cards/*.json` 读取 Agent Card（名字 → endpoint），执行层通过第 7 章的 `A2AClient` 调用对应服务。

## 安装

```bash
cd Chapter-8
pip install -e .
# 开发依赖
pip install -e ".[dev]"
```

## 环境配置

在书仓库根目录或 `Chapter-8/.env` 配置：

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` / `OPENROUTER_API_KEY` / `DASHSCOPE_API_KEY` | 大模型（三选一） |
| `WEATHERAPI_KEY` | 天气 MCP（weatherapi-mcp） |
| `BAIDU_MAP_API_KEY`（或 `BAIDU_MAP_AK`） | 百度地图 MCP |
| `VARIFLIGHT_API_KEY` | 飞常准 Aviation MCP |
| `MEMORY_BACKEND` | 记忆后端：`chroma`（默认）/ `store` |
| `A2A_SERVER_HOST` | A2A 服务监听地址，默认 `127.0.0.1` |

天气和百度地图 MCP 通过 `npx` 启动，需要本机安装 Node.js。

## 启动子智能体 A2A 服务

每个领域智能体一个进程（端口见 `travel_multi_agent/a2a/settings.py`）：

```bash
cd Chapter-8
python -m travel_multi_agent.a2a.server \
  --agent-card travel_multi_agent/agents/agent_cards/weather_agent.json \
  --host 127.0.0.1 --port 9031 --public-url http://127.0.0.1:9031/
```

其余服务同理：`hotel_agent.json`（9032）、`restaurant_agent.json`（9033）、`flight_agent.json`（9034）、`itinerary_agent.json`（9035）。

也可直接打开 `travel_multi_agent.ipynb`，由 Notebook 统一拉起/停止这 5 个服务。

## 运行中心编排器

服务就绪后，直接运行入口 Notebook `travel_multi_agent.ipynb`，或在代码中调用：

```python
from travel_multi_agent.config import load_project_dotenv
from travel_multi_agent.orchestration.fixed_graph import LangGraphOrchestrator

load_project_dotenv()
orchestrator = LangGraphOrchestrator(enable_memory=True)
result = await orchestrator.process_request("查询上海明天天气")
print(result["final_response"])
```

## 单元测试

```bash
cd Chapter-8
pytest
```

## 可观测性

沿用 `OTEL_*` 环境变量（详见下文）：span 前缀由 `OTEL_SERVICE_NAME` 派生，默认 `latc.travel-multi-agent`。

```
latc.travel-multi-agent.request
├── orchestration.pre_survey
├── orchestration.retrieve_memory
├── orchestration.build_plan
├── orchestration.route            ← 独立路由节点
├── orchestration.execute_layer
│   └── agent.invoke (agent.name=WeatherAgent)   ← 经 A2A 调用
├── orchestration.aggregate
└── orchestration.save_memory
```

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OTEL_SERVICE_NAME` | `travel-multi-agent` | 服务名（span 前缀由此派生） |
| `OTEL_TRACES_EXPORTER` | `console` | `console` / `file` / `otlp` / `none` |
| `OTEL_TRACES_DIR` | `Chapter-8/traces/` | `file` 模式写入目录 |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `LOG_JSON` | `false` | `true` 输出 JSON 日志 |
