# Chapter-11: 通用多智能体路由引擎（agent-platform）

> **产品定位**：**通用** Agent 路由与编排 SDK（Router → workflow / adaptive）——`agent_framework/` 与业务域解耦；**旅行（travel）** 作为书稿与仓库内的**完整示例域**，演示拆解、路由、五类子 Agent、MCP 与 Prompt 进化全链路。  
> **工作区**：本目录为 `Chapter-11` 生产化改造分支。  
> 升级计划见 [UPGRADE.md](UPGRADE.md)。

整合 **Router Engine（L1 路由）** 与 **LangGraph Fixed Graph / Supervisor（L2 执行）**：

`query` → 分类 / 改写 / 拆解 → 预填 `execution_plan` → 子 Agent 分层执行 → 聚合答复。

代码分为两层：

| 层 | 路径 | 职责 |
|----|------|------|
| **通用框架** | `agent_framework/` | Router、编排、追踪、插件协议、optimization 轨 |
| **领域插件** | `domains/<name>/` | 各业务的 Agent、prompts、工具与 `DomainPlugin` |

内置插件：`travel`（**主示例**）、`demo`（最小模板）。新增领域只需实现插件并 `register_domain`，路由与编排逻辑复用。

## SDK 快速入口

```python
from agent_framework.bootstrap import route

# 通用 API：显式指定 domain（以旅行为例）
result = await route(
    "帮我查北京明天天气，并推荐一家三亚海棠湾附近的酒店",
    domain="travel",
    profile="workflow",
)

# profile=auto：≥2 子 Agent → workflow，否则 adaptive（域内 Agent 集合由插件决定）
result = await route("查 7 月 5 日广州飞北京的航班", domain="travel")

# 进阶：显式 runtime
from agent_framework.bootstrap.platform import create_runtime
runtime = create_runtime("travel", profile="workflow")
result = await runtime.process_request("规划杭州三日游")
print(result["final_response"])
```

HTTP API：`POST /v1/chat`（`domain` 可显式传入或由 Router 推断）· 流式 `POST /v1/chat/stream`（SSE）· `GET /v1/domains`  
TypeScript SDK：`packages/router-client`  
文档：[sdk_integration](docs/sdk_integration.md) · [router_engine](docs/router_engine.md) · [orchestration_model](docs/orchestration_model.md) · [plugin_development](docs/plugin_development.md) · [prompt_evolution](docs/prompt_evolution.md) · [security](docs/security.md)

### 执行 Profile

| profile | 说明 |
|---------|------|
| **`auto`**（推荐） | 先路由；多 Agent 意图 → workflow，单 Agent → adaptive |
| **`workflow`** | Fixed Graph；Router 预填计划（travel 示例走语义 `agent_routing`） |
| **`adaptive`** | Supervisor 单 Agent handoff |
| **`hybrid`** | adaptive + A2A 混部 |

`create_orchestrator(domain)` 等价于 `create_runtime(domain, profile="workflow")`。

## 以旅行为例：子 Agent 与能力

`domains/travel/` 是仓库内**最完整的参考实现**，覆盖 Fixed Graph、外部 API、MCP、benchmark 与 TextGrad 优化。

| Agent | 职责 | 核心工具 |
|-------|------|----------|
| WeatherAgent | 城市日期天气 | `get_weather` / `get_weather_forecast` |
| HotelAgent | 酒店推荐 | `recommend_hotel` |
| RestaurantAgent | 美食推荐 | `recommend_restaurant` |
| FlightAgent | 航班查询 | `search_flights` |
| ItineraryAgent | POI + 行程骨架 | `fetch_candidate_pois` / `plan_itinerary` |

**示例请求流水线（travel）：**

```
「下周去西安，查天气、推荐酒店和本地美食」
  → 预调查 → 记忆检索 → 任务拆解 → 依赖排序 → 路由到子 Agent
  → 并行执行（MCP / 外部 API）→ 聚合答复 → 写入记忆
```

扩展新域时复制 `domains/demo/`，对照 [docs/plugin_development.md](docs/plugin_development.md)，无需 fork 框架代码。

## 目录结构

```
Chapter-11/
├── agent_framework/          # 通用 SDK（Router / 编排 / tracing / optimization）
├── domains/
│   ├── travel/                 # ★ 主示例域（Agent + API + benchmark + knowledge）
│   └── demo/                   # 最小插件模板（扩展新域时复制此骨架）
├── services/api/               # FastAPI：/v1/chat、/v1/jobs …
├── scripts/
│   ├── run_demo.py             # 主 CLI 入口（默认 travel）
│   ├── run_router.py / run_api.py / run_legacy.py / show_graph.py
│   ├── travel/                 # travel 评测与 TextGrad 优化
│   └── dev/                    # 开发工具（ingest、benchmark、就绪度检查）
├── data/benchmark/             # 以 travel 为主的 benchmark 与 optimized 产物
├── book/                       # 书稿精简 demo（非生产路径）
├── notebooks/                  # 实验 notebook（见下方「交互式教程」）
├── packages/
│   ├── router-client/          # TypeScript SDK
│   └── demo-web/               # 调用 API 的最小 Web UI
├── docs/
└── tests/                      # 见 tests/README.md
```

**运行时目录**（本地生成，勿提交）：`traces/`（链路日志）· `chroma_memory/`（记忆向量库）· `logs/`（部分组件文件日志）· `data/knowledge/`（KB ingest 持久化）

## 安装

```bash
cd Chapter-11
pip install -e ".[api,dev]"
pip install -e domains/              # entry_points 注册 travel 等插件
pip install -e ".[evolution]"        # 可选：TextGrad prompt 进化
pytest -m "not integration"          # 默认单元测试（无需 live LLM）
```

`.env`：书仓库根或 `Chapter-11/.env` 配置 `DASHSCOPE_API_KEY`（必填）；地图 POI 可选 `AMAP_KEY` / `BAIDU_MAP_AK`。

**PyCharm / IDEA**：`Chapter-11` → Sources Root；`tests` → Test Sources Root。

## 运行

| 用途 | 命令 |
|------|------|
| Router 入口 | `python scripts/run_router.py` |
| 完整 CLI | `python scripts/run_demo.py --domain travel --profile workflow --stream` |
| legacy Fixed Graph | `python scripts/run_legacy.py` |
| HTTP API | `python scripts/run_api.py` |
| Web Demo 前端 | 见下方「Web Demo」；`packages/demo-web` → http://localhost:5173 |
| 编排图（无需 Key） | `python scripts/show_graph.py` |
| 单元测试 | `pytest -m "not integration"` |
| 就绪度自检 | `python scripts/dev/product_readiness_check.py` |
| Planner L1/L2 优化教程 | 打开 `notebooks/planner_b1_textgrad_graph.ipynb`（需 `.[evolution]`） |

## Web Demo（前端 + 后端）

`packages/demo-web` 是最小 Web 调试页（Vite + `@agent-platform/router-client`），用于在浏览器里调用 Router API，并查看 **SSE 流式** 的 `router.*` / `handoff.*` / `final` 事件时间线。适合本地联调，不是生产管理后台。

### 启动（两个终端）

**终端 1 — 后端 API（默认 `8780`）：**

```powershell
cd Chapter-11
pip install -e ".[api,dev]"
pip install -e domains/
python scripts/run_api.py
```

看到 `Uvicorn running on http://127.0.0.1:8780` 即表示 API 已就绪。

**终端 2 — 前端（默认 `5173`）：**

```powershell
cd Chapter-11\packages\demo-web
npm install          # 首次需要；Node.js >= 18
npm run dev          # 会自动 build ../router-client
```

浏览器打开 **http://localhost:5173/**（或终端里 Vite 打印的 Local 地址）。

开发模式下 Vite 将 `/v1/*`、`/health` 代理到 `8780`，页面上 **API Base URL 留空** 即可。

### 页面怎么用

| 字段 / 选项 | 说明 |
|-------------|------|
| **Query** | 用户问题；默认已填 travel 示例 |
| **Domain** | 可选，默认 `travel` |
| **X-API-Key** | 若 `.env` 配置了 `API_KEYS` 则必填；本地未配置时可留空 |
| **SSE 流式** | 勾选：左侧显示 Router 阶段时间线 + 流式最终回复；不勾选：同步 `POST /v1/chat` |

首次请求会调用 LLM，耗时取决于 query 复杂度（travel 多 Agent 场景可能数十秒）。

### 环境变量（节选）

| 变量 | 说明 |
|------|------|
| `DASHSCOPE_API_KEY` | 后端调 LLM 必填（与 CLI demo 相同） |
| `API_PORT` | API 端口，默认 `8780` |
| `API_KEYS` | 非空时启用 `X-API-Key` 鉴权 |
| `VITE_API_BASE_URL` | 前端直连 API 时使用（生产 build）；dev 模式一般留空 |

生产构建示例（前端直连 API，不经 Vite 代理）：

```powershell
cd packages\demo-web
$env:VITE_API_BASE_URL="http://127.0.0.1:8780"
npm run build
npm run preview
```

更多细节见 [packages/demo-web/README.md](packages/demo-web/README.md)、[docs/sdk_integration.md](docs/sdk_integration.md)。SDK 冒烟：`python scripts/dev/smoke_router_client.py`（需 API 已启动）。

## 以旅行为例：Benchmark 与 Prompt 进化

`agent_framework/optimization/` 与生产编排**并行**；当前 benchmark 与脚本以 **travel** 为主，可作为其他域的模板。

```powershell
# L1 任务拆解 · L2 路由 · L3 端到端
python scripts/travel/eval_travel_decomposition.py --split dev
python scripts/travel/eval_travel_routing.py --split dev
python scripts/travel/eval_travel_e2e.py --split dev

# 单子 Agent（如 FlightAgent）
python scripts/travel/eval_travel_agents.py --agent FlightAgent --split dev

# TextGrad 优化（需 .[evolution]）
python scripts/travel/optimize_travel_agent.py --agent FlightAgent --verbose
```

产物：`data/benchmark/**/optimized/`。详见 [docs/prompt_evolution.md](docs/prompt_evolution.md)。

### 交互式教程（Notebook）

面向第一次接触优化流程的读者，推荐从 **`notebooks/planner_b1_textgrad_graph.ipynb`** 动手：

| 内容 | 说明 |
|------|------|
| 目标 | L1 任务拆解 + L2 子任务路由的 **textgrad_graph** 优化（不跑完整 E2E 编排） |
| 步骤 | Step 1 基线评测 → Step 2 优化 → Step 3 读报告 → Step 4 复评对比 |
| 产物 | `data/benchmark/travel_planner/optimized/zh.json` 与 optimization report |
| 依赖 | `pip install -e ".[evolution]"`，配置 `DASHSCOPE_API_KEY`；Jupyter 建议 `pip install nest_asyncio` |

在 IDE 或 Jupyter 中打开 `Chapter-11/notebooks/planner_b1_textgrad_graph.ipynb` 即可按单元格 Step 1→4 执行。完整 E2E 优化见同目录 `planner_e2e_textgrad_graph.ipynb`。

## Claude Code MCP（travel 示例）

`scripts/travel/travel_agent_mcp_server.py` 将 **travel 域** 多智能体暴露为 MCP stdio：

```json
{
  "mcpServers": {
    "travel-agent": {
      "type": "stdio",
      "command": "D:\\conda\\envs\\agent-systems-book\\python.exe",
      "args": ["Chapter-11/scripts/travel/travel_agent_mcp_server.py"]
    }
  }
}
```

工具：`ask_travel_agent` · `ask_travel_agent_detailed`（含 `execution_plan`、`trace_id`）。

## 环境变量（节选）

| 变量 | 说明 |
|------|------|
| `DASHSCOPE_API_KEY` | 大模型（必填） |
| `EXECUTOR_MODEL` / `OPTIMIZER_MODEL` | 评测 / TextGrad（可选） |
| `TRAVEL_OPTIMIZED_AGENT_PROMPTS` | `0` 关闭 travel 子 Agent optimized 覆盖 |
| `OTEL_TRACES_EXPORTER` | `console` / `file` / `otlp` |
| `API_KEYS` | HTTP 鉴权（生产必填，见 [security.md](docs/security.md)） |

Trace：`Chapter-11/traces/` · 记忆向量库：`Chapter-11/chroma_memory/` · 链路设计：[tracing_design.md](docs/tracing_design.md)

## 代码示例

**通用 Router（travel 域）：**

```python
import asyncio
from agent_framework.bootstrap import route
from agent_framework.config import load_project_dotenv

async def main() -> None:
    load_project_dotenv()
    result = await route(
        "查上海到成都 6 月 30 日的航班，并说说成都天气",
        domain="travel",
        profile="workflow",
    )
    print(result["final_response"])

asyncio.run(main())
```

**legacy 直连编排器（不经 Router，travel 域）：**

```python
from agent_framework.orchestration.fixed_graph import LangGraphOrchestrator
from domains.travel import TravelPrompts, create_travel_registry, travel_domain_config
# …见 scripts/run_legacy.py
```

## 架构关系（一句话）

```
agent_framework  = 通用路由 + 编排内核（DomainPlugin 协议）
domains/travel   = 书稿级完整示例（Agent / benchmark / MCP / 优化脚本）
domains/demo     = 最小插件模板；新业务域按同一契约扩展
```

新增领域见 [docs/domains.md](docs/domains.md) 与 [docs/plugin_development.md](docs/plugin_development.md)。
