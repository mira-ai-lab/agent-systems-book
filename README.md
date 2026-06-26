# agent-systems-book

《智能体系统》配套代码仓库：从单 Agent 推理、记忆与任务规划，到多智能体编排与工程化整合（Chapter-8），MCP / A2A 远程协作，轨迹失败归因（Chapter-9），再到通用多智能体路由平台（Chapter-11）的完整示例。

## 仓库结构

```
agent-systems-book/
├── .env.example          # 环境变量模板（复制为 .env 后填写）
├── Chapter-2/            # 思维链推理（Chain-of-Thought）
├── Chapter-3/            # 长期记忆：向量存储与检索
├── Chapter-4/            # 任务拆解与依赖排序
├── Chapter-5/            # 单领域 LangChain Agent（酒店推荐）
├── Chapter-6/            # 旅行多智能体系统（核心章节）
│   ├── chapter6/         # 路径与 .env 加载（统一入口）
│   ├── sub_agents.py     # 6 个子智能体
│   ├── fixed_graph/      # LangGraph 固定 StateGraph 编排
│   └── supervisor/       # Supervisor handoff 动态调度
├── Chapter-7/            # 协议化协作
│   ├── A2A/              # Agent-to-Agent（HTTP 远程 Agent + Supervisor）
│   └── Mcp/              # Model Context Protocol（HTTP/SSE 工具服务）
├── Chapter-8/            # 旅行多智能体工程化整合（LangGraph 固定图 + 可观测性）
│   ├── travel_multi_agent/   # Python 包（domain / agents / orchestration / tracing）
│   ├── scripts/          # run_demo.py（支持 --stream）、show_graph.py
│   ├── book/             # 书稿离线演示（central_agent_demo_short.py 等）
│   └── tests/
├── Chapter-9/            # 轨迹转换 + 失败归因（AutoFA）
│   ├── testdata/         # span JSONL 样本与 convert_spans_to_whowhen.py
│   ├── Automated_FA/     # inference.py / evaluate.py
│   └── automated_failure_attribution_eval.ipynb
└── Chapter-11/           # 通用多智能体路由引擎（agent-platform）
    ├── agent_framework/  # Router / 编排 / tracing / optimization SDK
    ├── domains/          # travel（主示例）、demo（插件模板）
    ├── services/api/     # FastAPI：/v1/chat、SSE 流式
    ├── scripts/          # run_demo.py、run_api.py、travel/、dev/
    ├── packages/         # router-client（TS SDK）、demo-web（Web UI）
    ├── notebooks/        # planner TextGrad 交互教程
    └── tests/
```

## 各章内容概览

| 章节 | 主题 | 典型入口 |
|------|------|----------|
| **Chapter-2** | 思维链预调查、推理链 | `reasoning_chain_of_thought.ipynb` |
| **Chapter-3** | 文本向量、Chroma 记忆读写 | `memory_store_retrieve_simple.py` |
| **Chapter-4** | 任务分解、依赖图排序 | `task_decompose_demo.py` |
| **Chapter-5** | `@tool` + LangChain 酒店 Agent | `hotel_recommendation_demo.py` |
| **Chapter-6** | 6 子智能体 + 三种编排模式 | 见下方「Chapter-6 快速开始」 |
| **Chapter-7 A2A** | 远程 Agent 协议 + Supervisor 调度 | `A2A/supervisor_local.py` |
| **Chapter-7 MCP** | 酒店工具 MCP Server（SSE） | `Mcp/hotel_mcp_server.py` |
| **Chapter-8** | Ch2–5 能力整合 + LangGraph 固定图 + Tracing | `scripts/run_demo.py --stream` |
| **Chapter-9** | OpenTelemetry 轨迹 → Who&When → 失败归因 | `testdata/convert_spans_to_whowhen.py` |
| **Chapter-11** | Router + 域插件 + HTTP API + Web Demo + TextGrad | `scripts/run_demo.py --domain travel --stream` |

Chapter-6 整合了前几章能力，是全书的主线 demo；**Chapter-8** 在相同业务能力上提供 **工程化 Python 包**（`travel_multi_agent`）、**OpenTelemetry 可观测性** 与 **书稿离线案例**（`book/`）；Chapter-7 演示 **MCP 工具化** 与 **A2A 服务化** 两种对外协作方式；**Chapter-9** 基于 Chapter-8 导出的 span 轨迹做 **格式转换与 Automated Failure Attribution（AutoFA）**；**Chapter-11** 将 Chapter-8 能力 **平台化** 为通用 Router SDK（`agent_framework` + 域插件），并提供 HTTP API、TypeScript SDK 与 Web Demo。

---

## 环境准备

### Python 与依赖

推荐 Python **3.10+**，可使用 Conda 虚拟环境：

```bash
conda create -n agent-systems-book python=3.10 -y
conda activate agent-systems-book
```

各章有独立 `requirements.txt`；**Chapter-6 / Chapter-8 / Chapter-11 建议可编辑安装**（后续 import 更顺畅）：

```bash
cd Chapter-6
pip install -e .

cd Chapter-8
pip install -e .
pip install -e ".[dev]"    # pytest

cd Chapter-11
pip install -e ".[api,dev]"
pip install -e domains/              # 注册 travel 等域插件
pip install -e ".[evolution]"        # 可选：TextGrad prompt 优化
```

Chapter-7 A2A / MCP 另需安装对应目录下的 `requirements.txt`（含 `a2a-sdk`、`mcp` 等）。

Chapter-9 AutoFA：

```bash
cd Chapter-9/Automated_FA
pip install openai python-dotenv    # 通义千问 API 模式最小依赖
# 详见 Chapter-9/testdata/README.md
```

Chapter-11 Web Demo 前端另需 **Node.js ≥ 18**（`packages/demo-web`）。

### 配置 `.env`（必读）

本书所有需要 LLM / 地图 / 航班的脚本，都通过 **`python-dotenv`** 读取环境变量。  
**不要**把真实 Key 提交到 Git；仓库只提供模板 **[`.env.example`](.env.example)**（含各变量申请链接与默认值注释）。

**配置原则**：通用 Key 放**书根 `.env`**；某章有特殊项时，再复制对应章节的局部 `.env`（如 Chapter-9 的 `Automated_FA/.env.example`）。

#### 第一步：复制模板

在**仓库根目录**执行：

```powershell
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

#### 第二步：按场景填写

打开书根 `.env`，把 `your-xxx` 占位符换成真实值。**不必把 `.env.example` 里每一行都取消注释**——只配你当前要跑的章节即可。

| 使用场景 | 最少要配 | 建议额外配置 |
|----------|----------|--------------|
| Chapter-2～5 notebook / 单 Agent | `DASHSCOPE_API_KEY` | Chapter-3 记忆：`DASHSCOPE_EMBEDDING_*` |
| Chapter-6 / 8 / 11 多智能体 CLI | `DASHSCOPE_API_KEY` | `AMAP_KEY` 或 `BAIDU_MAP_AK`（真实 POI）；`WEATHERAPI_KEY`（天气 MCP） |
| Chapter-7 MCP / A2A | 书根 LLM Key + 对应章 `requirements.txt` | `HOTEL_MCP_*`（MCP 地址）；A2A 服务目录下局部 `.env` |
| Chapter-8 导出 trace | 同上 | 六、`OTEL_TRACES_EXPORTER=file` 等（见 `.env.example`） |
| Chapter-9 失败归因 | 书根或 `Automated_FA/.env` 的 `DASHSCOPE_API_KEY` | `FA_REFERENCE_DATE`（复现样例轨迹时） |
| Chapter-11 HTTP API + Web Demo | `DASHSCOPE_API_KEY` | 七、`API_PORT`（默认 8780）；生产环境配 `API_KEYS` |

**最少配置（多智能体 CLI 能跑通）：**

| 变量 | 是否必填 | 说明 |
|------|----------|------|
| `DASHSCOPE_API_KEY` | **必填** | 阿里云百炼 DashScope API Key |
| `DASHSCOPE_CHAT_MODEL` | 可选 | 默认 `qwen-plus` |
| `AMAP_KEY` / `BAIDU_MAP_AK` | 可选 | 地图 POI；不配则回退模拟数据或备用接口 |
| `WEATHERAPI_KEY` | 可选 | 天气 MCP；不配则走高德或 wttr.in |

变量分段、可选开关与申请入口的完整说明见 **[`.env.example`](.env.example)**。

#### 第三步：保存并运行

保存 `.env` 后，直接运行各章脚本或 notebook，**无需**在命令行手动 `export` 变量。

#### 加载顺序

不同章节的加载方式略有差异，原则一致：**先读章节本地 `.env`，再读书根目录 `.env`**（后者可覆盖前者，取决于 `override` 参数）。

| 模块 | 加载逻辑 |
|------|----------|
| **Chapter-6** | `chapter6/paths.py` → `Chapter-6/.env` → 书根 `.env` |
| **Chapter-8** | `travel_multi_agent/config.py` → `Chapter-8/.env` → 书根 `.env` |
| **Chapter-11** | `agent_framework/config.py` → `Chapter-11/.env` → 书根 `.env` |
| **Chapter-7 A2A** | `hotel_recommendation_agent/.env` → 书根 `.env` |
| **Chapter-9 AutoFA** | `Automated_FA/.env`（可与书根共用 Key；含 `FA_REFERENCE_DATE`） |
| **Chapter-3 等 notebook** | 多数直接 `load_dotenv(书根/.env)` |

#### `.env.example` 段落对照

根目录 [`.env.example`](.env.example) 按八段组织；README 只列索引，**详细注释以文件为准**：

| 段落 | 变量示例 | 影响范围 |
|------|----------|----------|
| 一、大模型 | `DASHSCOPE_API_KEY`、`OPENAI_*` | Chapter-2～11 通用 |
| 二、地图 POI | `AMAP_KEY`、`BAIDU_MAP_AK` | 酒店 / 景点 / 美食 Agent；Chapter-8 / 11 travel |
| 二点五、天气 MCP | `WEATHERAPI_KEY`、`WEATHER_USE_MCP` | Chapter-6 / 8 / 11 WeatherAgent |
| 二点六、酒店 MCP | `HOTEL_MCP_*` | Chapter-7 MCP Server |
| 二点七、旅行 MCP | `TRAVEL_MCP_*` | Chapter-8 / 11 `travel_agent_mcp_server.py` |
| 三、航班 | `AVIATIONSTACK_KEY`、`VARIFLIGHT_API_KEY` | Chapter-6 / 8 / 11 FlightAgent |
| 四、其他 | `TMDB_BEARER_TOKEN` | 扩展 demo |
| 五、长期记忆 | `MEMORY_BACKEND`、`MEMORY_NAMESPACE_PREFIX` | Chapter-6 / 8 / 11 的 `chroma_memory/` |
| 六、Chapter-8 可观测性 | `OTEL_TRACES_EXPORTER`、`LOG_LEVEL` | OpenTelemetry span 与结构化日志 |
| 七、Chapter-11 平台 | `API_PORT`、`API_KEYS`、`CHECKPOINT_*`、`DEFAULT_DOMAIN` | HTTP API、Web Demo、多轮 checkpoint |
| 八、Chapter-9 AutoFA | `FA_REFERENCE_DATE` 等 | 见 `Chapter-9/Automated_FA/.env.example` |

Chapter-11 运维向变量全集见 [Chapter-11/docs/operations.md](Chapter-11/docs/operations.md)；Web Demo 前端 build 变量见 [Chapter-11/packages/demo-web/README.md](Chapter-11/packages/demo-web/README.md)（`VITE_API_BASE_URL`）。

> **安全提示**：`.env` 已在 `.gitignore` 中忽略；若误提交，请立即在云平台轮换 Key。

---

## Chapter-6 快速开始

Chapter-6 提供三种编排方式，能力等价、调度机制不同：

| 模式 | 入口 | 说明 |
|------|------|------|
| 顺序编排 | `python central_orchestrator.py` | 最简单，适合入门 |
| 固定图 | `python -m fixed_graph.run_demo` | 显式 StateGraph，可可视化 |
| Supervisor | `cd supervisor && python local_supervisor.py` | 动态 handoff + 复合任务规划流水线 |

子智能体：**Weather · Attraction · Hotel · Restaurant · Flight · Itinerary**

详细对比见：

- [Chapter-6/README.md](Chapter-6/README.md)
- [Chapter-6/supervisor/README.md](Chapter-6/supervisor/README.md)
- [Chapter-6/fixed_graph/README.md](Chapter-6/fixed_graph/README.md)

---

## Chapter-8 快速开始

Chapter-8 将 **Chapter-2～5** 能力整合为独立 Python 包 `travel_multi_agent`，采用 **LangGraph 固定图** 编排，并内置 **OpenTelemetry + 结构化日志**。

固定图六步：

```
pre_survey → retrieve_memory → build_plan → execute_layer → aggregate → save_memory
```

| 用途 | 命令 |
|------|------|
| 查看图结构（无需 API Key） | `cd Chapter-8 && python scripts/show_graph.py` |
| 完整演示（流式输出，推荐） | `python scripts/run_demo.py --stream` |
| 完整演示（批量输出） | `python scripts/run_demo.py` |
| 交互对话 | `python scripts/run_demo.py --stream --chat` |
| 书稿离线案例（无需 API） | `cd book && python central_agent_demo_short.py` |
| 书稿扩展案例 | `cd book && python central_agent_demo.py` |
| 单元测试 | `cd Chapter-8 && pytest` |

子智能体：**Weather · Attraction · Hotel · Restaurant · Flight · Itinerary**（与 Chapter-6 同源能力）

与 Chapter-6 关系：Chapter-6 提供多种编排入口（顺序 / fixed_graph / supervisor）；Chapter-8 聚焦 **固定图工程化实现**（包结构、tracing、测试与书稿案例）。

详细说明见 [Chapter-8/README.md](Chapter-8/README.md)

---

## Chapter-7 快速开始

### A2A：远程 Agent + Supervisor

```powershell
# 终端 1：启动酒店 A2A 服务
cd Chapter-7/A2A/hotel_recommendation_agent
python server.py --host 127.0.0.1 --port 9012

# 终端 2：连通性测试
cd Chapter-7/A2A
python check_a2a_call.py --timing

# 终端 3：Supervisor 调度远程 A2A
python supervisor_local.py
```

书稿注释版入口：`Chapter-7/A2A/book/supervisor_local_book.py`

说明见 [Chapter-7/A2A/hotel_recommendation_agent/README.md](Chapter-7/A2A/hotel_recommendation_agent/README.md)

### MCP：HTTP/SSE 工具服务

```powershell
# 终端 1：启动 MCP Server
cd Chapter-7/Mcp
python hotel_mcp_server.py

# 终端 2：客户端演示
python hotel_recommendation_demo_mcp.py
```

说明见 [Chapter-7/Mcp/README.md](Chapter-7/Mcp/README.md)

---

## Chapter-9 快速开始

Chapter-9 演示 **OpenTelemetry span 轨迹 → Who&When 格式 → 失败归因（AutoFA）** 全链路，输入可来自 Chapter-8 等多智能体系统导出的 `spans_*.jsonl`。

```powershell
# 1. 格式转换（仓库内已含示例 span）
cd Chapter-9/testdata
python convert_spans_to_whowhen.py spans_20260615_180713.jsonl

# 2. 失败归因（需配置 Automated_FA/.env 中的 DASHSCOPE_API_KEY）
cd ../Automated_FA
python inference.py

# 3. 可选：与人工标注对比准确率
python evaluate.py
```

交互式全流程见 `automated_failure_attribution_eval.ipynb`。

详细说明见 [Chapter-9/testdata/README.md](Chapter-9/testdata/README.md)

---

## Chapter-11 快速开始

Chapter-11 是 **通用多智能体路由引擎（agent-platform）**：`agent_framework/` 与业务域解耦，**travel** 为仓库内完整示例域；整合 Router（L1）与 LangGraph Fixed Graph / Supervisor（L2）。

| 用途 | 命令 |
|------|------|
| 完整 CLI（流式，推荐） | `python scripts/run_demo.py --domain travel --profile workflow --stream` |
| HTTP API | `python scripts/run_api.py`（默认 `8780`） |
| Web Demo | API + `packages/demo-web` → `npm run dev`（`5173`） |
| 编排图（无需 Key） | `python scripts/show_graph.py` |
| 单元测试 | `pytest -m "not integration"` |
| TextGrad 教程 | `notebooks/planner_b1_textgrad_graph.ipynb` |

```powershell
cd Chapter-11
pip install -e ".[api,dev]"
pip install -e domains/

# CLI 演示
python scripts/run_demo.py --domain travel --profile workflow --stream

# Web Demo：终端 1 启动 API，终端 2 启动前端
python scripts/run_api.py
cd packages\demo-web && npm install && npm run dev
```

与 Chapter-8 关系：业务能力同源（旅行多 Agent + 固定图流水线）；Chapter-11 增加 **语义 Router**、**域插件协议**、**HTTP / TS SDK**、**benchmark + TextGrad 优化轨** 与 **Web Demo**。

详细说明见 [Chapter-11/README.md](Chapter-11/README.md)

---

## 架构演进（全书主线）

```
Chapter-2  思维链预调查
    ↓
Chapter-3  长期记忆检索 / 写入
    ↓
Chapter-4  任务拆解 → 依赖排序
    ↓
Chapter-5  单 Agent + Tool（酒店）
    ↓
Chapter-6  多 Agent 编排（fixed_graph / supervisor）
    ↓
Chapter-8  工程化整合（travel_multi_agent 包 · 固定图 · 可观测性 · 书稿案例）
    ↓
Chapter-7  对外协作（MCP 工具协议 · A2A 远程 Agent 协议）
    ↓
Chapter-9  轨迹转换 · 失败归因（AutoFA · Who&When）
    ↓
Chapter-11 平台化（Router SDK · 域插件 · API · Web Demo · Prompt 进化）
```

---

## 常见问题

**Q：只配了 `DASHSCOPE_API_KEY`，能跑吗？**  
A：可以。Chapter-6 / Chapter-8 / Chapter-11 核心流程可运行；地图、航班等未配置时会回退 stub 或公开备用接口（如 wttr.in 天气）。

**Q：Chapter-6 和 Chapter-8 有什么区别？**  
A：业务能力等价（6 个子智能体 + 预调查 + 记忆 + 任务规划）。Chapter-6 含三种编排入口与 Supervisor 模式；Chapter-8 以 **LangGraph 固定图** 为主，代码组织为 `travel_multi_agent` 包，并增加 **Tracing**、**pytest** 与 **`book/` 书稿离线案例**。

**Q：Chapter-8 和 Chapter-11 有什么区别？**  
A：旅行业务能力相近；Chapter-11 在 Chapter-8 工程化基础上 **抽象为通用 SDK**（`agent_framework` + `domains/` 插件），增加 **Router 语义路由**、**FastAPI + SSE**、**TypeScript SDK / Web Demo**、**benchmark 与 TextGrad 优化**，是当前仓库的 **生产化主线**。

**Q：Chapter-8 书稿案例用哪个文件？**  
A：正文配套精简版 `Chapter-8/book/central_agent_demo_short.py`（离线可跑）；扩展版 `central_agent_demo.py`；联网完整版 `scripts/run_demo.py --stream`。

**Q：Chapter-9 的轨迹从哪来？**  
A：适配 OpenTelemetry span JSONL（如 `latc.travel-multi-agent` 导出）；仓库 `Chapter-9/testdata/` 含示例 `spans_*.jsonl` 与转换脚本，也可替换为你自己系统的轨迹。

**Q：模型名写 `qwen3.6-plus` 报错？**  
A：请改为 `qwen-plus`（或 `.env` 里 `DASHSCOPE_CHAT_MODEL` / `DEPLOYMENT_NAME` 使用百炼控制台实际模型名）。

**Q：Chapter-6 和 Chapter-7 的 `travel_common` 是什么关系？**  
A：Chapter-6 为共享库源头；Chapter-7 A2A/MCP 目录各有副本或引用，酒店检索逻辑与 Chapter-5/6 保持一致。

**Q：长期记忆存在哪？**  
A：Chapter-6 默认 Chroma 持久化到 `Chapter-6/chroma_memory/`；Chapter-8 默认到 `Chapter-8/chroma_memory/`；Chapter-11 默认到 `Chapter-11/chroma_memory/`。均可通过 `MEMORY_BACKEND=store` 切换 LangGraph Store（进程内，重启清空）。

---

## 相关文档索引

| 文档 | 内容 |
|------|------|
| [.env.example](.env.example) | 全部环境变量说明与申请链接 |
| [Chapter-6/README.md](Chapter-6/README.md) | 多智能体目录结构与三种模式 |
| [Chapter-6/supervisor/README.md](Chapter-6/supervisor/README.md) | Supervisor handoff 与长期记忆 |
| [Chapter-8/README.md](Chapter-8/README.md) | 工程化多智能体包、Tracing、运行与 API |
| [Chapter-7/A2A/hotel_recommendation_agent/README.md](Chapter-7/A2A/hotel_recommendation_agent/README.md) | A2A 服务启动与测试 |
| [Chapter-7/Mcp/README.md](Chapter-7/Mcp/README.md) | MCP Server / Client |
| [Chapter-9/testdata/README.md](Chapter-9/testdata/README.md) | 轨迹转换与 AutoFA 失败归因 |
| [Chapter-11/README.md](Chapter-11/README.md) | Router SDK、域插件、API、Web Demo、TextGrad |
| [Chapter-11/tests/README.md](Chapter-11/tests/README.md) | 测试模块与 phase 对照 |
| [Chapter-11/docs/](Chapter-11/docs/) | SDK 集成、Router、编排、插件开发等设计文档 |
