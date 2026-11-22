# agent-systems-book

《智能体系统》配套代码仓库：从单 Agent 推理、记忆与任务规划，到多智能体编排与工程化整合（Chapter-8），MCP / A2A 远程协作，轨迹失败归因（Chapter-9），再到通用多智能体路由平台（Chapter-11）的完整示例。

## 仓库结构

```
agent-systems-book/
├── llm.py                # 全书共享模型客户端与同步/异步调用
├── .env.example          # 环境变量模板（复制为 .env 后填写）
├── Chapter-2/            # 思维链推理（Chain-of-Thought）
├── Chapter-3/            # 长期记忆：向量存储与检索
├── Chapter-4/            # 任务拆解与依赖排序
├── Chapter-5/            # 单领域 LangChain Agent（酒店推荐）
├── Chapter-6/            # 旅行多智能体系统（核心章节）
│   ├── config.py         # 路径与 .env 加载（统一入口）
│   ├── domain_agents.py  # 6 个领域智能体
│   └── orchestrator.py     # 中心编排器入口
├── Chapter-7/            # 协议化协作
│   ├── A2A/              # Agent-to-Agent（HTTP 远程 Agent + Supervisor）
│   └── Mcp/              # Model Context Protocol（Streamable HTTP 工具服务）
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
| **Chapter-3** | 文本向量、Chroma 记忆读写 | `memory.ipynb` |
| **Chapter-4** | 任务分解、依赖图排序 | `task_decomposition_and_planning.ipynb` |
| **Chapter-5** | `@tool` + LangChain 酒店 Agent | `hotel_recommendation.ipynb` |
| **Chapter-6** | 中心控制器 + 6 个子智能体 | 见下方「Chapter-6 快速开始」 |
| **Chapter-7 A2A** | 远程 Agent 协议 + Supervisor 调度 | `A2A/supervisor.py` |
| **Chapter-7 MCP** | 酒店工具 MCP Server（Streamable HTTP） | `Mcp/server.py` |
| **Chapter-8** | Ch2–5 能力整合 + LangGraph 固定图 + Tracing | `scripts/run_demo.py --stream` |
| **Chapter-9** | OpenTelemetry 轨迹 → Who&When → 失败归因 | `testdata/convert_spans_to_whowhen.py` |
| **Chapter-11** | Router + 域插件 + HTTP API + Web Demo + TextGrad | `scripts/run_demo.py --domain travel --stream` |

Chapter-6 整合了前几章能力，是全书的主线 demo；**Chapter-8** 在相同业务能力上提供 **工程化 Python 包**（`travel_multi_agent`）、**OpenTelemetry 可观测性** 与 **书稿离线案例**（`book/`）；Chapter-7 演示 **MCP 工具化** 与 **A2A 服务化** 两种对外协作方式；**Chapter-9** 基于 Chapter-8 导出的 span 轨迹做 **格式转换与 Automated Failure Attribution（AutoFA）**；**Chapter-11** 将 Chapter-8 能力 **平台化** 为通用 Router SDK（`agent_framework` + 域插件），并提供 HTTP API、TypeScript SDK 与 Web Demo。

> **关于书稿第 10 章（自演化 Agent）**  
> 正文第 10 章讲解 Agent **自演化**（评估反馈、Prompt / 策略迭代与复用等）。本仓库**不设独立 `Chapter-10/` 目录**，该主题已**并入 `Chapter-11/`**，与 Router、编排、benchmark 同一工作区：
>
> | 书稿概念 | 仓库落点（Chapter-11） |
> |----------|------------------------|
> | 自演化 / 优化闭环 | `agent_framework/optimization/`（与生产 LangGraph **并行**的优化轨） |
> | 基于评测的 Prompt 迭代 | TextGrad + `scripts/travel/eval_*.py` / `optimize_*.py` |
> | 交互式理解优化流程 | `notebooks/planner_b1_textgrad_graph.ipynb` |
> | 优化产物持久化 | `data/benchmark/**/optimized/` |
>
> 安装可选依赖：`pip install -e ".[evolution]"`。详见 [Chapter-11/docs/prompt_evolution.md](Chapter-11/docs/prompt_evolution.md) 与 [Chapter-11/README.md](Chapter-11/README.md) 中「Benchmark 与 Prompt 进化」小节。

---

## 环境准备

### Python 与依赖

推荐使用 **Python 3.12** 和 [uv](https://docs.astral.sh/uv/)统一管理虚拟环境与 Python 依赖。

在**仓库根目录**创建一个共享的 `.venv`，供各章代码和 notebook 共同使用：

```bash
uv venv --python 3.12
```

激活虚拟环境：

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

各章保留独立的 `requirements.txt`。阅读或运行某一章时，只需将该章依赖安装到根目录的共享环境中。例如：

```bash
uv pip install -r Chapter-2/requirements.txt
```

notebook 应选择根目录 `.venv` 对应的 Python 内核。

**Chapter-6 / Chapter-8 / Chapter-11 建议使用可编辑安装**，便于直接导入章节内的 Python 包：

```bash
uv pip install -e ./Chapter-6

uv pip install -e ./Chapter-8
uv pip install -e "./Chapter-8[dev]"    # pytest

uv pip install -e "./Chapter-11[api,dev]"
uv pip install -e ./Chapter-11/domains/              # 注册 travel 等域插件
uv pip install -e "./Chapter-11[evolution]"          # 可选：TextGrad prompt 优化
```

Chapter-7 A2A / MCP 需安装对应目录下的 `requirements.txt`（含 `a2a-sdk`、`mcp` 等），例如：

```bash
uv pip install -r Chapter-7/A2A/requirements.txt
uv pip install -r Chapter-7/Mcp/requirements.txt
```

Chapter-9 AutoFA：

```bash
uv pip install openai python-dotenv    # 通义千问 API 模式最小依赖
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
| Chapter-6 多智能体 CLI | `DASHSCOPE_API_KEY` | `AMAP_KEY` 或 `BAIDU_MAP_AK`（真实 POI）；WeatherAgent / FlightAgent 返回空结果 |
| Chapter-8 / 11 多智能体 CLI | `DASHSCOPE_API_KEY` | `AMAP_KEY` 或 `BAIDU_MAP_AK`；`WEATHERAPI_KEY`（天气 MCP） |
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
| `WEATHERAPI_KEY` | 可选 | Chapter-8 / 11 天气 MCP；Chapter-6 不依赖该变量 |

变量分段、可选开关与申请入口的完整说明见 **[`.env.example`](.env.example)**。

#### 第三步：保存并运行

保存 `.env` 后，直接运行各章脚本或 notebook，**无需**在命令行手动 `export` 变量。

#### 加载顺序

不同章节的加载方式略有差异，原则一致：**先读章节本地 `.env`，再读书根目录 `.env`**（后者可覆盖前者，取决于 `override` 参数）。

| 模块 | 加载逻辑 |
|------|----------|
| **Chapter-6** | `config.py` → `Chapter-6/.env` → 书根 `.env` |
| **Chapter-8** | `travel_multi_agent/config.py` → `Chapter-8/.env` → 书根 `.env` |
| **Chapter-11** | `agent_framework/config.py` → `Chapter-11/.env` → 书根 `.env` |
| **Chapter-7 A2A** | `hotel_recommendation_agent/.env` → 书根 `.env` |
| **Chapter-9 AutoFA** | `Automated_FA/.env`（可与书根共用 Key；含 `FA_REFERENCE_DATE`） |
| **Chapter-3～6** | 仓库根 `llm.py` → 仅加载书根 `.env`（不读取章节本地 `.env`） |

#### `.env.example` 段落对照

根目录 [`.env.example`](.env.example) 按八段组织；README 只列索引，**详细注释以文件为准**：

| 段落 | 变量示例 | 影响范围 |
|------|----------|----------|
| 一、大模型 | `DASHSCOPE_API_KEY`、`OPENAI_*` | Chapter-2～11 通用 |
| 二、地图 POI | `AMAP_KEY`、`BAIDU_MAP_AK` | 酒店 / 景点 / 美食 Agent；Chapter-8 / 11 travel |
| 二点五、天气 MCP | `WEATHERAPI_KEY`、`WEATHER_USE_MCP` | Chapter-8 / 11 WeatherAgent |
| 二点六、酒店 MCP | `HOTEL_MCP_*` | Chapter-7 MCP Server |
| 二点七、旅行 MCP | `TRAVEL_MCP_*` | Chapter-8 / 11 `travel_agent_mcp_server.py` |
| 三、航班 | `AVIATIONSTACK_KEY`、`VARIFLIGHT_API_KEY` | Chapter-8 / 11 FlightAgent；Chapter-6 返回空航班列表 |
| 四、其他 | `TMDB_BEARER_TOKEN` | 扩展 demo |
| 五、长期记忆 | `MEMORY_BACKEND`、`MEMORY_NAMESPACE_PREFIX` | Chapter-8 / 11 的后端切换；Chapter-6 固定使用 Chroma |
| 六、Chapter-8 可观测性 | `OTEL_TRACES_EXPORTER`、`LOG_LEVEL` | OpenTelemetry span 与结构化日志 |
| 七、Chapter-11 平台 | `API_PORT`、`API_KEYS`、`CHECKPOINT_*`、`DEFAULT_DOMAIN` | HTTP API、Web Demo、多轮 checkpoint |
| 八、Chapter-9 AutoFA | `FA_REFERENCE_DATE` 等 | 见 `Chapter-9/Automated_FA/.env.example` |

Chapter-11 运维向变量全集见 [Chapter-11/docs/operations.md](Chapter-11/docs/operations.md)；Web Demo 前端 build 变量见 [Chapter-11/packages/demo-web/README.md](Chapter-11/packages/demo-web/README.md)（`VITE_API_BASE_URL`）。

> **安全提示**：`.env` 已在 `.gitignore` 中忽略；若误提交，请立即在云平台轮换 Key。

---

## Chapter-6 快速开始

Chapter-6 使用一个中心控制器完成预调查、记忆检索、任务拆解、依赖分析、
子智能体调度和结果聚合。运行入口：

```bash
python orchestrator.py
```

子智能体：**Weather · Attraction · Hotel · Restaurant · Flight · Itinerary**

详细说明见 [Chapter-6/README.md](Chapter-6/README.md)。

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

与 Chapter-6 关系：Chapter-6 聚焦中心控制器编排；Chapter-8 将相同业务能力进一步落地为 **固定图工程化实现**（包结构、tracing、测试与书稿案例）。

详细说明见 [Chapter-8/README.md](Chapter-8/README.md)

---

## Chapter-7 快速开始

本章两个主 Notebook：

- [Chapter-7/01_mcp_tool_calling.ipynb](Chapter-7/01_mcp_tool_calling.ipynb)：MCP 远程化工具。
- [Chapter-7/02_a2a_agent_collaboration.ipynb](Chapter-7/02_a2a_agent_collaboration.ipynb)：A2A 远程化智能体。

### A2A：远程 Agent + Supervisor

```powershell
# 终端 1：启动酒店 A2A 服务
cd Chapter-7/A2A/hotel_recommendation_agent
python server.py --host 127.0.0.1 --port 9012

# 终端 2：连通性测试
cd Chapter-7/A2A
python check.py --timing

# 终端 3：Supervisor 调度远程 A2A
python supervisor.py
```

书稿注释版入口：`Chapter-7/A2A/supervisor_annotated.py`

说明见 [Chapter-7/A2A/README.md](Chapter-7/A2A/README.md)

### MCP：Streamable HTTP 工具服务

```powershell
cd Chapter-7
# 打开 01_mcp_tool_calling.ipynb 并按顺序执行
```

Notebook 会在 Part I 中启动 `server.py` 定义的 MCP Server，并在结束单元关闭服务；脚本版仍可按 `Mcp/README.md` 分终端运行。

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

Chapter-11 是 **通用多智能体路由引擎（agent-platform）**：`agent_framework/` 与业务域解耦，**travel** 为仓库内完整示例域；整合 Router（L1）与 LangGraph Fixed Graph / Supervisor（L2）。**书稿第 10 章自演化**（Prompt / 策略迭代）亦在本章 `optimization/` 与 TextGrad 脚本中，无独立 `Chapter-10/` 目录。

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
Chapter-6  多 Agent 中心控制器编排
    ↓
Chapter-8  工程化整合（travel_multi_agent 包 · 固定图 · 可观测性 · 书稿案例）
    ↓
Chapter-7  对外协作（MCP 工具协议 · A2A 远程 Agent 协议）
    ↓
Chapter-9  轨迹转换 · 失败归因（AutoFA · Who&When）
    ↓
Chapter-11 平台化（Router SDK · 域插件 · API · Web Demo · Prompt 进化）
         ↑ 书稿第 10 章「自演化」并入本章（optimization 轨 · TextGrad · benchmark）
```

---

## 常见问题

**Q：只配了 `DASHSCOPE_API_KEY`，能跑吗？**  
A：可以。Chapter-6 核心流程可运行，但天气和航班工具返回空结果；Chapter-8 / 11 未配置天气 MCP 时会回退其他天气源。

**Q：Chapter-6 和 Chapter-8 有什么区别？**  
A：业务能力相近（6 个子智能体 + 预调查 + 记忆 + 任务规划）。Chapter-6 聚焦中心控制器编排；Chapter-8 以 **LangGraph 固定图** 为主，代码组织为 `travel_multi_agent` 包，并增加 **Tracing**、**pytest** 与 **`book/` 书稿离线案例**。

**Q：Chapter-8 和 Chapter-11 有什么区别？**  
A：旅行业务能力相近；Chapter-11 在 Chapter-8 工程化基础上 **抽象为通用 SDK**（`agent_framework` + `domains/` 插件），增加 **Router 语义路由**、**FastAPI + SSE**、**TypeScript SDK / Web Demo**、**benchmark 与 TextGrad 优化**，是当前仓库的 **生产化主线**。

**Q：Chapter-8 书稿案例用哪个文件？**  
A：正文配套精简版 `Chapter-8/book/central_agent_demo_short.py`（离线可跑）；扩展版 `central_agent_demo.py`；联网完整版 `scripts/run_demo.py --stream`。

**Q：Chapter-9 的轨迹从哪来？**  
A：适配 OpenTelemetry span JSONL（如 `latc.travel-multi-agent` 导出）；仓库 `Chapter-9/testdata/` 含示例 `spans_*.jsonl` 与转换脚本，也可替换为你自己系统的轨迹。

**Q：模型名写 `qwen3.6-plus` 报错？**  
A：请改为 `qwen-plus`（或 `.env` 里 `DASHSCOPE_CHAT_MODEL` / `DEPLOYMENT_NAME` 使用百炼控制台实际模型名）。

**Q：书稿第 10 章（自演化 Agent）代码在哪？**  
A：仓库**无 `Chapter-10/` 目录**。正文自演化主题已并入 **Chapter-11**：`agent_framework/optimization/`、TextGrad 脚本与 `notebooks/planner_b1_textgrad_graph.ipynb`；与主编排并行，产物写入 `data/benchmark/**/optimized/`。见 [prompt_evolution.md](Chapter-11/docs/prompt_evolution.md)。

**Q：长期记忆存在哪？**  
A：Chapter-6 固定使用 Chroma，持久化到 `Chapter-6/chroma_memory/`；Chapter-8 和 Chapter-11 默认也使用各自章节目录下的 `chroma_memory/`，并通过 `MEMORY_BACKEND=store` 切换 LangGraph Store（进程内，重启清空）。

---

## 相关文档索引

| 文档 | 内容 |
|------|------|
| [.env.example](.env.example) | 全部环境变量说明与申请链接 |
| [Chapter-6/README.md](Chapter-6/README.md) | 中心控制器编排与子智能体 |
| [Chapter-8/README.md](Chapter-8/README.md) | 工程化多智能体包、Tracing、运行与 API |
| [Chapter-7/A2A/README.md](Chapter-7/A2A/README.md) | A2A 服务、协议测试与 Supervisor |
| [Chapter-7/Mcp/README.md](Chapter-7/Mcp/README.md) | MCP Server / Client |
| [Chapter-9/testdata/README.md](Chapter-9/testdata/README.md) | 轨迹转换与 AutoFA 失败归因 |
| [Chapter-11/README.md](Chapter-11/README.md) | Router SDK、域插件、API、Web Demo、TextGrad（含书稿第 10 章自演化） |
| [Chapter-11/docs/prompt_evolution.md](Chapter-11/docs/prompt_evolution.md) | Prompt 进化 / TextGrad / benchmark 优化轨 |
| [Chapter-11/tests/README.md](Chapter-11/tests/README.md) | 测试模块与 phase 对照 |
| [Chapter-11/docs/](Chapter-11/docs/) | SDK 集成、Router、编排、插件开发等设计文档 |

---

## 关于本书案例

本书案例源自编者团队的近年研究工作与项目实践。在梳理并转化为教材过程中，考虑到知识讲解的连贯性和理解成本，进行了简化与调整。我们非常鼓励读者在理解和掌握案例的基础上，结合自身的思考与实际需求，尝试对现有方案进行优化，探索更多可能的实现路径。

本书的编写过程，也恰好经历了 AI 技术快速迭代与开发范式持续演进的阶段。从传统编程，到 Vibe Coding，再到以 Spec 为核心的规范驱动开发；从 RAG 技术的探索，到长期记忆、智能体编排，再到智能体行为归因，每一章不仅介绍了一项技术或一种方法，也承载着编者在不同阶段的研究探索、技术实践与项目落地。

这些方向本身也始终处于演进之中。例如，在长期记忆领域，技术探索经历了从基于文本的记忆，到基于向量化的语义检索，再到引入事件图谱等结构化记忆组织方式的演进，近期也出现从记忆出发进一步探索大模型信念建模的研究。在智能体编排领域，实践则从早期基于工作流的任务组织，逐步发展到以中心编排为代表的多智能体协作模式。在这一过程中，围绕长上下文的存储、压缩与传递，以及智能体路由、任务分解和协作机制等问题，也形成了持续深入的研究与实践。

因此，我们并不试图将某一种技术方案定义为最终答案，也不希望将读者的探索局限于书中的案例。技术仍在演进，探索也不会止步。期待读者以书中的知识与案例为基础，在实践中发现问题、提出新的思路，开展更有价值的研究探索，并将技术创新转化为能够解决客户实际问题、创造真实价值的项目实践。
