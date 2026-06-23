# agent-systems-book

《智能体系统》配套代码仓库：从单 Agent 推理、记忆与任务规划，到多智能体编排与工程化整合（Chapter-8），MCP / A2A 远程协作，再到 Hermes 自我进化 Agent 的完整示例。

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
└── Chapter-10/           # Hermes 自我进化 Agent（技能学习 + LangGraph）
    ├── Hermes_evolution_langgraph.py   # 完整闭环 + SubAgent
    ├── book/             # 书稿简化版（langgraph_evolu.py 等）
    └── my_agent_memory/  # 运行时技能库与热记忆（可删后重跑）
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
| **Chapter-10** | Hermes 自我进化：技能抽取 / skill_view / patch | `Hermes_evolution_langgraph.py` |

Chapter-6 整合了前几章能力，是全书的主线 demo；**Chapter-8** 在相同业务能力上提供 **工程化 Python 包**（`travel_multi_agent`）、**OpenTelemetry 可观测性** 与 **书稿离线案例**（`book/`）；Chapter-7 演示 **MCP 工具化** 与 **A2A 服务化** 两种对外协作方式；**Chapter-10** 在 LangGraph 上实现 **跨任务技能学习与复用**（Hermes 闭环）。

---

## 环境准备

### Python 与依赖

推荐 Python **3.10+**，可使用 Conda 虚拟环境：

```bash
conda create -n agent-systems-book python=3.10 -y
conda activate agent-systems-book
```

各章有独立 `requirements.txt`；**Chapter-6 / Chapter-8 建议可编辑安装**（后续 import 更顺畅）：

```bash
cd Chapter-6
pip install -e .

cd Chapter-8
pip install -e .
# 开发依赖（pytest）
pip install -e ".[dev]"
```

Chapter-7 A2A / MCP 另需安装对应目录下的 `requirements.txt`（含 `a2a-sdk`、`mcp` 等）。

Chapter-10 安装：

```bash
cd Chapter-skill
pip install -r requirements.txt
```

（完整版 SubAgent 会复用 Chapter-6 子智能体，建议已安装 Chapter-6 依赖。）

### 配置 `.env`（必读）

本书所有需要 LLM / 地图 / 航班的脚本，都通过 **`python-dotenv`** 读取环境变量。  
**不要**把真实 Key 提交到 Git；仓库只提供模板 **`.env.example`**。

#### 第一步：复制模板

在**仓库根目录**执行：

```powershell
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

#### 第二步：填写 Key

用编辑器打开根目录 `.env`，把 `your-xxx` 占位符换成你自己的值。

**最少配置（能跑通 Chapter-6 / Chapter-8 多智能体）：**

| 变量 | 是否必填 | 说明 |
|------|----------|------|
| `DASHSCOPE_API_KEY` | **必填** | 阿里云百炼 DashScope API Key |
| `DASHSCOPE_CHAT_MODEL` | 可选 | 默认 `qwen-plus` |
| `AMAP_KEY` / `BAIDU_MAP_AK` | 可选 | 地图 POI；不配则部分 Agent 回退模拟数据或备用接口 |

申请入口见 `.env.example` 内注释（DashScope、高德、百度等）。

#### 第三步：保存并运行

保存 `.env` 后，直接运行各章脚本或 notebook，**无需**在命令行手动 `export` 变量。

#### 加载顺序

不同章节的加载方式略有差异，原则一致：**先读章节本地 `.env`，再读书根目录 `.env`**（后者可覆盖前者，取决于 `override` 参数）。

| 模块 | 加载逻辑 |
|------|----------|
| **Chapter-6** | `chapter6/paths.py` → `Chapter-6/.env` → 书根 `.env` |
| **Chapter-8** | `travel_multi_agent/config.py` → `Chapter-8/.env` → 书根 `.env` |
| **Chapter-7 A2A** | `hotel_recommendation_agent/.env` → 书根 `.env` |
| **Chapter-3 等 notebook** | 多数直接 `load_dotenv(书根/.env)` |

**推荐做法**：把通用 Key（LLM、地图）统一放在**书根 `.env`**；仅某章有特殊配置时，再在对应章节目录放局部 `.env`。

#### `.env.example` 章节对照

`.env.example` 已按用途分段，可按需配置：

| 段落 | 变量示例 | 影响范围 |
|------|----------|----------|
| 一、大模型 | `DASHSCOPE_API_KEY`、`OPENAI_*` | Chapter-2～8、Chapter-10 通用 |
| 二、地图 POI | `AMAP_KEY`、`BAIDU_MAP_AK` | 酒店 / 景点 / 美食 Agent；Chapter-10 SubAgent |
| 二点五、天气 MCP | `WEATHERAPI_KEY` | Chapter-6 / Chapter-8 / Chapter-10 WeatherAgent（MCP 优先） |
| 二点六、酒店 MCP | `HOTEL_MCP_*` | Chapter-7 MCP Server 地址 |
| 三、航班 | `AVIATIONSTACK_KEY`、`VARIFLIGHT_API_KEY` | Chapter-6 / Chapter-8 FlightAgent |
| 四、其他 | `TMDB_BEARER_TOKEN` | 扩展 demo |
| 五、Chapter-6 选项 | `MEMORY_BACKEND=chroma\|store` | 长期记忆后端 |
| 六、Chapter-8 可观测性 | `OTEL_TRACES_EXPORTER`、`LOG_LEVEL` | OpenTelemetry span 与结构化日志 |

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

## Chapter-10 快速开始

Chapter-10 演示 **Hermes 自我进化闭环**：任务执行 → LLM 评估 → 抽取技能 → 下次 `skill_view` 复用；旅游 Demo 为 **丽江 3 日游学习技能 → 大理 3 日游复用**。

| 模式 | 入口 | 说明 |
|------|------|------|
| 完整 Hermes | `python Hermes_evolution_langgraph.py` | ReAct + `skill_view` + SubAgent（较慢，需 API） |
| 书稿简化版 | `cd book && python langgraph_evolu.py` | 纯 LLM 规划，课堂快速跑通 |

```powershell
cd Chapter-10

# 可选：清空旧技能，重新演示「学习 → 复用」
# $env:TRAVEL_DEMO_FRESH="1"

python Hermes_evolution_langgraph.py
```

可选环境变量：`WEATHER_USE_MCP=0` 跳过天气 MCP；`HERMES_CHECKPOINT=sqlite` 持久化 LangGraph checkpoint。

详细说明见 [Chapter-10/README.md](Chapter-skill/README.md)

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
Chapter-10 自我进化（技能库 SKILL.md · skill_view · 评估 / patch / Hub）
```

---

## 常见问题

**Q：只配了 `DASHSCOPE_API_KEY`，能跑吗？**  
A：可以。Chapter-6 / Chapter-8 核心流程可运行；地图、航班等未配置时会回退 stub 或公开备用接口（如 wttr.in 天气）。

**Q：Chapter-6 和 Chapter-8 有什么区别？**  
A：业务能力等价（6 个子智能体 + 预调查 + 记忆 + 任务规划）。Chapter-6 含三种编排入口与 Supervisor 模式；Chapter-8 以 **LangGraph 固定图** 为主，代码组织为 `travel_multi_agent` 包，并增加 **Tracing**、**pytest** 与 **`book/` 书稿离线案例**。

**Q：Chapter-8 书稿案例用哪个文件？**  
A：正文配套精简版 `Chapter-8/book/central_agent_demo_short.py`（离线可跑）；扩展版 `central_agent_demo.py`；联网完整版 `scripts/run_demo.py --stream`。

**Q：模型名写 `qwen3.6-plus` 报错？**  
A：请改为 `qwen-plus`（或 `.env` 里 `DASHSCOPE_CHAT_MODEL` / `DEPLOYMENT_NAME` 使用百炼控制台实际模型名）。

**Q：Chapter-6 和 Chapter-7 的 `travel_common` 是什么关系？**  
A：Chapter-6 为共享库源头；Chapter-7 A2A/MCP 目录各有副本或引用，酒店检索逻辑与 Chapter-5/6 保持一致。

**Q：长期记忆存在哪？**  
A：Chapter-6 默认 Chroma 持久化到 `Chapter-6/chroma_memory/`；Chapter-8 默认到 `Chapter-8/chroma_memory/`。均可通过 `MEMORY_BACKEND=store` 切换 LangGraph Store（进程内，重启清空）。

**Q：Chapter-10 和 Chapter-6 子智能体什么关系？**  
A：Chapter-10 完整版通过 `@tool` 包装调用 `Chapter-10/sub_agents.py`（与 Chapter-6 同源能力）；书稿简化版 `book/langgraph_evolu.py` 不调用 SubAgent，仅用 LLM 生成行程。

**Q：Chapter-10 技能存在哪？**  
A：默认 `./my_agent_memory/skills/`（Hermes 为 `*.md`，书稿版为 `*.json`）；设置 `TRAVEL_DEMO_FRESH=1` 可清空后重跑 Demo。

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
| [Chapter-10/README.md](Chapter-skill/README.md) | Hermes 自我进化 Agent、两版对比与运行 |
