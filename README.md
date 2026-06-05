# agent-systems-book

《智能体系统》配套代码仓库：从单 Agent 推理、记忆与任务规划，到多智能体编排，再到 MCP / A2A 远程协作的完整示例。

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
└── Chapter-7/            # 协议化协作
    ├── A2A/              # Agent-to-Agent（HTTP 远程 Agent + Supervisor）
    └── Mcp/              # Model Context Protocol（HTTP/SSE 工具服务）
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

Chapter-6 整合了前几章能力，是全书的主线 demo；Chapter-7 在相同业务能力上演示 **MCP 工具化** 与 **A2A 服务化** 两种对外协作方式。

---

## 环境准备

### Python 与依赖

推荐 Python **3.10+**，可使用 Conda 虚拟环境：

```bash
conda create -n agent-systems-book python=3.10 -y
conda activate agent-systems-book
```

各章有独立 `requirements.txt`；**Chapter-6 建议可编辑安装**（后续 import 更顺畅）：

```bash
cd Chapter-6
pip install -e .
```

Chapter-7 A2A / MCP 另需安装对应目录下的 `requirements.txt`（含 `a2a-sdk`、`mcp` 等）。

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

**最少配置（能跑通 Chapter-6 多智能体）：**

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
| **Chapter-7 A2A** | `hotel_recommendation_agent/.env` → 书根 `.env` |
| **Chapter-3 等 notebook** | 多数直接 `load_dotenv(书根/.env)` |

**推荐做法**：把通用 Key（LLM、地图）统一放在**书根 `.env`**；仅某章有特殊配置时，再在对应章节目录放局部 `.env`。

#### `.env.example` 章节对照

`.env.example` 已按用途分段，可按需配置：

| 段落 | 变量示例 | 影响范围 |
|------|----------|----------|
| 一、大模型 | `DASHSCOPE_API_KEY`、`OPENAI_*` | Chapter-2～7 通用 |
| 二、地图 POI | `AMAP_KEY`、`BAIDU_MAP_AK` | 酒店 / 景点 / 美食 Agent |
| 二点五、天气 MCP | `WEATHERAPI_KEY` | Chapter-6 WeatherAgent（MCP 优先） |
| 二点六、酒店 MCP | `HOTEL_MCP_*` | Chapter-7 MCP Server 地址 |
| 三、航班 | `AVIATIONSTACK_KEY`、`VARIFLIGHT_API_KEY` | Chapter-6 FlightAgent |
| 四、其他 | `TMDB_BEARER_TOKEN` | 扩展 demo |
| 五、Chapter-6 选项 | `MEMORY_BACKEND=chroma\|store` | 长期记忆后端 |

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
Chapter-7  对外协作（MCP 工具协议 · A2A 远程 Agent 协议）
```

---

## 常见问题

**Q：只配了 `DASHSCOPE_API_KEY`，能跑吗？**  
A：可以。Chapter-6 核心流程可运行；地图、航班等未配置时会回退 stub 或公开备用接口（如 wttr.in 天气）。

**Q：模型名写 `qwen3.6-plus` 报错？**  
A：请改为 `qwen-plus`（或 `.env` 里 `DASHSCOPE_CHAT_MODEL` / `DEPLOYMENT_NAME` 使用百炼控制台实际模型名）。

**Q：Chapter-6 和 Chapter-7 的 `travel_common` 是什么关系？**  
A：Chapter-6 为共享库源头；Chapter-7 A2A/MCP 目录各有副本或引用，酒店检索逻辑与 Chapter-5/6 保持一致。

**Q：长期记忆存在哪？**  
A：默认 Chroma 持久化到 `Chapter-6/chroma_memory/`；可通过 `MEMORY_BACKEND=store` 切换 LangGraph Store（进程内，重启清空）。

---

## 相关文档索引

| 文档 | 内容 |
|------|------|
| [.env.example](.env.example) | 全部环境变量说明与申请链接 |
| [Chapter-6/README.md](Chapter-6/README.md) | 多智能体目录结构与三种模式 |
| [Chapter-6/supervisor/README.md](Chapter-6/supervisor/README.md) | Supervisor handoff 与长期记忆 |
| [Chapter-7/A2A/hotel_recommendation_agent/README.md](Chapter-7/A2A/hotel_recommendation_agent/README.md) | A2A 服务启动与测试 |
| [Chapter-7/Mcp/README.md](Chapter-7/Mcp/README.md) | MCP Server / Client |
