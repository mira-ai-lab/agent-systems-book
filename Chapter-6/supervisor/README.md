# Chapter-6 Supervisor 模式

使用 **LangGraph Supervisor**（`create_supervisor` + handoff）实现与 `fixed_graph/` 等价的旅行多智能体能力。

## 与 fixed_graph/ 对比

| 维度 | `fixed_graph/` | `supervisor/` |
|------|--------------|---------------|
| 调度模式 | 固定流水线：预调查→规划→按层执行→聚合 | Supervisor 动态 handoff |
| 子智能体 | Chapter-5/6 本地 SubAgent | 同上（本地 SubAgent） |
| 短期记忆 | MemorySaver checkpoint | **MemorySaver**（Supervisor 原生 thread 记忆） |
| 长期记忆 | LongTermMemory (Chroma) | **LongTermMemory**（共用 chroma_memory） |
| 单任务查询 | 直达子智能体回复 | 同样支持（单 agent handoff 后直达） |

## 长期记忆后端切换

| 后端 | 参数 | 说明 |
|------|------|------|
| **Chroma** | `long_term_backend="chroma"` | Chapter-3 实现，持久化到 `chroma_memory/` |
| **LangGraph Store** | `long_term_backend="store"` | LangGraph 原生 Store + 向量索引，注入 `compile(store=...)` |

```python
# Chroma（默认）
orchestrator = SupervisorOrchestrator(enable_memory=True, long_term_backend="chroma")

# LangGraph Store
orchestrator = SupervisorOrchestrator(enable_memory=True, long_term_backend="store")
```

环境变量：`MEMORY_BACKEND=chroma|store`

> **注意**：Store 后端当前使用 `InMemoryStore`（进程内单例共享，重启后清空）。需磁盘持久化请用 `chroma` 后端，或后续接入 PostgresStore。

`fixed_graph/LangGraphOrchestrator` 同样支持 `long_term_backend` 参数。

## 安装（推荐）

在 `Chapter-6` 目录可编辑安装后，全项目使用正常 `import`（无需 `_ch6_loader`）：

```bash
cd Chapter-6
pip install -e .
```

## 运行

```bash
cd Chapter-6/supervisor
python test.py
```

## 依赖

```bash
pip install langgraph-supervisor
```

## 交互式单文件入口（推荐）

`local_supervisor.py` — 双模式入口，全本地 sub_agents：

| 模式 | 触发条件 | 机制 |
|------|----------|------|
| **Supervisor** | TaskPlanner 拆解为 **1** 个子任务 | `create_supervisor` 动态 handoff |
| **规划流水线** | TaskPlanner 拆解为 **>1** 个子任务 | `planned_pipeline.py`：`build_plan` + `execute_layer`（同 `fixed_graph/`） |

路由不再依赖关键词匹配，每次请求先跑 `build_plan` 看子任务数量。

```bash
cd supervisor
python local_supervisor.py
```

单条非交互：

```bash
python local_supervisor.py -q "查询上海明天天气"
python local_supervisor.py -q "我下周从上海去成都玩3天，帮我查天气、订机票、推荐景点和美食，最后出个行程"
```

