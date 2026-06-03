# Chapter-6: 旅行多智能体系统

整合 **Chapter-2 / 3 / 4 / 5** 的能力，提供三种编排方式。

## 目录结构

```
Chapter-6/
├── chapter6/              # 路径与 .env 引导（唯一路径来源）
│   └── paths.py
├── travel_common.py       # 共享库：API、日期锚定、工具函数
├── sub_agents.py          # 6 个子智能体
├── task_planner.py        # Ch2 预调查 + Ch4 拆解
├── central_orchestrator.py# 顺序编排入口
├── memory_*.py, prompts.py, aggregation_helpers.py
├── fixed_graph/           # LangGraph 固定 StateGraph
├── supervisor/            # Supervisor handoff + 规划流水线
│   └── book/              # 书籍伪代码案例（不参与运行时）
└── pyproject.toml
```

> **注意**：`supervisor/` 下不再复制共享模块；所有编排层统一 import 根目录共享库。

## 安装

```bash
cd Chapter-6
pip install -e .
```

## 三种运行方式

| 模式 | 入口 | 适用场景 |
|------|------|----------|
| 顺序编排 | `python central_orchestrator.py` / notebook | 教学、最简单 |
| 固定图 | `python -m fixed_graph.run_demo` | 显式 StateGraph + 可视化 |
| Supervisor | `cd supervisor && python local_supervisor.py` | 动态 handoff + 复合任务规划流水线 |

## 环境配置

书仓库根目录或 `Chapter-6/.env`：

- `DASHSCOPE_API_KEY` — 百炼大模型
- `AMAP_KEY` / `BAIDU_MAP_AK` — 地图 POI（可选）

Chroma 向量库统一目录：`Chapter-6/chroma_memory/`（`chapter6.paths.CHROMA_DIR`）

## 架构

```
用户请求
  → [Ch2] 思维链预调查
  → [Ch3] 长期记忆检索
  → [Ch4] 任务拆解 → 依赖排序
  → [Ch5+] 6 个子智能体执行
  → 聚合 → [Ch3] 写入记忆
```

## 子智能体

WeatherAgent · AttractionAgent · HotelAgent · RestaurantAgent · FlightAgent · ItineraryAgent

## 与各章对应

- **Ch2** — `FACTS_PROMPT` 预调查
- **Ch3** — `LongTermMemory` 向量检索与写入
- **Ch4** — 任务拆解、依赖分析、Agent 路由
- **Ch5** — `SubAgentFactory` + `@tool`
- **Ch6** — 中心编排（三种模式见上表）

详细说明见 `supervisor/README.md`、`fixed_graph/README.md`。
