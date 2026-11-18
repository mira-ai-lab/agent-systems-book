# Chapter-6: 旅行多智能体系统

整合 **Chapter-2 / 3 / 4 / 5** 的能力，由一个中心控制器完成请求分诊、任务规划、领域智能体路由和分层调度。

本章的工具均为本地函数或普通 HTTPS API，不涉及 MCP 工具调用；天气和航班工具当前返回空结果。MCP 与 A2A 的远程协作方式在 Chapter-7 介绍。

## 目录结构

```
Chapter-6/
├── config.py              # 路径与 .env 引导（唯一路径来源）
├── domain_agents.py       # 6 个领域智能体
├── task_plan.py           # Ch2 预调查 + Ch4 拆解 + 依赖校验
├── agent_routing.py       # 子任务到领域智能体的能力路由
├── memory_adapter.py      # Chapter-3 记忆能力适配
├── orchestrator.py        # 依赖调度、最终结果选择与中心编排入口
└── pyproject.toml
```

## 安装

```bash
cd Chapter-6
pip install -e .
```

## 运行

中心化编排入口：

```bash
python orchestrator.py
```

教学入口为 `central_orchestration.ipynb`。中心智能体先判断请求类型：非旅行请求直接回复；旅行请求再由代码依次完成预调查、记忆检索、任务拆解、依赖分析、领域智能体路由、分层执行和最终规划。

## 环境配置

书仓库根目录或 `Chapter-6/.env`：

- `DASHSCOPE_API_KEY` — 百炼大模型
- `AMAP_KEY` / `BAIDU_MAP_AK` — 地图 POI（可选）

天气工具当前返回空结果，`data_source` 为 `not_integrated`；实时天气能力由后续章节补充。

Chroma 向量库统一目录：`Chapter-6/chroma_memory/`（`config.CHROMA_DIR`）

## 架构

```
用户请求
  → 中心智能体请求分诊
      ├─ 非旅行请求 → 直接回复
      └─ 旅行请求
  → [Ch2] 思维链预调查
  → [Ch3] 长期记忆检索
  → [Ch4] 任务拆解 → 依赖排序
  → 智能体路由 → 天气/景点/酒店/餐厅/航班信息获取
  → ItineraryAgent 汇总并生成最终日程
  → [Ch3] 写入记忆
```

## 领域智能体

WeatherAgent · AttractionAgent · HotelAgent · RestaurantAgent · FlightAgent · ItineraryAgent

领域智能体的名称、工具、提示词和创建方式统一由 `DOMAIN_AGENT_SPECS` 声明；缓存创建函数 `get_domain_agent()` 和能力目录 `DomainAgentRegistry` 都从这里派生，避免重复维护。

中心编排 Prompt 负责请求分诊和非旅行问题直答；旅行请求进入后续代码控制的规划、路由和执行流程。天气、景点、酒店、餐厅和航班结果最终统一交给 ItineraryAgent，由它完成日程规划与总结。

## 与各章对应

- **Ch2** — 复用 `Chapter-2/fact_survey.py` 的四类事实调查结构、Prompt 和解析器
- **Ch3** — 复用 `MemoryStore`、`MemoryRetriever`、`MemoryContextBuilder` 完成记忆写入、混合检索和短期上下文
- **Shared** — 根目录 `llm.py` 提供同步/异步模型调用
- **Ch4** — 从 `Chapter-4/task_decomposition.py` 与 `dependency_analysis.py` 导入任务拆解和依赖分析能力
- **Ch5** — 复用 `map_api.py` 的 POI 归一化，以及 `hotel_search.py`、`restaurant_search.py`、`attraction_search.py` 的领域搜索
- **Ch6** — `agent_routing.py` 独立完成子任务到领域智能体的匹配，中心控制器按代码顺序完成分诊、规划、路由和执行
