# Chapter-6: 中心智能体系统

整合 **Chapter-2 / 3 / 4 / 5** 的能力，构建完整的旅行规划中心智能体。

## 架构

```
用户请求
  → [Ch2] 思维链预调查（四类事实）
  → [Ch3] 长期记忆向量检索
  → [Ch4] 任务拆解 → 依赖排序
  → [Ch6] 子任务路由到子智能体
  → [Ch5+] 6 个子智能体 LangChain Agent 执行
  → 聚合生成最终回复
  → [Ch3] 写入新记忆
```

## 文件说明

| 文件 | 来源章节 | 职责 |
|------|----------|------|
| `prompts.py` | Ch2/3/4/6 | 预调查、记忆、拆解、依赖、路由、中心 system prompt |
| `memory_system.py` | Ch3 | Chroma 向量检索 + 短期对话缓冲 |
| `task_planner.py` | Ch2 + Ch4 | 预调查 → 拆解 → 依赖 → Agent 路由 |
| `sub_agents.py` | Ch5 扩展 | 6 个专业子智能体（LangChain Agent + Tool） |
| `central_orchestrator.py` | Ch6 | 中心编排器 |
| `central_agent_demo.ipynb` | 演示 | Jupyter 交互演示 |

## 子智能体团队（由 Chapter-5 HotelAgent 扩展）

- **WeatherAgent** — 天气查询
- **AttractionAgent** — 景点推荐
- **HotelAgent** — 酒店推荐（保留 Ch5 地图关键词/主观偏好分离逻辑）
- **RestaurantAgent** — 美食推荐
- **FlightAgent** — 航班查询
- **ItineraryAgent** — 行程规划

## 环境配置

```bash
cd Chapter-6
pip install -r requirements.txt
```

项目根目录 `.env` 需配置：
- `DASHSCOPE_API_KEY` — 百炼大模型 + 嵌入
- `BAIDU_MAP_AK` — 酒店/景点/餐厅（可选）

## 运行

```bash
python central_orchestrator.py
```

或打开 `central_agent_demo.ipynb`，从第 0 节顺序运行。

## 与各章对应关系

### Chapter-2 思维链
- `FACTS_PROMPT` 独立预调查步骤
- 输出四类事实：已给出 / 需查阅 / 需推导 / 有根据猜测

### Chapter-3 长期记忆
- `LongTermMemory.search_memories()` 真实向量检索
- `build_prompt()` 注入记忆上下文
- 对话结束后 `ingest()` 写入偏好

### Chapter-4 任务拆解
- `PROMPT_TP_ZH` 按 Agent 团队能力拆解子任务
- `DEPENDENCY_SYSTEM_PROMPT_ZH` 分析 input/output 依赖并排序
- `parse_decomposition_response()` 解析 `# 目标` / `# 任务拆解`

### Chapter-5 工具调用
- 每个子智能体 = `create_agent` + `@tool` + 专属 system prompt
- 中心编排器通过 `SubAgentFactory` 调用 Agent（非直接 bypass 工具）

### Chapter-6 路由
- `AGENT_ROUTING_PROMPT` 为每个子任务选择 agent + params
- 同层无依赖子任务 `asyncio.gather` 并行执行
