# Hotel Recommendation Agent

按城市与偏好推荐酒店（百度 Place API / Place Pro 多维检索；未配置 `BAIDU_MAP_AK` 时回退高德）。

## 环境

```bash
conda activate agent-systems-book
pip install -r requirements.txt
```

| 配置项 | 位置 | 说明 |
|--------|------|------|
| `DEPLOYMENT_NAME` / `CHAT_API_KEY` / `CHAT_ENDPOINT` | 本目录 `.env` 或书根 `.env` | LLM（如 DashScope 兼容模式） |
| `BAIDU_MAP_AK` | 书根 `.env` | 百度地图 POI；未配置则回退高德 |

本地快速验证 Agent（不经过 A2A 协议）：

```bash
python agent.py
```

## 启动 A2A 服务

在本目录执行：

```bash
python server.py --host 127.0.0.1 --port 9012
```

成功后会监听 `http://127.0.0.1:9012/`，Agent Card 在：

`http://127.0.0.1:9012/.well-known/agent-card.json`

## 测试 A2A 是否连通

使用上级目录的 **`check_a2a_call.py`** 作为独立客户端，验证：拉取 Agent Card → 发送消息 → 流式收回复 → 任务进入终态。

**前提**：`server.py` 已在目标地址运行。

```powershell
# 终端 1：启动服务（见上一节）

# 终端 2：连通性测试（从本 Agent 目录）
cd ..
python check_a2a_call.py --base-url http://127.0.0.1:9012/
```

或从书根目录：

```powershell
python Chapter-7\A2A\check_a2a_call.py --base-url http://127.0.0.1:9012/
```

### 常用参数

```powershell
# 自定义查询
python check_a2a_call.py --query "推荐一个大同安静亲子的酒店，预算500/晚"

# 打印耗时（首 token、总耗时）
python check_a2a_call.py --timing

# 远程 Agent
python check_a2a_call.py --base-url http://10.112.57.99:9012/ --timeout 120
```

### 预期输出

连通正常时大致会看到：

```
OK: agent-card.json fetched
agent: HotelRecommendationAgent
task_state: TASK_STATE_SUBMITTED
...（流式酒店推荐正文）...
task_state: TASK_STATE_COMPLETED
OK: task finished (stream)
```

退出码：`0` 成功；`2` 连接/调用失败；`3` 超时未完成。

### 常见问题

| 现象 | 处理 |
|------|------|
| `create_client error` / 连接拒绝 | 确认 `server.py` 已启动，`--base-url` 与 `--host`/`--port` 一致 |
| 端口被占用 `[Errno 10048]` | 换端口启动：`server.py --port 9013`，客户端 `--base-url http://127.0.0.1:9013/` |
| 长时间无输出后超时 | 检查 LLM Key；可加 `--timeout 180` |
| `hotels` 为空 | 检查书根 `.env` 中 `BAIDU_MAP_AK` |

## LangGraph 调度（supervisor_local.py）

使用 **`langgraph_supervisor.create_supervisor`** 作为调度器，远程 A2A 智能体在 `a2a_agents.py` 的 `A2A_AGENT_SPECS` 中注册（默认仅 hotel）：

```python
A2A_AGENT_SPECS = [
    ("hotel_agent", "http://127.0.0.1:9012/", "酒店推荐（A2A）"),
    # ("weather_agent", "http://127.0.0.1:9013/", "天气查询（A2A）"),
]
```

**前提**：`server.py` 已启动；Supervisor 自身需要 LLM（`CHAT_API_KEY` / `DASHSCOPE_API_KEY`）。

```powershell
# 终端 1：A2A 服务
python server.py --host 127.0.0.1 --port 9012

# 终端 2：Supervisor 调度
cd ..
C:\Users\zhanghong26\.conda\envs\agent-systems-book\python.exe supervisor_local.py
```

Supervisor 会根据用户意图 handoff 到 `hotel_agent` 等远程 A2A 节点；增删 `A2A_AGENT_SPECS` 即可扩展多个 A2A 服务。

依赖：`pip install langgraph-supervisor`（Chapter-6 requirements 已含）。
