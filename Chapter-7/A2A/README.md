# Chapter-7 A2A

本章把第 6 章的任务规划与酒店能力封装成两个独立的 A2A 服务：

- `TravelPlannerAgent`：复用第 6 章事实调查、任务拆解和依赖分析；
- `HotelAgent`：复用第 6 章领域智能体与酒店检索工具；
- `Supervisor`：从零构建 LangGraph Supervisor，并通过 handoff 调用远程 A2A Agent。

本目录默认把两个服务启动在同一台设备上，以降低教学复现成本。正式部署时，
服务可以由第三方长期托管，调用方只需配置 URL，无需与 Agent 实现共享进程或
内部工具接口。

配套 Notebook：
[`02_a2a_agent_collaboration.ipynb`](../02_a2a_agent_collaboration.ipynb)。

## 代码结构

| 文件或目录 | 作用 |
|---|---|
| `agent_cards/` | 两个 Agent 的公开定义，使用标准 Agent Card JSON |
| `agents.py` | Planner 与 Hotel 的业务实现，以及规划能力适配 |
| `server.py` | 公共 Executor、Agent Card 加载、实现选择与 HTTP 装配 |
| `a2a_client.py` | Agent Card 发现、流式调用与终态校验 |
| `a2a_models.py` | Planner 请求/响应模型与 Agent Card 能力转换 |
| `supervisor.py` | 从零构建 LangGraph Supervisor，并通过 handoff 调用远程 A2A Agent |
| `process_manager.py` | Notebook 使用的同机服务生命周期管理 |
| `settings.py` | 服务端口与 URL 配置 |

新增 Agent 时只需：

1. 在 `agents.py` 中实现业务 Agent，并提供可被 `GenericAgentExecutor` 调用的执行方法；
2. 新增该 Agent 的 Card JSON；
3. 在 `server.py` 的 `get_agent()` 映射中增加一行。

协议装配、流式映射、HTTP 路由和 CLI 无需重复实现。

## Notebook 内运行

打开 `02_a2a_agent_collaboration.ipynb` 并依次运行单元格即可。Notebook 会：

1. 为 Planner 与 Hotel 分配空闲端口；
2. 启动两个同机 A2A 服务；
3. 发现并展示 Agent Card；
4. 将远程 Agent 注册给 LangGraph Supervisor；
5. 通过 Supervisor 调用 Planner 与 Hotel；
6. 停止本地教学服务。

## 分终端运行

```powershell
uv pip install -r Chapter-7/A2A/requirements.txt

cd Chapter-7/A2A
python server.py --agent-card agent_cards/planner_agent.json --port 9011
python server.py --agent-card agent_cards/hotel_agent.json --port 9012
```

外部托管服务通过以下变量替换本地地址：

```dotenv
A2A_PLANNER_URL=https://planner.example.com/
A2A_HOTEL_URL=https://hotel.example.com/
```
