# Chapter 7: Agent Collaboration Protocols

This chapter answers one question: **how do system boundaries change when agents
and tools no longer run in the same process?**

Hotel domain capability is not reimplemented here. Chapter 5 provides hotel
search, Chapter 6 provides `HotelAgent`, and the repository root `llm.py`
centralizes model configuration. Chapter 7 compares the following approaches on
top of that foundation:

| Approach | What is exposed | What the caller sees | Code |
|---|---|---|---|
| Chapter 5 local tools | Python `@tool` | In-process function calls | Reused |
| MCP | Tools | Tool list, input schemas, JSON results | `Mcp/` |
| A2A | Agents | Agent Card, messages, task states, artifacts | `A2A/` |
| Supervisor | Multi-agent orchestration | Handoff to remote A2A agents | `A2A/supervisor.py` |

## Two Notebooks

1. [`01_mcp_tool_calling.ipynb`](01_mcp_tool_calling.ipynb)
   - Exposes tools through an MCP server.
   - Discovers tools from the client side.
   - Lets an agent invoke remote tools automatically.
2. [`02_a2a_agent_collaboration.ipynb`](02_a2a_agent_collaboration.ipynb)
   - Discovers Agent Cards and remote agents.
   - Packages Planner and Hotel as independent services.
   - Coordinates Planner and Hotel with a LangGraph Supervisor.
   - Includes an extension exercise for registering additional A2A agents and
     sharing your Agent Card with classmates or on a reachable host.

Notebooks are not placed in the subdirectories.

## Quick Start

```powershell
uv pip install -r Chapter-7/Mcp/requirements.txt
uv pip install -r Chapter-7/A2A/requirements.txt

# MCP
# Open Chapter-7/01_mcp_tool_calling.ipynb.
# The notebook starts and stops the inline-managed MCP server.

# A2A: running 02_a2a_agent_collaboration.ipynb directly is recommended.
# To run the services in separate terminals:
cd Chapter-7/A2A
python server.py --agent-card agent_cards/planner_agent.json --port 9011
python server.py --agent-card agent_cards/hotel_agent.json --port 9012
```

---

# Chapter-7：Agent 协作协议

本章只回答一个问题：**当 Agent 和工具不再运行在同一个进程里，系统边界如何变化？**

酒店领域能力不是本章重新实现的重点。Chapter-5 已提供酒店检索，Chapter-6
已提供 `HotelAgent`，根目录 `llm.py` 已统一模型配置。Chapter-7 在此基础上比较：

| 方式 | 暴露对象 | 调用方看到的内容 | 本章代码 |
|---|---|---|---|
| Chapter-5 本地工具 | Python `@tool` | 同进程函数调用 | 复用 |
| MCP | 工具 | 工具列表、输入 Schema、JSON 结果 | `Mcp/` |
| A2A | 智能体 | Agent Card、消息、任务状态、产物 | `A2A/` |
| Supervisor | 多智能体编排 | handoff 到远程 A2A Agent | `A2A/supervisor.py` |

## 两个 Notebook

1. [`01_mcp_tool_calling.ipynb`](01_mcp_tool_calling.ipynb)
   - MCP Server 暴露工具
   - Client 发现工具
   - Agent 自动调用远程工具
2. [`02_a2a_agent_collaboration.ipynb`](02_a2a_agent_collaboration.ipynb)
   - Agent Card 与远程 Agent 发现
   - Planner 与 Hotel 的独立服务封装
   - LangGraph Supervisor 对 Planner 与 Hotel 的协作调度
   - 扩展练习：注册更多 A2A 智能体，并将自己的 Agent Card 分享给同学或部署到可访问的主机

子目录不再放置 Notebook。

## 快速启动

```powershell
uv pip install -r Chapter-7/Mcp/requirements.txt
uv pip install -r Chapter-7/A2A/requirements.txt

# MCP
# 打开 Chapter-7/01_mcp_tool_calling.ipynb
# Notebook 会启动并关闭内联管理的 MCP Server

# A2A：推荐直接运行 02_a2a_agent_collaboration.ipynb
# 分终端运行时可使用：
cd Chapter-7/A2A
python server.py --agent-card agent_cards/planner_agent.json --port 9011
python server.py --agent-card agent_cards/hotel_agent.json --port 9012
```
