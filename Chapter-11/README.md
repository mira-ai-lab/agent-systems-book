# travel-agent：端到端可用的旅行智能体

本章把第 8 章的旅行多智能体系统发布为一个可被编码智能体直接调用的服务（以 stdio MCP 作为通信协议），包装
`travel_multi_agent.orchestration.fixed_graph.LangGraphOrchestrator`，对外暴露
`ask_travel_agent` / `ask_travel_agent_detailed` 两个工具。

## 前置：先启动 5 个 A2A 子智能体服务

新第 8 章的中心编排器通过 A2A 调用子智能体，所以 MCP 进程被调用前，需要先跑起
5 个领域智能体服务（端口 9031~9035）。最简单的方式是运行
`Chapter-8/travel_multi_agent.ipynb` 的「启动 5 个 A2A 子智能体服务」Cell，
或逐个执行：

    python -m travel_multi_agent.a2a.server --agent-card ... --port 9031 ...

## Claude Code / Codex 注册

把 `.mcp.json` 里的 `command` 换成你的 python 绝对路径，然后在项目根（或
`Chapter-11`）启动 Claude Code，它会读取 `.mcp.json` 并注册 `travel-agent`。
Codex 同样走 MCP，配置方式等价。

## 环境变量

- `DASHSCOPE_API_KEY`（或 `OPENAI_API_KEY`）：LLM，必填
- `TRAVEL_MCP_TRANSPORT`：`stdio`（默认）| `sse`
- `TRAVEL_MCP_HOST` / `TRAVEL_MCP_PORT`：SSE 模式用
- `TRAVEL_MCP_ENABLE_MEMORY`：`1`（默认）/`0`