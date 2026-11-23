# A2A Server：将子 Agent 暴露为远程服务（Phase 7D）

与 Phase 7C（A2A Client / Supervisor handoff 调用远程）对称，平台可将 **DomainPlugin registry 中的子 Agent** 以标准 A2A HTTP 服务对外暴露。

## 依赖

```bash
pip install "agent-platform[a2a]"
pip install -e domains/
```

## CLI 启动

```bash
# demo 领域 EchoAgent → http://127.0.0.1:9012/
python -m agent_framework.transport.a2a.server \
  --domain demo \
  --agent EchoAgent \
  --host 127.0.0.1 \
  --port 9012

# 或使用 Supervisor 节点名
python -m agent_framework.transport.a2a.server \
  --domain travel \
  --node-name hotel_agent \
  --port 9012
```

Agent Card：`http://127.0.0.1:9012/.well-known/agent-card.json`

## SDK 编程式启动

```python
from agent_framework.transport.a2a.server import serve_sub_agent

serve_sub_agent("travel", registry_agent="HotelAgent", host="127.0.0.1", port=9012)
```

仅构建 Starlette app（测试 / 自定义 uvicorn）：

```python
from agent_framework.transport.a2a.server import create_sub_agent_a2a_app

app = create_sub_agent_a2a_app("demo", registry_agent="EchoAgent", port=9013)
```

## 与 Supervisor mixed 联调

```bash
# 终端 1：暴露酒店 Agent
python -m agent_framework.transport.a2a.server --domain travel --agent HotelAgent --port 9012

# 终端 2：Supervisor 以 A2A 调用
export TRAVEL_A2A_HOTEL_URL=http://127.0.0.1:9012/
python scripts/run_api.py

curl -X POST http://127.0.0.1:8780/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"domain":"travel","mode":"supervisor","transport":"mixed","query":"上海酒店推荐"}'
```

## 架构

```
HTTP JSON-RPC
    → DefaultRequestHandler
    → RegistrySubAgentExecutor
    → registry.get_agent(factory).ainvoke(...)
    → TaskArtifactUpdateEvent（响应文本）
```

- `context_id` 对齐 LangGraph `thread_id`（多轮对话）
- 入站请求经 `TraceContextMiddleware` 提取 W3C `traceparent`，与 Phase 8 Client inject 形成跨服务 trace

书稿参考实现：`Chapter-7/A2A/hotel_recommendation_agent/`（流式 + 自定义 Agent 类）；本平台 Server 直接复用 registry 已编译的 LangChain Agent。
