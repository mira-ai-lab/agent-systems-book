# Chapter-7 酒店推荐 MCP（HTTP/SSE）

## 文件说明

| 文件 | 作用 |
|------|------|
| `hotel_core.py` | 酒店 POI 查询（复用 Chapter-6 `travel_common`，无 LangChain） |
| `hotel_tools.py` | LangChain `@tool recommend_hotel` + Agent |
| `hotel_mcp_server.py` | **MCP Server（HTTP/SSE）**，对外暴露两个工具 |
| `hotel_mcp_client.py` | MCP SSE 客户端，供其他 Agent 远程调用 |
| `hotel_recommendation_demo.py` | Agent 绑定 **MCP 工具**（非本地 @tool）完整演示 |

## 安装

```bash
cd Chapter-7
pip install -r requirements.txt
```

书根目录 `.env` 需配置 `DASHSCOPE_API_KEY`；地图可选 `BAIDU_MAP_AK`。

## 运行

**第一步：启动 MCP 服务（需常驻）**

```bash
python hotel_mcp_server.py
# 默认监听 http://127.0.0.1:8765/sse
```

**第二步：客户端 / 演示**

```bash
python hotel_mcp_client.py            # 直接调 MCP 工具 / Agent
python hotel_recommendation_demo_mcp.py   # Agent 绑定 MCP 工具（需服务已启动）
```

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `HOTEL_MCP_HOST` | `127.0.0.1` | SSE 服务绑定地址 |
| `HOTEL_MCP_PORT` | `8765` | SSE 服务端口 |
| `HOTEL_MCP_SSE_URL` | `http://127.0.0.1:8765/sse` | 客户端连接地址 |

## MCP 工具

1. **`recommend_hotel_tool`** — 查酒店列表（JSON），等价于 LangChain `@tool recommend_hotel`
2. **`hotel_agent_query`** — 传入自然语言，内部跑 LangChain Agent 并返回推荐文案

## 在 Cursor 中配置 MCP（SSE）

先启动 `python hotel_mcp_server.py`，再在 `.cursor/mcp.json` 或 Cursor Settings → MCP：

```json
{
  "mcpServers": {
    "hotel-recommendation": {
      "url": "http://127.0.0.1:8765/sse"
    }
  }
}
```

`.env` 会从书根目录自动加载。

## 架构

```
用户 / Cursor / 其他 Agent
         │  HTTP GET /sse  +  POST /messages/
         ▼
hotel_mcp_server.py  (FastMCP + uvicorn)
    ├── recommend_hotel_tool  →  hotel_core.recommend_hotel_impl  →  travel_common
    └── hotel_agent_query     →  hotel_tools.run_hotel_agent      →  LangChain Agent
```

相比 stdio 模式，HTTP/SSE 适合：
- 服务独立部署、多客户端共享
- 远程调用（改 `HOTEL_MCP_HOST` 绑定 `0.0.0.0` 并配置防火墙）
- 与 Cursor / Claude Desktop 的 URL 型 MCP 配置对齐
