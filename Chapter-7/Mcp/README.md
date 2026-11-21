# Chapter-7 MCP (Streamable HTTP) / 第 7 章 MCP（Streamable HTTP）

## English

This module covers three MCP scenarios:

1. Expose the Chapter-6 hotel tool as a local MCP Server.
2. Connect an Agent to WeatherAPI's external MCP and reuse the Chapter-6 WeatherAgent definition.
3. Discover Baidu Map MCP tools and let the reused Chapter-6 HotelAgent choose tools autonomously.

### Files

| File | Responsibility |
|---|---|
| `server.py` | MCP Server exposing `recommend_hotel_tool` |
| `client.py` | MCP client helpers used by the Notebook |
| `settings.py` | Server host, port, client URL, and dynamic-port configuration |
| `external_mcp.py` | Shared helpers for external stdio MCP sessions and complete output |
| `paths.py` | Cross-chapter import-path and environment bootstrap |

### Setup

```bash
cd Chapter-7/Mcp
uv pip install -r requirements.txt
```

Configure model and map credentials in the repository-root `.env`. WeatherAPI and Baidu Map examples require Node.js and `npx`.

### Notebook

Open `Chapter-7/01_mcp_tool_calling.ipynb` and execute the cells in order. Part I starts and stops the MCP Server automatically; no separate terminal is required.

The script entry remains available for manual server startup with `python server.py`.

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `HOTEL_MCP_HOST` | `127.0.0.1` | Server bind host |
| `HOTEL_MCP_PORT` | `8765` | Server bind port |
| `HOTEL_MCP_URL` | `http://127.0.0.1:8765/mcp` | Client endpoint |

### External MCP credentials

- `WEATHERAPI_KEY`: WeatherAPI key used by the external weather MCP. Apply at <https://www.weatherapi.com/signup.aspx>.
- `BAIDU_MAP_API_KEY`: Baidu Map key used by the external map MCP. Apply at <https://lbsyun.baidu.com/apiconsole/key>. The example also accepts the repository's existing `BAIDU_MAP_AK`.

Configure keys in the repository-root `.env`; never commit real keys.

### Chapter-6 reuse

The external examples keep the Chapter-6 domain roles and system prompts. WeatherAPI tools are bound to `WeatherAgent`; Baidu Map tools are passed to `HotelAgent`, which selects tools from their MCP descriptions.

### Extension

Configure VariFlight Aviation MCP for the Chapter-6 `FlightAgent`. Use `VARIFLIGHT_MCP_URL`, `VARIFLIGHT_API_KEY`, and the `X-API-Key` header. Apply for a key at <https://ai.variflight.com/keys>.

### Architecture

```text
Agent / MCP Client
        │ Streamable HTTP /mcp
        ▼
server.py
    └── recommend_hotel_tool
          → Chapter-6 recommend_hotel
          → Chapter-5 search_hotels
```

---

## 中文

本章包含三类 MCP 使用场景：

1. 将 Chapter-6 的酒店工具发布为本地 MCP Server。
2. 让 Agent 接入 WeatherAPI 外部 MCP，并复用 Chapter-6 的 WeatherAgent 定义。
3. 发现百度地图 MCP 工具，由复用自 Chapter-6 的 HotelAgent 自主选择工具。

### 文件说明

| 文件 | 作用 |
|---|---|
| `server.py` | 提供 `recommend_hotel_tool` 的 MCP Server |
| `client.py` | Notebook 使用的 MCP Client 函数 |
| `settings.py` | Server 地址、端口、Client URL 和动态端口配置 |
| `external_mcp.py` | 外部 stdio MCP 会话与完整输出辅助函数 |
| `paths.py` | 跨章节导入路径与环境变量引导 |

### 安装

```bash
cd Chapter-7/Mcp
uv pip install -r requirements.txt
```

模型和地图凭据统一配置在仓库根目录 `.env`。WeatherAPI 和百度地图示例需要安装 Node.js，并通过 `npx` 启动 MCP。

### Notebook

打开 `Chapter-7/01_mcp_tool_calling.ipynb` 并按顺序执行。Part I 会自动启动和关闭 MCP Server，无需另开终端。

`server.py` 仍可用于手工启动脚本版服务。

### 配置

| 变量 | 默认 | 说明 |
|---|---|---|
| `HOTEL_MCP_HOST` | `127.0.0.1` | Server 绑定地址 |
| `HOTEL_MCP_PORT` | `8765` | Server 绑定端口 |
| `HOTEL_MCP_URL` | `http://127.0.0.1:8765/mcp` | Client 访问地址 |

### 外部 MCP 凭据

- `WEATHERAPI_KEY`：WeatherAPI 外部 MCP 使用的 Key，申请地址：<https://www.weatherapi.com/signup.aspx>。
- `BAIDU_MAP_API_KEY`：百度地图外部 MCP 使用的 Key，申请地址：<https://lbsyun.baidu.com/apiconsole/key>。示例也兼容仓库现有的 `BAIDU_MAP_AK`。

Key 配置在仓库根目录 `.env`，请勿提交真实 Key。

### Chapter-6 复用

外部 MCP 示例保留 Chapter-6 的领域角色和 System Prompt。WeatherAPI 工具绑定给 `WeatherAgent`；百度地图工具交给 `HotelAgent`，由模型根据 MCP 工具描述自主选择。

### 扩展练习

为 Chapter-6 的 `FlightAgent` 配置飞常准 Aviation MCP。使用 `VARIFLIGHT_MCP_URL`、`VARIFLIGHT_API_KEY` 和 `X-API-Key` 请求头。Key 申请地址：<https://ai.variflight.com/keys>。

### 架构

```text
Agent / MCP Client
        │ Streamable HTTP /mcp
        ▼
server.py
    └── recommend_hotel_tool
          → Chapter-6 recommend_hotel
          → Chapter-5 search_hotels
```
