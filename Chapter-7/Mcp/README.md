# Chapter-7 MCP (Streamable HTTP) / 第 7 章 MCP（Streamable HTTP）

## English

This module demonstrates three complementary MCP scenarios:

1. Expose an existing Chapter-6 hotel tool as a Streamable HTTP MCP Server.
2. Connect a local Agent to the official WeatherAPI MCP and reuse Chapter-6's
   WeatherAgent definition.
3. Discover Baidu Map MCP tools and let a reused Chapter-6 HotelAgent choose
   tools autonomously.

### Files

| File | Responsibility |
|---|---|
| `server.py` | Streamable HTTP MCP Server exposing `recommend_hotel_tool` |
| `client.py` | Low-level MCP client for tool discovery and invocation |
| `settings.py` | Shared server address, client URL, and dynamic-port configuration |
| `external_mcp.py` | Shared stdio MCP session and complete non-streaming output helpers |
| `paths.py` | Cross-chapter import-path and environment bootstrap |

### Installation

```bash
cd Chapter-7/Mcp
uv pip install -r requirements.txt
```

Configure `DASHSCOPE_API_KEY` or `OPENAI_API_KEY` in the repository-root
`.env`. For hotel search, configure either `BAIDU_MAP_AK` or `AMAP_KEY`.
External WeatherAPI and Baidu Map examples also require Node.js and `npx`.

### Notebook usage

Open `Chapter-7/01_mcp_tool_calling.ipynb` and execute the cells in order.
Part I starts the MCP Server from `server.py` in a background thread and stops
it after the Agent example, so no separate terminal is required.

### Script usage

Start the server in the first terminal:

```bash
python server.py
# Default endpoint: http://127.0.0.1:8765/mcp
```

Run the low-level client in another terminal:

```bash
python client.py
```

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `HOTEL_MCP_HOST` | `127.0.0.1` | Server bind host |
| `HOTEL_MCP_PORT` | `8765` | Server bind port |
| `HOTEL_MCP_URL` | `http://127.0.0.1:8765/mcp` | Client endpoint |

### Internal MCP tool

`recommend_hotel_tool` exposes the Chapter-6 `recommend_hotel` tool through
MCP. It returns structured hotel candidates; the client-side Agent remains
responsible for producing the final recommendation.

### External MCP: WeatherAPI

- Package: `weatherapi-mcp`
- Repository: <https://github.com/weatherapicom/weatherapi-mcp>
- License: MIT
- Start command: `npx -y weatherapi-mcp`
- Required variable: `WEATHERAPI_KEY`
- Key registration: <https://www.weatherapi.com/signup.aspx>

Configure the key in the repository-root `.env`:

```text
WEATHERAPI_KEY=your_weatherapi_key
```

The Notebook reads the key with `os.getenv("WEATHERAPI_KEY")` and forwards it
to the external MCP subprocess. The example reuses Chapter-6
`DOMAIN_AGENT_SPECS["WeatherAgent"]` and only replaces its tool source.

### External MCP: Baidu Maps

- Repository: <https://github.com/baidu-maps/mcp>
- npm package: `@baidumap/mcp-server-baidu-map`
- License: MIT
- Key registration: <https://lbsyun.baidu.com/apiconsole/key>
- Official Node variable: `BAIDU_MAP_API_KEY`

The example also accepts the repository's existing `BAIDU_MAP_AK`. The
Notebook discovers all ten Baidu Map tools and passes them to the reusable
Chapter-6 `HotelAgent`; the model chooses tools based on their descriptions
rather than receiving a pre-bound `map_search_places` tool.

The ten tools cover geocoding, reverse geocoding, POI search, POI details,
route planning, distance matrix, weather, IP location, traffic, and POI
extraction.

### Extension: VariFlight

The extension exercise asks readers to configure VariFlight Aviation MCP for
the Chapter-6 `FlightAgent`, preserving the Chapter-6 role and system prompt
while replacing the tool source.

- Documentation: <https://ai.variflight.com/docs/aviation>
- Key registration: <https://ai.variflight.com/keys>
- Endpoint: `https://ai.variflight.com/servers/aviation/mcp`
- Authentication header: `X-API-Key`

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

## 文件说明

| 文件 | 作用 |
|------|------|
| `server.py` | **MCP Server（Streamable HTTP）**，只暴露一个酒店查询工具 |
| `client.py` | MCP Streamable HTTP 客户端，供其他 Agent 远程调用 |
| `settings.py` | Server 绑定地址与 Client 访问地址的统一配置 |
| `external_mcp.py` | 外部 stdio MCP 会话与完整非流式输出辅助函数 |
| `paths.py` | 跨章节导入路径与环境变量引导 |

## 安装

```bash
cd Chapter-7/Mcp
uv pip install -r requirements.txt
```

仓库根目录的 `.env` 需配置 `DASHSCOPE_API_KEY`（或 `OPENAI_API_KEY`）；
地图配置 `BAIDU_MAP_AK` 或 `AMAP_KEY` 之一。

## Notebook 运行方式

打开仓库根目录的 `01_mcp_tool_calling.ipynb` 并按顺序执行。Part I 的启动单元会在后台线程中启动 `server.py` 中定义的 MCP Server，结束单元会在 Agent 调用完成后关闭服务，无需另开终端。

## 脚本版运行方式

脚本版需要分终端启动。第一步启动 MCP 服务：

```bash
python server.py
# 默认监听 http://127.0.0.1:8765/mcp
```

第二步运行客户端：

```bash
python client.py                      # 直接调用 MCP 工具
```

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `HOTEL_MCP_HOST` | `127.0.0.1` | MCP 服务绑定地址 |
| `HOTEL_MCP_PORT` | `8765` | MCP 服务端口 |
| `HOTEL_MCP_URL` | `http://127.0.0.1:8765/mcp` | 客户端连接地址 |

## MCP 工具

**`recommend_hotel_tool`**：查酒店候选列表并返回 JSON。它是 Chapter-6
`recommend_hotel` 工具的 MCP 适配，不包含最终推荐 Agent。

## 调用外部 MCP

外部 WeatherAPI 和百度地图示例要求安装 Node.js，并可通过 `npx` 启动 MCP。

`Chapter-7/01_mcp_tool_calling.ipynb` 演示如何调用 WeatherAPI.com 官方 MCP：

- 包名：`weatherapi-mcp`
- 仓库：<https://github.com/weatherapicom/weatherapi-mcp>
- License：MIT
- 启动：`npx -y weatherapi-mcp`
- 环境变量：`WEATHERAPI_KEY`

运行本示例前，需要先在 WeatherAPI.com 申请 Key：

<https://www.weatherapi.com/signup.aspx>

将申请到的 Key 配置在仓库根目录的 `.env` 文件中：

```text
WEATHERAPI_KEY=your_weatherapi_key
```

Notebook 通过 `os.getenv("WEATHERAPI_KEY")` 从环境变量读取 Key，
并在启动 `weatherapi-mcp` 子进程时写入其 `env`。请勿将真实 Key
直接写入 Notebook 或提交到 Git。

该示例通过 stdio 连接外部 Server，并复用 Chapter-6 的
`DOMAIN_AGENT_SPECS["WeatherAgent"]` 角色与 System Prompt，只将工具来源
替换为外部 MCP。运行前需安装 Node.js，并提供有效的 `WEATHERAPI_KEY`。

Notebook 还演示百度地图官方 MCP：

- 仓库：<https://github.com/baidu-maps/mcp>
- npm 包：`@baidumap/mcp-server-baidu-map`
- License：MIT
- 申请 Key：<https://lbsyun.baidu.com/apiconsole/key>
- 认证环境变量：`BAIDU_MAP_API_KEY`；示例兼容现有 `BAIDU_MAP_AK`

百度地图 MCP 提供地址地理编码、地点搜索、POI 详情、路线规划、距离矩阵、
天气、IP 定位、实时路况和文本 POI 提取等 10 个工具。Notebook 会先发现
全部工具，再由复用自 Chapter-6 的 HotelAgent 根据用户请求自主选择工具。

课程扩展还要求读者参照天气 MCP 的接入方式，为 Chapter-6 的 `FlightAgent`
配置飞常准 Aviation MCP：

- 文档：<https://ai.variflight.com/docs/aviation>
- 申请 Key：<https://ai.variflight.com/keys>
- 端点：`https://ai.variflight.com/servers/aviation/mcp`
- 认证头：`X-API-Key`

其工具覆盖直飞航班、指定航班、联程方案、舒适度、实时飞机位置、机场天气、
行程摘要和结构化票价。接入时保持 Chapter-6 FlightAgent 的角色与 System
Prompt，仅替换 `tools=` 的来源。

## 在 Cursor 中配置 MCP

先启动 `python server.py`，再在 `.cursor/mcp.json` 或 Cursor Settings → MCP：

```json
{
  "mcpServers": {
    "hotel-recommendation": {
      "url": "http://127.0.0.1:8765/mcp"
    }
  }
}
```

`.env` 会从仓库根目录自动加载。

## 架构

```
用户 / Cursor / 其他 Agent
         │  Streamable HTTP POST /mcp
         ▼
server.py  (FastMCP + uvicorn)
    └── recommend_hotel_tool
          → Chapter-6 recommend_hotel
          → Chapter-5 search_hotels
```

相比 stdio 模式，Streamable HTTP 适合：
- 服务独立部署、多客户端共享
- 远程调用（改 `HOTEL_MCP_HOST` 绑定 `0.0.0.0` 并配置防火墙）
- 与 Cursor / Claude Desktop 的 URL 型 MCP 配置对齐
