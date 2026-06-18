# @agent-platform/demo-web

最小 Web Demo：**Vite + `@agent-platform/router-client`**，展示同步 `route()` 与 SSE `routeStream()` 时间线。

## 前置

1. 启动 platform API（默认 `8780`）：

```bash
cd Chapter-8-20260616
python scripts/run_api.py
```

2. 安装并启动 Demo（`5173`）：

```bash
cd packages/demo-web
npm install
npm run dev
```

浏览器打开 http://127.0.0.1:5173 。开发模式下 Vite 将 `/v1/*` 代理到 `8780`，**API Base URL 留空**即可。

## 环境变量

| 变量 | 说明 |
|------|------|
| `VITE_API_BASE_URL` | 可选；不设则使用当前页面 origin（配合 proxy） |

生产构建直连 API：

```bash
VITE_API_BASE_URL=http://127.0.0.1:8780 npm run build
npm run preview
```

## 功能

- Query + 可选 `domain` / `X-API-Key`
- 勾选 **SSE 流式**：左侧 Router 阶段时间线（`router.*` / `handoff.*` / `final`）
- 取消勾选：同步 `POST /v1/chat`

## 相关

- [router-client README](../router-client/README.md)
- [sdk_integration.md](../../docs/sdk_integration.md) — 三步联调
- [router_engine.md](../../docs/router_engine.md)
- `python scripts/smoke_router_client.py` — SDK 联调冒烟
