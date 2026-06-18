# SDK 本地联调指南（Phase 26.6）

三步跑通 **Platform API → router-client → Demo Web**，验证同步 `route()`、SSE `routeStream()` 与异步 `submitJob()`。

## 前置

| 依赖 | 版本 |
|------|------|
| Python | ≥ 3.10 |
| Node.js | ≥ 18 |
| npm | 随 Node 安装 |

```bash
cd Chapter-8-20260616
pip install -e ".[api,dev]"
pip install -e domains/
```

可选：复制 `.env.example` 为 `.env` 并配置 LLM Key（真实推理时需要）。联调冒烟使用 mock 或短 query 即可。

---

## 第一步：启动 Platform API

```bash
cd Chapter-8-20260616
python scripts/run_api.py
```

默认监听 **http://127.0.0.1:8780**。

验证：

```bash
curl http://127.0.0.1:8780/health
# {"status":"ok",...}
```

若配置了 `API_KEYS`，后续请求需加 `-H "X-API-Key: your-key"`。

---

## 第二步：router-client 冒烟

对已运行的 API 执行 Node integration 测试（`route` + `routeStream` + `submitJob`）：

```bash
python scripts/smoke_router_client.py
```

或手动：

```bash
cd packages/router-client
npm install
npm run build
ROUTER_CLIENT_BASE_URL=http://127.0.0.1:8780 npm run test:integration
```

**自动化（无需真实 LLM）**：pytest 内嵌 uvicorn 临时服务 + mock 编排器：

```bash
pytest tests/test_phase26_router_client_integration.py -q
```

---

## 第三步：Demo Web

```bash
# 终端 1：API 保持运行（见第一步）
# 终端 2：
cd packages/demo-web
npm install
npm run dev
```

浏览器打开 **http://127.0.0.1:5173**。

| 项 | 说明 |
|----|------|
| API Base URL | **留空**（Vite dev 将 `/v1/*` 代理到 `:8780`） |
| Query | 如 `退货政策是什么？` |
| Domain | 默认 `customer_service` |
| SSE 流式 | 勾选后左侧显示 Router 阶段时间线 |

生产构建直连 API：

```bash
VITE_API_BASE_URL=http://127.0.0.1:8780 npm run build
npm run preview
```

---

## 快速对照

| 能力 | HTTP | router-client |
|------|------|---------------|
| 同步问答 | `POST /v1/chat` | `client.route(query)` |
| SSE 流式 | `POST /v1/chat/stream` | `client.routeStream(query)` |
| 异步任务 | `POST /v1/jobs` | `client.submitJob(query)` |
| 任务状态 | `GET /v1/jobs/{id}` | `client.getJob(jobId)` |

TypeScript 示例：

```typescript
import { createRouterClient } from "@agent-platform/router-client";

const client = createRouterClient({ baseUrl: "http://127.0.0.1:8780" });

const chat = await client.route("退货政策是什么？", { domain: "customer_service" });
console.log(chat.final_response);

const job = await client.submitJob("规划上海苏州杭州七日游", { domain: "travel" });
const status = await client.getJob(job.job_id);
console.log(status.status);
```

---

## CI 对齐

GitHub Actions `chapter8-upgrade-ci.yml` 包含：

- **test** job：`sync_package_versions.py --check` + `pytest`
- **router-client** job：`npm test` + `test_phase26_router_client_integration.py`
- **demo-web** job：`npm run build`

本地等效：

```bash
python scripts/sync_package_versions.py --check
pytest -q
cd packages/router-client && npm test
cd packages/demo-web && npm run build
```

---

## 发版与版本同步（26.4）

以下包共用 **同一 semver**（当前 `0.21.0`）：

| 包 | 版本文件 |
|----|----------|
| `agent-platform` | `pyproject.toml` |
| `@agent-platform/router-client` | `packages/router-client/package.json` |
| `@agent-platform/demo-web` | `packages/demo-web/package.json` |

> `domains/pyproject.toml`（`agent-platform-domains-builtin`）独立版本，不在此同步范围。

校验：

```bash
python scripts/sync_package_versions.py --check
```

发版 bump（以 `pyproject.toml` 为准对齐 npm 包，或指定版本）：

```bash
python scripts/sync_package_versions.py --sync 0.22.0
# 或仅对齐到 pyproject 当前值：
python scripts/sync_package_versions.py --sync
```

改版本后建议在 `packages/router-client` 与 `packages/demo-web` 执行 `npm install` 刷新 lockfile。

---

## 故障排查

| 现象 | 处理 |
|------|------|
| `ECONNREFUSED 8780` | 确认 `python scripts/run_api.py` 已启动 |
| `401 / 403` | 检查 `API_KEYS` 与 `X-API-Key` / Demo 页 API Key 输入 |
| Demo 页 CORS 错误 | dev 模式请 **留空 Base URL**，走 Vite proxy |
| `npm` 在 Windows 报 FileNotFound | 使用 `shell=True` 或 PowerShell 直接 `npm test` |
| integration 跳过 | 需设置 `ROUTER_CLIENT_BASE_URL` |

---

## 相关文档

- [router-client README](../packages/router-client/README.md)
- [demo-web README](../packages/demo-web/README.md)
- [router_engine.md](router_engine.md)
- [UPGRADE.md](../UPGRADE.md) Phase 26
