# @agent-platform/router-client

agent-platform 企业路由引擎的最小 TypeScript/JavaScript 客户端（Phase 25 P3）。

对齐后端：

- `POST /v1/chat` — 同步 `route(query)`
- `POST /v1/chat/stream` — SSE `routeStream(query)`
- `POST /v1/jobs` — 异步 `submitJob(query)`（Phase 26.7）
- `GET /v1/jobs/{id}` — `getJob(jobId)`

## 安装（占位）

```bash
cd packages/router-client
npm install
npm run build
```

npm 发布占位（尚未实际上线 registry 时可本地 link）：

```bash
npm link
# 或在 monorepo 内 file: 引用
```

## 快速开始

```typescript
import { createRouterClient } from "@agent-platform/router-client";

const client = createRouterClient({
  baseUrl: "http://127.0.0.1:8780",
  apiKey: process.env.API_KEY, // 可选，对应服务端 API_KEYS
});

// 同步 route（推荐：仅 query + profile=auto）
const result = await client.route("规划杭州三日游", {
  domain: "travel",
  profile: "auto",
});
console.log(result.final_response);
console.log(result.routing_plan);
console.log(result.knowledge_matches);

// SSE 流式
for await (const event of client.routeStream("规划杭州三日游")) {
  if (event.type === "final") {
    console.log("done", event.data?.final_response);
  } else if (event.type.startsWith("router.")) {
    console.log("router stage", event.type);
  }
}
```

## 异步任务（26.7）

```typescript
const job = await client.submitJob("规划上海苏州杭州七日游", {
  domain: "travel",
  userId: "alice",
  profile: "auto",
});
console.log(job.job_id, job.status); // pending

const detail = await client.getJob(job.job_id);
console.log(detail.status, detail.result);
```

便捷函数：`submitJob(baseUrl, query, opts)` / `getJob(baseUrl, jobId, opts)`。

## 重试 / 超时（26.8）

客户端 HTTP 策略（默认：`timeoutMs=120_000`、`retries=2`、`retryDelayMs=500`，指数退避）：

```typescript
import {
  createRouterClient,
  isRouterClientError,
  isRouterClientTimeoutError,
  isRouterClientNetworkError,
} from "@agent-platform/router-client";

const client = createRouterClient({
  baseUrl: "http://127.0.0.1:8780",
  fetchPolicy: {
    timeoutMs: 60_000,
    retries: 3,
    retryDelayMs: 300,
  },
});

try {
  await client.route("hello", {
    fetchPolicy: { timeoutMs: 10_000 }, // 单次请求覆盖
  });
} catch (error) {
  if (isRouterClientTimeoutError(error)) {
    console.error("client timeout");
  } else if (isRouterClientNetworkError(error)) {
    console.error("network", error.cause);
  } else if (isRouterClientError(error)) {
    console.error("HTTP", error.status, error.detail);
  }
}
```

| 错误类型 | `status` | 何时抛出 |
|----------|----------|----------|
| `RouterClientError` | HTTP 状态码 | 4xx/5xx（不可重试或重试耗尽） |
| `RouterClientTimeoutError` | `408` | `AbortSignal.timeout` / 超时 |
| `RouterClientNetworkError` | `0` | fetch 失败（DNS、连接断开等） |

自动重试：**429 / 502 / 503 / 504** 与网络/超时错误（不含 4xx 业务错误）。`routeStream` 仅对**建立 SSE 连接**的重试生效。

## 便捷函数

```typescript
import { route } from "@agent-platform/router-client";

const result = await route("http://127.0.0.1:8780", "hello", {
  apiKey: "dev-key",
  profile: "auto",
});
```

## 类型

| 导出 | 说明 |
|------|------|
| `RouterClient` | HTTP + SSE 客户端 |
| `createRouterClient` | 工厂 |
| `route` | 一次性同步调用 |
| `routeStream` | `RouterClient#routeStream` |
| `submitJob` | `POST /v1/jobs` |
| `getJob` | `GET /v1/jobs/{id}` |
| `StreamEvent` | SSE 事件（`type` / `stage` / `data`） |
| `ChatResponse` | 与 OpenAPI `ChatResponse` 对齐 |
| `JobSubmitResponse` / `JobRecord` | 异步任务提交与查询 |
| `FetchPolicy` | 客户端 `timeoutMs` / `retries` / `retryDelayMs` |
| `RouterClientTimeoutError` / `RouterClientNetworkError` | 超时与网络错误 |
| `isRouterClientError` 等 | 错误类型守卫 |

## 联调验证

对已运行的 platform API：

```bash
# 终端 1
python scripts/run_api.py

# 终端 2（仓库根目录）
python scripts/smoke_router_client.py

# 或仅 SDK 目录
ROUTER_CLIENT_BASE_URL=http://127.0.0.1:8780 npm run test:integration
```

自动化联调（pytest 启动临时 API + Node integration）：

```bash
pytest tests/test_phase26_router_client_integration.py -q
```

## 开发

```bash
npm run build
npm test
npm run test:integration   # 需 ROUTER_CLIENT_BASE_URL
```

## 相关文档

- [sdk_integration.md](../../docs/sdk_integration.md) — 本地联调三步（API + smoke + demo）
- [demo-web](../demo-web/README.md) — Vite Chat + SSE 时间线 Demo
- [router_engine.md](../../docs/router_engine.md)
- [UPGRADE.md](../../UPGRADE.md) Phase 25 P3 / Phase 26
