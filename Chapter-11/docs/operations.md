# 生产运维指南

本文档面向 **agent-platform 0.22.0** 工作区（`Chapter-11`）的部署与日常运维。与书稿主线 `Chapter-8/` 相比，本分支提供 HTTP API、Docker、多租户、异步 Job、Router SDK 等企业化能力。

## 部署拓扑

```text
Client / Demo Web / router-client
        │
        ▼
  platform-api (:8780)
        │
        ├── RouterEngine (L1)
        ├── RouterOrchestrator → Fixed Graph / Supervisor (L2)
        ├── SQLite checkpoints + JobStore
        └── OTEL traces (file / OTLP)
```

### 本地开发

```bash
cd Chapter-11
pip install -e ".[api,dev]"
pip install -e domains/
pytest
python scripts/run_api.py
# http://127.0.0.1:8780/health
```

### Docker Compose

```bash
cd Chapter-11
docker compose up --build
# http://localhost:8780/health
```

`docker-compose.yml` 默认：

| 项 | 值 |
|----|-----|
| 端口 | `8780` |
| Checkpoint | SQLite → `/app/data/checkpoints.db` |
| Job 存储 | `/app/data/jobs.db` |
| Trace | `OTEL_TRACES_EXPORTER=file`，目录 `/app/data/traces` |
| 环境文件 | 上级 `../.env`（勿将含密钥的 `.env` 提交到 Git） |

持久卷：`platform_api_data` 挂载 `/app/data`。

### 就绪预热

`GET /ready` 可在首次请求前预热领域编排器。可选环境变量：

```bash
READY_DOMAIN=travel   # 启动后预热 travel 租户池
```

## 健康与探针

| 端点 | 用途 |
|------|------|
| `GET /health` | **存活探针**（Liveness）— 进程可用即 200 |
| `GET /ready` | **就绪探针**（Readiness）— 可选领域预热 |
| `GET /metrics` | Prometheus 指标（需 `prometheus-client`） |

Kubernetes 建议：`livenessProbe` → `/health`，`readinessProbe` → `/ready`。

## 环境变量（运维常用）

| 变量 | 默认 | 说明 |
|------|------|------|
| `API_HOST` | `127.0.0.1` | 绑定地址；Docker 内用 `0.0.0.0` |
| `API_PORT` | `8780` | 监听端口 |
| `API_KEYS` | *(空)* | 逗号分隔 API Key；**空则跳过鉴权**（仅本地） |
| `DEFAULT_DOMAIN` | *(空)* | 无 `domain` 请求时的回落 |
| `OPENAI_API_KEY` / `DASHSCOPE_API_KEY` | — | LLM 提供商密钥（按 `create_llm` 配置） |
| `CHECKPOINT_BACKEND` | `memory` | `memory` / `sqlite` |
| `CHECKPOINT_SQLITE_PATH` | — | SQLite checkpoint 路径 |
| `JOB_DB_PATH` | — | 异步 Job SQLite 路径 |
| `OTEL_SERVICE_NAME` | `multi-agent-platform` | OTEL 服务名 |
| `OTEL_TRACES_EXPORTER` | `console` | `console` / `file` / `otlp` / `none` |
| `OTEL_TRACES_DIR` | — | file 模式输出目录 |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | OTLP 收集器 |
| `TRAVEL_A2A_HOTEL_URL` | *(空)* | travel mixed 模式下 Hotel Agent A2A 基址 |

完整 OTEL 变量见 [tracing_design.md](tracing_design.md)。

## 多租户与隔离

- HTTP `user_id`（或 `tenant_id`）映射到 `TenantOrchestratorPool`，按 `(domain, user_id)` LRU 缓存编排器实例。
- 长期记忆与 KB 检索支持租户命名空间（Phase 25 KB 多租户）。
- **生产建议**：为每个外部客户分配独立 `user_id`；敏感领域启用 `API_KEYS` + 网络隔离。

## Registry 联邦

- `GET /v1/agents?federated=true` 合并本地 Registry 与远程 A2A 端点。
- 可选 `&health=true` 对远程 Agent 做 HTTP 探测。
- 运维注意：联邦节点超时/失败会反映在响应 `health` 字段，不影响本地 Agent 可用性。

## 并发、超时与重试

| 层级 | 机制 |
|------|------|
| LLM 调用 | `async_retry` + 可重试错误判定（TaskPlanner / Agent） |
| HTTP 出站 | `async_http_request` 超时 + 重试（travel 外部 API） |
| router-client | `RouterClientTimeoutError` / 网络重试（见 [sdk_integration.md](sdk_integration.md)） |
| API Job 队列 | `POST /v1/jobs` 异步执行，避免长连接超时 |

生产建议：在反向代理（Nginx / Gateway）设置 `POST /v1/chat/stream` 的 SSE 空闲超时 ≥ 120s。

## 可观测性

- Span 命名：`latc.{OTEL_SERVICE_NAME}.{suffix}`（见 tracing_design）。
- Router 阶段事件：`router.classification`、`router.task_decomposition`、`router.semantic_routing`（travel workflow）。
- 文件 trace 适合开发；生产推荐 `OTEL_TRACES_EXPORTER=otlp` 接入 Jaeger / Tempo。

## CI 与发版

GitHub Actions：`.github/workflows/chapter8-upgrade-ci.yml`

- Python：`pytest` 全量
- npm：`packages/router-client` unit + integration
- Demo Web build
- **版本同步**：`python scripts/sync_package_versions.py --check`

发版流程：

```bash
# 1.  bump pyproject.toml version
python scripts/sync_package_versions.py --sync
pytest
# 2. tag / release（npm 包当前为 monorepo 内 workspace，未发布到 npm registry）
```

## 领域插件安装

生产镜像需显式安装领域包，否则 entry_points 为空（开发环境有 `load_dev_fallback_plugins` 回退，**生产不应依赖**）：

```bash
pip install -e domains/
# 或 pip install -e domains/travel domains/customer_service
```

## 故障排查

| 现象 | 排查 |
|------|------|
| `未知领域 'xxx'` | 未 `pip install -e domains/` 或未注册插件 |
| travel 只跑单 Agent | 确认 `profile=workflow` 且 Phase 27 语义路由开启；书稿全链路可用 `--legacy-graph` |
| 401 on `/v1/*` | 检查 `API_KEYS` 与 `X-API-Key` 请求头 |
| Job 一直 pending | 查看 API 日志、`JOB_DB_PATH` 可写性 |
| OTEL 无输出 | 确认 `OTEL_TRACES_EXPORTER` 非 `none`，目录可写 |

## 相关文档

- [security.md](security.md) — 鉴权、密钥、租户隔离
- [sdk_integration.md](sdk_integration.md) — 本地联调
- [router_engine.md](router_engine.md) — Router 阶段与 Profile
- [domains.md](domains.md) — 内置领域与 CLI 示例
