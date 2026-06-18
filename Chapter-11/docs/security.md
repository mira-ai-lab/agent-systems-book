# 安全指南

本文档说明 **agent-platform**（`Chapter-11`）在开发、测试与生产环境中的安全边界与推荐配置。平台本身不是完整 SaaS 多租户隔离产品；部署方需叠加网络、身份与密钥管理策略。

## 威胁模型（简要）

| 资产 | 风险 | 缓解 |
|------|------|------|
| LLM API 密钥 | 泄露导致费用与数据外泄 | 环境变量 / Secret Manager，禁止提交 `.env` |
| HTTP API | 未授权调用编排与 Job | `API_KEYS` + HTTPS |
| 多租户数据 | 记忆 / KB 串租户 | 强制 `user_id`，KB tenant 命名空间 |
| 出站 HTTP | SSRF / 恶意 URL | 限制 travel 外部 API 白名单（部署侧） |
| A2A 联邦 | 不可信远程 Agent | TLS、端点白名单、`health` 探测后再路由 |
| Trace 日志 | 含用户 query 与 PII | 采样、截断、访问控制 |

## API 鉴权

实现：`services/api/auth.py`

- 环境变量 **`API_KEYS`**：逗号分隔的有效 Key 列表。
- 请求头 **`X-API-Key`**：与列表匹配则放行。
- **`API_KEYS` 为空时跳过鉴权** — 仅适用于本地开发；**生产必须设置**。

```bash
# 生产示例
API_KEYS=prod-key-1,prod-key-2
```

客户端（router-client）：

```bash
export ROUTER_CLIENT_API_KEY=prod-key-1
```

集成测试参考：`tests/test_phase6_platform.py`（`API_KEYS=test-secret`）。

## 密钥与配置管理

### 禁止入库

以下文件**不得**提交 Git（已在 `.gitignore` 建议中）：

- `.env`、各服务 `.env`
- `checkpoints.db`、`jobs.db`（可能含会话数据）
- OTEL trace 文件（含用户输入）

### 推荐做法

1. 使用 Kubernetes Secret / Vault / 云厂商 Secret Manager 注入 `OPENAI_API_KEY`、`API_KEYS`。
2. Docker Compose 生产环境改用 `env_file` 指向**宿主机只读**密钥文件，而非仓库内 `.env`。
3. CI 使用 GitHub Encrypted Secrets，不在 workflow 日志中 echo 密钥。

### LLM 提供商

`create_llm()` 从环境读取提供商密钥（如 `DASHSCOPE_API_KEY`）。轮换密钥时：

1. 在 Secret Manager 更新新 Key  
2. 滚动重启 `platform-api` Pod / 容器  
3. 撤销旧 Key  

## 传输安全

- **生产**：API 必须置于 TLS 终止层之后（Ingress、API Gateway、反向代理）。
- **A2A**：`TRAVEL_A2A_HOTEL_URL` 等远程 Agent 应使用 `https://`，并校验证书（部署侧配置）。
- **SSE**（`/v1/chat/stream`）：与 REST 相同 TLS 要求；避免在公网无鉴权暴露。

## 多租户与数据隔离

| 机制 | 隔离级别 |
|------|----------|
| `user_id` → Orchestrator 池 | 进程内 LRU，**非硬隔离** |
| Memory namespace | 按 tenant 分命名空间 |
| KB（Phase 25） | `tenant_id` 过滤检索 |
| Checkpoint SQLite | 按 thread / user 分键；同库文件共享 filesystem |

**生产限制**：

- 高敏感客户应独立部署实例或独立 DB 卷，而非仅依赖 `user_id` 字符串。
- 定期清理 `CHECKPOINT_SQLITE_PATH` 与 `JOB_DB_PATH` 中的过期数据。

## 输入与输出

- 用户 query 经 LLM 与外部工具（weather/hotel API）处理；部署方需评估数据出境与留存政策。
- Trace attribute 默认截断（`OTEL_TRACE_ATTR_MAX_LEN=500`）；仍可能含 PII，限制 trace 存储访问权限。
- Router **extraction** 阶段从 query 抽取结构化 events；日志中避免打印完整 prompt（默认 OTEL 已截断）。

## 依赖与供应链

- Python：`pyproject.toml` 固定最低版本；CI 跑 `pytest` 锁定行为。
- npm：`packages/router-client` 在 monorepo 内，发版前 `sync_package_versions.py --check`。
- 领域插件：仅安装受信任的 `domains/*`；第三方插件需代码审查后再 `register_domain`。

## Docker 镜像

- 使用非 root 用户运行（若自定义 Dockerfile，建议 `USER` 指令）。
- 不在镜像层 bake `.env` 或 API Key。
- 挂载卷权限：`/app/data` 仅 API 进程可写。

## 安全相关环境变量速查

| 变量 | 安全用途 |
|------|----------|
| `API_KEYS` | HTTP API 鉴权 |
| `OPENAI_API_KEY` / `DASHSCOPE_API_KEY` | LLM 调用 |
| `OTEL_TRACES_EXPORTER=none` | 禁用 trace 外泄（高敏环境） |
| `OTEL_TRACES_SAMPLE_ALL=0` | 生产采样，降低 PII 采集面 |

## 事件响应（建议）

1. **API Key 泄露**：立即从 `API_KEYS` 移除，轮换所有 Key，审计 Job / checkpoint 异常调用。  
2. **异常 LLM 费用**：检查 Key 使用方、启用网关速率限制。  
3. **租户数据疑似串读**：暂停服务，检查 `user_id` 传递链路与 KB tenant 过滤。

## 相关文档

- [operations.md](operations.md) — 部署、探针、CI
- [tracing_design.md](tracing_design.md) — OTEL 与采样
- [sdk_integration.md](sdk_integration.md) — 客户端 Key 传递
