# 内置领域定位

`agent-platform` 通过 `domains/` 插件扩展业务能力。内置三个领域，**产品叙事**与**能力展示**分工如下。

## 默认产品域：customer_service

| 属性 | 值 |
|------|-----|
| 推荐场景 | 企业路由引擎默认 demo、HTTP `POST /v1/chat` 无 domain 时的回落推断 |
| CLI | `python scripts/run_demo.py`（默认 `customer_service` + `profile=auto`） |
| Agent | FAQAgent、TicketAgent |
| 编排 | `profile=auto` → 多 Agent 走 workflow，单 Agent 走 adaptive |

**为何作为默认：** 话术短、依赖少、无需 MCP/A2A，新用户可在 5 分钟内跑通「只传 query」路径。

## 能力展示域：travel

| 属性 | 值 |
|------|-----|
| 定位 | **书稿 / 能力展示**，非默认产品叙事 |
| 典型场景 | Fixed Graph 全链路、MCP 工具、A2A mixed 混部、多 Agent 行程规划 |
| CLI | `python scripts/run_demo.py --domain travel --legacy-graph --stream`（书稿 Fixed Graph 直连） |
| Router 路径 | `python scripts/run_demo.py --domain travel --profile workflow`（语义路由 + **travel 默认 `pre_survey_mode=full_ch2`**） |
| Agent | Weather / Hotel / Restaurant / Flight / Itinerary（5 个子 Agent） |

**说明：** `travel` 已是正式产品域（`recommended=true`），但在文档与 demo 中标注为「能力展示」，避免读者误以为必须部署旅行业务才能使用平台。

## 最小模板：demo

| 属性 | 值 |
|------|-----|
| 定位 | 插件开发脚手架 |
| Agent | EchoAgent |
| 用途 | 验证 entry_points、`DomainPlugin` 四件套 |

## 如何选择 domain

```python
from agent_framework.bootstrap import route

# 产品默认（推荐）
await route("退货政策是什么？", domain="customer_service")

# 能力展示（Fixed Graph + 多 Agent）
await route("规划北京三日游", domain="travel", profile="workflow")

# 仅 query，跨域 LLM 推断
await route("我要咨询退货政策")
```

HTTP 同等：`POST /v1/chat` 的 `domain` 可省略；`GET /v1/domains` 返回各域 `recommended_profile=auto`。

## 相关文档

- [router_engine.md](router_engine.md) — L1 路由与 Profile
- [plugin_development.md](plugin_development.md) — 新增领域插件
- [operations.md](operations.md) — 部署与运维
- [security.md](security.md) — 鉴权与密钥
- [UPGRADE.md](../UPGRADE.md) — Phase 24+ 演进记录
