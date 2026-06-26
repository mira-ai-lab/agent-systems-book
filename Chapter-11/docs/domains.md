# 内置领域定位

`agent-platform` 通过 `domains/` 插件扩展业务能力。当前内置两个领域：**travel** 为书稿与仓库内的完整示例，**demo** 为最小插件模板。

## 主示例域：travel

| 属性 | 值 |
|------|-----|
| 定位 | **书稿 / 完整参考实现**（Agent、benchmark、MCP、Prompt 进化） |
| 推荐场景 | Fixed Graph 全链路、语义路由、多 Agent 行程规划、TextGrad 评测与优化 |
| CLI 默认 | `python scripts/run_demo.py`（默认 `travel` + `profile=auto` + 多城市样例 query） |
| Router 路径 | `python scripts/run_demo.py --domain travel --profile workflow --stream` |
| legacy Fixed Graph | `python scripts/run_demo.py --legacy-graph --stream`（不经 RouterEngine） |
| Agent | Weather / Hotel / Restaurant / Flight / Itinerary（5 个子 Agent） |
| 编排 | `profile=auto` → 多 Agent 走 workflow，单 Agent 走 adaptive |

扩展新业务域时，对照 `domains/travel/` 的能力边界与 [plugin_development.md](plugin_development.md)，或从 `demo` 骨架起步。

## 最小模板：demo

| 属性 | 值 |
|------|-----|
| 定位 | 插件开发脚手架 |
| Agent | EchoAgent |
| 用途 | 验证 entry_points、`DomainPlugin` 四件套 |

## 如何选择 domain

```python
from agent_framework.bootstrap import route

# 书稿示例（推荐）
await route(
    "帮我查北京明天天气，并推荐一家三亚海棠湾附近的酒店",
    domain="travel",
    profile="workflow",
)

# profile=auto：多 Agent → workflow，单 Agent → adaptive
await route("查 7 月 5 日广州飞北京的航班", domain="travel")

# 仅 query：由 Router 跨域 LLM 推断（需配置 DASHSCOPE_API_KEY）
await route("规划杭州三日游")
```

HTTP 同等：`POST /v1/chat` 的 `domain` 可省略（走 `DEFAULT_DOMAIN` 环境变量或 LLM 推断）；`GET /v1/domains` 返回各域 `recommended_profile` 等元数据。

## 相关文档

- [router_engine.md](router_engine.md) — L1 路由与 Profile
- [plugin_development.md](plugin_development.md) — 新增领域插件
- [operations.md](operations.md) — 部署与运维
- [security.md](security.md) — 鉴权与密钥
- [UPGRADE.md](../UPGRADE.md) — Phase 24+ 演进记录
