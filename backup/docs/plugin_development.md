# 领域插件开发指南

`agent-platform` 通过 **DomainPlugin** + **entry_points** 扩展业务领域。框架只提供固定图编排；领域代码独立打包。

## 1. 最小四文件结构

以 `domains/demo/` 为模板（可直接复制改名）：

```
domains/my_finance/
├── plugin.py          # DomainPlugin 实例 + entry point 目标
├── registry.py        # SubAgentRegistry 工厂
├── prompt_bundle.py   # DomainPrompts 工厂
└── prompts.py         # prompt 原文（可选拆分）
```

| 文件 | 职责 |
|------|------|
| `registry.py` | 注册子 Agent 元数据与 `create_*_agent` 工厂 |
| `prompt_bundle.py` | 返回 `DomainPrompts`（分解 / 路由 / 聚合等） |
| `plugin.py` | 组装 `DomainPlugin(name=..., create_registry=..., ...)` |
| `tests/test_my_finance_plugin.py` | 插件加载与 registry 元数据测试 |

参考实现：`domains/demo/plugin.py`（单文件极简版）。

## 2. 注册插件（entry_points）

在领域包的 `pyproject.toml` 中声明：

```toml
[project.entry-points."agent_platform.domains"]
finance = "domains.finance.plugin:FINANCE_PLUGIN"
```

安装后框架通过 `importlib.metadata` 自动发现，**无需修改** `agent_framework`。

书稿内置插件见 `domains/pyproject.toml`（`agent-platform-domains-builtin`）。

## 3. 使用插件

```python
from agent_framework.bootstrap.platform import create_runtime

# 推荐：profile=auto
runtime = create_runtime("finance", profile="auto", user_id="tenant-1")
result = await runtime.process_request("查询理财产品收益率")

# 固定 workflow：Router → FixedGraph
runtime = create_runtime("finance", profile="workflow")
```

HTTP API：

```bash
curl -H "X-API-Key: $KEY" -X POST http://localhost:8780/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"domain":"finance","query":"..."}'
```

## 4. 安装方式

| 场景 | 命令 |
|------|------|
| 仅 SDK 框架 | `pip install -e .` |
| 框架 + HTTP 服务 | `pip install -e ".[api]"` |
| 框架 + 内置示例领域 | `pip install -e . && pip install -e domains/` |
| 书稿全量开发 | `pip install -e ".[api,dev]" && pip install -e domains/` |

## 5. 检查清单

- [ ] `DomainPlugin.name` 全局唯一
- [ ] 子 Agent 在 `SubAgentRegistry.register()` 中提供 `description` / `skills`
- [ ] `decomposition_prompt` 含 `{user_query}`、`{agent_list}` 等占位符（与 TaskPlanner 一致）
- [ ] 单元测试：`get_domain_plugin("your_domain")` 与 registry stub
- [ ] 不在 `agent_framework/` 内 import 你的领域包

## 6. 与 travel 示例的关系

`travel` 是书稿附带的**完整参考实现**（含外部 API、MCP），不是框架默认。新业务请复制 `demo` 或 `travel` 骨架，而非修改框架。
