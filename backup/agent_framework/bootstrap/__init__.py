"""领域 demo 组装与平台入口。

推荐：``await route(query)`` — 企业统一入口。

``create_runtime`` / ``create_orchestrator`` 为进阶用法。
"""

from agent_framework.bootstrap.entry import route
from agent_framework.bootstrap.platform import create_orchestrator, create_runtime
from agent_framework.bootstrap.tenant_pool import TenantOrchestratorPool, get_tenant_pool

__all__ = [
    "route",
    "create_runtime",
    "create_orchestrator",
    "TenantOrchestratorPool",
    "get_tenant_pool",
]
