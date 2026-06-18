"""从 setuptools entry_points 发现并加载领域插件。"""

from __future__ import annotations

import sys
from typing import List

from agent_framework.domain.plugin import DomainPlugin
from agent_framework.domain.plugin_registry import register_domain

ENTRYPOINT_GROUP = "agent_platform.domains"


def _iter_entry_points(group: str):
    if sys.version_info >= (3, 10):
        from importlib.metadata import entry_points

        eps = entry_points(group=group)
        return list(eps)
    from importlib.metadata import entry_points as eps_legacy

    selected = eps_legacy()
    return list(selected.get(group, []))


def load_plugins_from_entrypoints() -> List[DomainPlugin]:
    """加载 ``agent_platform.domains`` 组内全部 entry point，返回已注册插件列表。"""
    loaded: List[DomainPlugin] = []
    for ep in _iter_entry_points(ENTRYPOINT_GROUP):
        obj = ep.load()
        if not isinstance(obj, DomainPlugin):
            raise TypeError(
                f"entry point {ep.name} ({ep.value}) 必须返回 DomainPlugin 实例，"
                f"实际: {type(obj)!r}"
            )
        register_domain(obj)
        loaded.append(obj)
    return loaded


def load_dev_fallback_plugins() -> List[DomainPlugin]:
    """entry_points 未安装时，从仓库 ``domains/`` 直接注册（本地开发）。"""
    loaded: List[DomainPlugin] = []
    try:
        from domains.travel.plugin import TRAVEL_PLUGIN
        from domains.customer_service.plugin import CUSTOMER_SERVICE_PLUGIN
        from domains.demo.plugin import DEMO_PLUGIN
    except ImportError:
        return loaded
    for plugin in (TRAVEL_PLUGIN, CUSTOMER_SERVICE_PLUGIN, DEMO_PLUGIN):
        register_domain(plugin)
        loaded.append(plugin)
    return loaded
