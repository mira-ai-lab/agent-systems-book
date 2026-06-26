"""按 (domain, mode, transport, user_id) 缓存编排运行时（多租户 + 多范式）。"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Optional

from agent_framework.config import DEFAULT_DOMAIN, TENANT_ORCHESTRATOR_CACHE_SIZE
from agent_framework.i18n.locale import normalize_locale
from agent_framework.orchestration.protocol import (
    MODE_FIXED_GRAPH,
    MODE_SUPERVISOR,
    OrchestrationBackend,
    OrchestrationMode,
    TRANSPORT_LOCAL,
)
from agent_framework.orchestration.router_orchestrator import MODE_ROUTER
from agent_framework.router.profile import PROFILE_AUTO, PROFILE_HYBRID, normalize_profile, profile_to_mode


def _normalize_user_id(user_id: Optional[str]) -> str:
    uid = (user_id or "default").strip()
    return uid or "default"


def _normalize_domain(domain: Optional[str]) -> str:
    name = (domain or DEFAULT_DOMAIN or "").strip()
    if not name:
        raise ValueError(
            "domain 不能为空。请指定已注册领域，例如 travel、demo。"
            "可通过 GET /v1/domains 查看列表。"
        )
    return name


def _normalize_mode(mode: Optional[str]) -> OrchestrationMode:
    value = (mode or MODE_FIXED_GRAPH).strip() or MODE_FIXED_GRAPH
    if value not in (MODE_FIXED_GRAPH, MODE_SUPERVISOR):
        raise ValueError(f"不支持的 mode='{value}'，可选: fixed_graph, supervisor")
    return value  # type: ignore[return-value]


def _normalize_transport(transport: Optional[str], *, mode: str, profile: str) -> str:
    value = (transport or TRANSPORT_LOCAL).strip() or TRANSPORT_LOCAL
    if profile == PROFILE_AUTO:
        if value not in ("local", "a2a", "mixed"):
            raise ValueError(f"不支持的 agent_transport='{value}'")
        return value
    if profile == PROFILE_HYBRID and (transport or "").strip() == "":
        return "mixed"
    if mode != MODE_SUPERVISOR and profile not in (PROFILE_AUTO, PROFILE_HYBRID) and value != TRANSPORT_LOCAL:
        raise ValueError("agent_transport 仅适用于 mode='supervisor'、profile='auto' 或 profile='hybrid'")
    if value not in ("local", "a2a", "mixed"):
        raise ValueError(f"不支持的 agent_transport='{value}'")
    return value


class TenantOrchestratorPool:
    """LRU 缓存：每个 (domain, profile, mode, transport, user_id) 独立运行时。"""

    def __init__(self, max_size: int | None = None) -> None:
        self._max_size = max_size or TENANT_ORCHESTRATOR_CACHE_SIZE
        self._cache: OrderedDict[str, OrchestrationBackend] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(
        self,
        user_id: Optional[str] = None,
        *,
        domain: Optional[str] = None,
        profile: Optional[str] = None,
        mode: Optional[str] = None,
        transport: Optional[str] = None,
        locale: Optional[str] = None,
    ) -> OrchestrationBackend:
        uid = _normalize_user_id(user_id)
        dom = _normalize_domain(domain)
        loc = normalize_locale(locale)
        execution_profile = normalize_profile(profile)
        if execution_profile == PROFILE_AUTO:
            orchestration_mode = MODE_ROUTER  # type: ignore[assignment]
        else:
            orchestration_mode = _normalize_mode(mode or profile_to_mode(execution_profile))
        agent_transport = _normalize_transport(
            transport,
            mode=orchestration_mode if orchestration_mode != "router" else MODE_SUPERVISOR,
            profile=execution_profile,
        )
        cache_key = f"{dom}:{execution_profile}:{orchestration_mode}:{agent_transport}:{loc}:{uid}"
        async with self._lock:
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                return self._cache[cache_key]

            from agent_framework.bootstrap.platform import create_runtime

            runtime = create_runtime(
                dom,
                profile=execution_profile,
                mode=orchestration_mode if execution_profile != PROFILE_AUTO else None,
                transport=agent_transport,
                user_id=uid,
                enable_guess_agent=True,
                locale=loc,
            )
            self._cache[cache_key] = runtime
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
            return runtime

    def clear(self) -> None:
        self._cache.clear()

    def invalidate(self, domain: Optional[str] = None) -> int:
        if domain is None:
            count = len(self._cache)
            self._cache.clear()
            return count
        prefix = f"{domain.strip()}:"
        keys = [key for key in self._cache if key.startswith(prefix)]
        for key in keys:
            self._cache.pop(key, None)
        return len(keys)


_default_pool: TenantOrchestratorPool | None = None


def get_tenant_pool() -> TenantOrchestratorPool:
    global _default_pool
    if _default_pool is None:
        _default_pool = TenantOrchestratorPool()
    return _default_pool
