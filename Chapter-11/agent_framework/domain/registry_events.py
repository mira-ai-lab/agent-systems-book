"""Registry 变更通知：SSE 事件 schema + 可选 webhook。"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from agent_framework.stream.events import registry_updated_event


async def notify_registry_updated(
    *,
    domain: str,
    action: str,
    agent_name: Optional[str] = None,
    scope: str = "domain",
    source: str = "dynamic",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """构建 registry.updated 事件；若配置 REGISTRY_WEBHOOK_URL 则异步 POST。"""
    event = registry_updated_event(
        domain=domain,
        action=action,
        agent_name=agent_name,
        scope=scope,
        source=source,
        extra=extra,
    )
    webhook = (os.getenv("REGISTRY_WEBHOOK_URL") or "").strip()
    if webhook:
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(webhook, json=event)
        except Exception:
            pass
    return event
