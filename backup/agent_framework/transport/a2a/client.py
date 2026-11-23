"""A2A 客户端（复用 Chapter-7 协议，委托 call_traced 实现）。"""

from __future__ import annotations

from typing import Optional

from agent_framework.transport.a2a.call_traced import a2a_call_remote


class A2AClient:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint.rstrip("/") + "/"

    async def call(
        self,
        query: str,
        context_id: Optional[str] = None,
    ) -> tuple[str, Optional[str]]:
        text, new_context_id, _success = await a2a_call_remote(
            self.endpoint,
            query,
            context_id,
        )
        return text, new_context_id

    async def close(self) -> None:
        return None
