"""编排后端统一协议（Fixed Graph / Supervisor 等）。"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Literal, Optional, Protocol, runtime_checkable

OrchestrationMode = Literal["fixed_graph", "supervisor"]

MODE_FIXED_GRAPH: OrchestrationMode = "fixed_graph"
MODE_SUPERVISOR: OrchestrationMode = "supervisor"

AgentTransport = Literal["local", "a2a", "mixed"]

TRANSPORT_LOCAL: AgentTransport = "local"
TRANSPORT_A2A: AgentTransport = "a2a"
TRANSPORT_MIXED: AgentTransport = "mixed"

SUPPORTED_MODES: tuple[OrchestrationMode, ...] = (MODE_FIXED_GRAPH, MODE_SUPERVISOR)


@runtime_checkable
class OrchestrationBackend(Protocol):
    """平台层对外统一的编排运行时契约。"""

    mode: str
    domain: Optional[str]
    user_id: str

    async def process_request(
        self,
        user_query: str,
        thread_id: str = "default",
        timeout_sec: Optional[float] = None,
    ) -> Dict[str, Any]: ...

    async def iter_request_stream(
        self,
        user_query: str,
        thread_id: str = "default",
    ) -> AsyncIterator[Dict[str, Any]]: ...
