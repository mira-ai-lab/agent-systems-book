"""A2A HTTP Server 应用工厂（Starlette + a2a-sdk）。"""

from __future__ import annotations

import contextvars
from typing import Any, Optional

from agent_framework.tracing.trace_provider import extract_trace_context
from agent_framework.transport.a2a.server.executor import RegistrySubAgentExecutor

_trace_detach_token: contextvars.ContextVar[Optional[object]] = contextvars.ContextVar(
    "_trace_detach_token",
    default=None,
)


class TraceContextMiddleware:
    """从入站 HTTP headers 提取 W3C traceparent，挂到当前 OTEL context。"""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        token = None
        if scope.get("type") == "http":
            headers = {
                k.decode("latin-1").lower(): v.decode("latin-1")
                for k, v in scope.get("headers", [])
            }
            token = extract_trace_context(headers)
            _trace_detach_token.set(token)
        try:
            await self.app(scope, receive, send)
        finally:
            detach = _trace_detach_token.get()
            if detach is not None:
                from opentelemetry import context as otel_context

                otel_context.detach(detach)
                _trace_detach_token.set(None)


def _require_a2a_server() -> None:
    try:
        import a2a.server.routes  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "A2A Server 需要安装：pip install 'agent-platform[a2a]'"
        ) from exc


def build_a2a_server_app(
    *,
    executor: RegistrySubAgentExecutor,
    host: str,
    port: int,
    agent_name: Optional[str] = None,
    description: Optional[str] = None,
    skill_id: Optional[str] = None,
) -> Any:
    """构建 Starlette A2A Server 应用。"""
    _require_a2a_server()
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
    from a2a.server.tasks import InMemoryTaskStore
    from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
    from starlette.applications import Starlette

    display_name = agent_name or executor.display_name
    rpc_url = f"http://{host}:{port}/"
    agent_card = AgentCard(
        name=display_name,
        description=description or executor.description,
        version="1.0.0",
        supported_interfaces=[
            AgentInterface(
                url=rpc_url,
                protocol_binding="JSONRPC",
                protocol_version="1.0",
            )
        ],
        capabilities=AgentCapabilities(streaming=True, push_notifications=False),
        default_input_modes=RegistrySubAgentExecutor.SUPPORTED_CONTENT_TYPES,
        default_output_modes=RegistrySubAgentExecutor.SUPPORTED_CONTENT_TYPES,
        skills=[
            AgentSkill(
                id=skill_id or display_name.lower().replace(" ", "_"),
                name=display_name,
                description=description or executor.description,
                tags=["agent-platform", "sub-agent"],
            )
        ],
    )
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )
    routes = []
    routes.extend(create_agent_card_routes(agent_card))
    routes.extend(
        create_jsonrpc_routes(
            request_handler,
            rpc_url="/",
            enable_v0_3_compat=True,
        )
    )
    return TraceContextMiddleware(Starlette(routes=routes))
