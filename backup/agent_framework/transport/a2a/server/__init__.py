"""A2A Server：将领域子 Agent 暴露为远程服务。"""

from agent_framework.transport.a2a.server.executor import RegistrySubAgentExecutor
from agent_framework.transport.a2a.server.serve import (
    build_sub_agent_executor,
    create_sub_agent_a2a_app,
    resolve_registry_agent,
    serve_sub_agent,
)

__all__ = [
    "RegistrySubAgentExecutor",
    "build_sub_agent_executor",
    "create_sub_agent_a2a_app",
    "resolve_registry_agent",
    "serve_sub_agent",
]
