"""启动 registry 子 Agent 的 A2A HTTP 服务。"""

from __future__ import annotations

from typing import Any, Optional

from agent_framework.config import create_llm, load_project_dotenv
from agent_framework.domain.agent_registry import SubAgentRegistry
from agent_framework.domain.plugin_registry import get_domain_plugin
from agent_framework.infra.agent_runtime import configure_agent_llm
from agent_framework.orchestration.supervisor.agent_names import registry_agent_to_node_name
from agent_framework.tracing import setup_observability
from agent_framework.transport.a2a.server.app_factory import build_a2a_server_app
from agent_framework.transport.a2a.server.executor import RegistrySubAgentExecutor


def resolve_registry_agent(
    registry: SubAgentRegistry,
    *,
    registry_agent: Optional[str] = None,
    node_name: Optional[str] = None,
) -> str:
    if registry_agent:
        name = registry_agent.strip()
        if name not in registry.agents:
            known = ", ".join(registry.get_agent_names())
            raise ValueError(f"未知 registry_agent='{name}'，可选: {known}")
        return name
    if node_name:
        target = node_name.strip()
        for factory_name in registry.get_agent_names():
            if registry_agent_to_node_name(factory_name) == target:
                return factory_name
        raise ValueError(f"未知 node_name='{target}'")
    raise ValueError("须指定 registry_agent 或 node_name 之一")


def build_sub_agent_executor(
    domain: str,
    *,
    registry_agent: Optional[str] = None,
    node_name: Optional[str] = None,
    llm: Optional[Any] = None,
) -> RegistrySubAgentExecutor:
    load_project_dotenv()
    setup_observability()
    plugin = get_domain_plugin(domain)
    registry = plugin.create_registry()
    resolved_llm = llm or create_llm()
    configure_agent_llm(resolved_llm)
    factory_name = resolve_registry_agent(
        registry,
        registry_agent=registry_agent,
        node_name=node_name,
    )
    info = registry.agents.get(factory_name, {})
    description = str(info.get("description") or factory_name)
    agent = registry.get_agent(factory_name)

    async def _invoke(query: str, context_id: str) -> Any:
        return await agent.ainvoke(
            {"messages": [("user", query)]},
            {"configurable": {"thread_id": context_id}},
        )

    return RegistrySubAgentExecutor(
        factory_name=factory_name,
        invoke_fn=_invoke,
        display_name=factory_name,
        description=description,
    )


def serve_sub_agent(
    domain: str,
    *,
    registry_agent: Optional[str] = None,
    node_name: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 9012,
    llm: Optional[Any] = None,
) -> None:
    """阻塞启动 uvicorn，将领域子 Agent 暴露为 A2A Server。"""
    executor = build_sub_agent_executor(
        domain,
        registry_agent=registry_agent,
        node_name=node_name,
        llm=llm,
    )
    app = build_a2a_server_app(
        executor=executor,
        host=host,
        port=port,
        agent_name=executor.display_name,
        description=executor.description,
    )
    import uvicorn

    uvicorn.run(app, host=host, port=port)


def create_sub_agent_a2a_app(
    domain: str,
    *,
    registry_agent: Optional[str] = None,
    node_name: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 9012,
    llm: Optional[Any] = None,
) -> Any:
    """返回 Starlette app（供测试或自定义 uvicorn 启动）。"""
    executor = build_sub_agent_executor(
        domain,
        registry_agent=registry_agent,
        node_name=node_name,
        llm=llm,
    )
    return build_a2a_server_app(
        executor=executor,
        host=host,
        port=port,
        agent_name=executor.display_name,
        description=executor.description,
    )
