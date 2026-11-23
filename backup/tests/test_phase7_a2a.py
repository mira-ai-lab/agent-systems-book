"""Phase 7C：Supervisor A2A transport（远程 handoff 目标）。"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent_framework.bootstrap.platform import create_runtime
from agent_framework.domain.a2a_spec import A2AEndpoint
from agent_framework.domain.agent_registry import SubAgentRegistry
from agent_framework.domain.plugin_registry import get_domain_plugin
from agent_framework.orchestration.protocol import (
    MODE_FIXED_GRAPH,
    MODE_SUPERVISOR,
    TRANSPORT_A2A,
    TRANSPORT_LOCAL,
    TRANSPORT_MIXED,
)
from agent_framework.orchestration.supervisor.graph import resolve_supervisor_subgraphs
from agent_framework.orchestration.supervisor.orchestrator import SupervisorOrchestrator
from agent_framework.transport.a2a.agent_graphs import build_a2a_agent_graph


def _demo_registry_with_two_agents() -> SubAgentRegistry:
    registry = SubAgentRegistry()
    registry.register("EchoAgent", lambda: MagicMock(), description="本地 Echo", requires_tool=False)
    registry.register("FAQAgent", lambda: MagicMock(), description="本地 FAQ", requires_tool=False)
    return registry


def test_resolve_supervisor_subgraphs_local():
    registry = _demo_registry_with_two_agents()
    graphs, handoff = resolve_supervisor_subgraphs(registry, transport=TRANSPORT_LOCAL)
    assert len(graphs) == 2
    nodes = {n for n, _ in handoff}
    assert nodes == {"echo_agent", "faq_agent"}


def test_resolve_supervisor_subgraphs_a2a_requires_endpoint():
    registry = _demo_registry_with_two_agents()
    with pytest.raises(ValueError, match="至少一个有效 A2AEndpoint"):
        resolve_supervisor_subgraphs(registry, transport=TRANSPORT_A2A, a2a_endpoints=())


def test_resolve_supervisor_subgraphs_mixed_replaces_local():
    registry = _demo_registry_with_two_agents()
    ep = A2AEndpoint(
        node_name="hotel_agent",
        url="http://127.0.0.1:9012/",
        description="酒店推荐",
        registry_agent="FAQAgent",
    )
    graphs, handoff = resolve_supervisor_subgraphs(
        registry,
        transport=TRANSPORT_MIXED,
        a2a_endpoints=[ep],
    )
    nodes = {n for n, _ in handoff}
    assert "echo_agent" in nodes
    assert "hotel_agent" in nodes
    assert "faq_agent" not in nodes
    assert len(graphs) == 2


def test_travel_a2a_endpoints_from_env(monkeypatch):
    monkeypatch.setenv("TRAVEL_A2A_HOTEL_URL", "http://127.0.0.1:9012/")
    plugin = get_domain_plugin("travel")
    endpoints = plugin.resolved_a2a_endpoints()
    assert len(endpoints) == 1
    assert endpoints[0].node_name == "hotel_agent"
    assert endpoints[0].registry_agent == "HotelAgent"


def test_create_runtime_transport_invalid_on_fixed_graph():
    with pytest.raises(ValueError, match="agent_transport 仅适用于"):
        create_runtime("demo", mode=MODE_FIXED_GRAPH, transport="a2a", enable_memory=False, llm=MagicMock())


def test_create_runtime_supervisor_passes_transport(monkeypatch):
    captured: dict = {}

    def fake_build(*args, **kwargs):
        captured.update(kwargs)
        mock_app = MagicMock()
        mock_app.ainvoke = AsyncMock(
            return_value={"messages": [AIMessage(content="ok", name="echo_agent")]}
        )
        return mock_app

    mock_llm = MagicMock()
    monkeypatch.setattr(
        "agent_framework.orchestration.supervisor.orchestrator.build_supervisor_app",
        fake_build,
    )
    monkeypatch.setattr(
        "agent_framework.orchestration.supervisor.orchestrator.create_long_term_memory",
        lambda *a, **kw: (None, None),
    )
    monkeypatch.setattr(
        "agent_framework.orchestration.supervisor.orchestrator.setup_observability",
        lambda: None,
    )
    monkeypatch.setattr(
        "agent_framework.orchestration.supervisor.orchestrator.load_project_dotenv",
        lambda: None,
    )

    ep = A2AEndpoint(node_name="remote_agent", url="http://example.test/", description="远程")
    runtime = create_runtime(
        "demo",
        mode=MODE_SUPERVISOR,
        transport="mixed",
        enable_memory=False,
        llm=mock_llm,
    )
    assert isinstance(runtime, SupervisorOrchestrator)
    assert runtime.transport == TRANSPORT_MIXED
    assert captured.get("transport") == TRANSPORT_MIXED


def test_build_a2a_agent_graph_invokes_client():
    ep = A2AEndpoint(node_name="hotel_agent", url="http://127.0.0.1:9012/", description="酒店")
    graph = build_a2a_agent_graph(ep)

    mock_call = AsyncMock(return_value=("推荐希尔顿", "ctx-1", True))

    async def _run():
        with patch("agent_framework.transport.a2a.agent_graphs.a2a_call_remote", mock_call):
            return await graph.ainvoke(
                {"messages": [HumanMessage(content="上海酒店")]},
                config={"configurable": {"thread_id": "t-a2a"}},
            )

    result = asyncio.run(_run())
    mock_call.assert_awaited_once()
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_msgs) == 1
    assert ai_msgs[0].name == "hotel_agent"
    assert "希尔顿" in ai_msgs[0].content


def test_tenant_pool_cache_key_includes_transport():
    from agent_framework.bootstrap.tenant_pool import TenantOrchestratorPool

    pool = TenantOrchestratorPool(max_size=8)
    created = []

    def fake_create(domain, mode=MODE_FIXED_GRAPH, transport=TRANSPORT_LOCAL, **kwargs):
        runtime = MagicMock()
        created.append((domain, mode, transport, kwargs.get("user_id")))
        return runtime

    with patch("agent_framework.bootstrap.platform.create_runtime", side_effect=fake_create):
        o1 = asyncio.run(pool.get("u1", domain="demo", mode="supervisor", transport="local"))
        o2 = asyncio.run(pool.get("u1", domain="demo", mode="supervisor", transport="mixed"))

    assert o1 is not o2
    assert ("demo", "supervisor", "local", "u1") in created
    assert ("demo", "supervisor", "mixed", "u1") in created
