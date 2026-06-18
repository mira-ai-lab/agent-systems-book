"""Phase 8：Supervisor / A2A tracing。"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from agent_framework.domain.agent_registry import SubAgentRegistry
from agent_framework.orchestration.supervisor.invoke_traced import invoke_local_sub_agent
from agent_framework.tracing import setup_observability
from agent_framework.tracing.trace_provider import (
    get_current_span_context,
    inject_trace_context,
    span_name,
)
from agent_framework.transport.a2a.call_traced import a2a_call_remote


def test_inject_trace_context_adds_traceparent():
    setup_observability()
    from agent_framework.tracing import span

    with span(span_name("request"), step="test"):
        carrier: dict[str, str] = {}
        inject_trace_context(carrier)
        assert "traceparent" in carrier
        assert carrier["traceparent"].startswith("00-")


def test_invoke_local_sub_agent_creates_span():
    setup_observability()
    registry = SubAgentRegistry()
    mock_agent = MagicMock()
    mock_agent.ainvoke = AsyncMock(
        return_value={"messages": [AIMessage(content="echo reply")]}
    )
    registry.register("EchoAgent", lambda: mock_agent, description="echo")

    async def _run():
        return await invoke_local_sub_agent(
            registry,
            factory_name="EchoAgent",
            node_name="echo_agent",
            description="echo",
            query="hello",
        )

    text = asyncio.run(_run())
    assert text == "echo reply"


def test_a2a_call_remote_emits_error_event_on_failure():
    pytest.importorskip("a2a")
    setup_observability()

    async def _run():
        with patch("a2a.client.create_client", side_effect=RuntimeError("connection refused")):
            return await a2a_call_remote("http://127.0.0.1:9/", "ping")

    text, ctx, success = asyncio.run(_run())
    assert success is False
    assert "connection refused" in text


def test_supervisor_handoff_events(monkeypatch):
    from agent_framework.orchestration.supervisor.orchestrator import SupervisorOrchestrator

    events: list = []

    monkeypatch.setattr(
        "agent_framework.orchestration.supervisor.orchestrator.current_trace_add_event",
        lambda name, attrs=None: events.append((name, attrs or {})),
    )
    monkeypatch.setattr(
        "agent_framework.orchestration.supervisor.orchestrator.resolve_supervisor_subgraphs",
        lambda *a, **kw: ([], [("echo_agent", "echo")]),
    )
    monkeypatch.setattr(
        "agent_framework.orchestration.supervisor.orchestrator.build_supervisor_app",
        lambda *a, **kw: MagicMock(),
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

    mock_app = MagicMock()
    mock_app.ainvoke = AsyncMock(
        return_value={
            "messages": [
                AIMessage(content="done", name="echo_agent"),
                AIMessage(content="final answer"),
            ]
        }
    )

    registry = SubAgentRegistry()
    registry.register("EchoAgent", lambda: MagicMock(), description="echo")
    from agent_framework.domain.domain_prompts import DomainPrompts

    prompts = DomainPrompts(
        central_agent_system="x",
        aggregation="a",
        facts_prompt="f",
        decomposition_prompt="d",
        dependency_system="ds",
        dependency_user="du",
        agent_routing="ar",
        supervisor_system="sup",
    )
    orch = SupervisorOrchestrator(
        MagicMock(),
        domain="demo",
        registry=registry,
        prompts=prompts,
        enable_memory=False,
        transport="local",
    )
    orch.app = mock_app
    orch._handoff_node_names = {"echo_agent"}
    orch._a2a_node_names = set()

    asyncio.run(orch.process_request("hi", thread_id="t-handoff"))
    handoff_events = [e for e in events if e[0] == "handoff.completed"]
    assert len(handoff_events) == 1
    assert handoff_events[0][1]["target"] == "echo_agent"
    assert handoff_events[0][1]["transport"] == "local"
