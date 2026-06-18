"""Phase 24 P2：profile_reason + adaptive 流式 handoff + 就绪度脚本。"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from agent_framework.domain.agent_registry import SubAgentRegistry
from agent_framework.domain.domain_prompts import DomainPrompts
from agent_framework.orchestration.supervisor.orchestrator import SupervisorOrchestrator
from agent_framework.router.plan import AgentCandidate
from agent_framework.router.profile import (
    STRONG_CANDIDATE_THRESHOLD,
    resolve_auto_profile_with_reason,
    resolve_profile_with_reason,
)
from agent_framework.stream.events import public_event


def test_resolve_auto_profile_with_reason_multi_agent():
    candidates = [
        AgentCandidate("FAQAgent", 0.9),
        AgentCandidate("TicketAgent", 0.85),
    ]
    profile, reason = resolve_auto_profile_with_reason(candidates)
    assert profile == "workflow"
    assert f">={STRONG_CANDIDATE_THRESHOLD}" in reason
    assert "FAQAgent" in reason


def test_resolve_auto_profile_with_reason_single_agent():
    candidates = [AgentCandidate("FAQAgent", 0.9)]
    profile, reason = resolve_auto_profile_with_reason(candidates)
    assert profile == "adaptive"
    assert "single_strong_agent=FAQAgent" in reason


def test_resolve_profile_with_reason_forced():
    profile, reason = resolve_profile_with_reason([], force_profile="workflow")
    assert profile == "workflow"
    assert reason == "forced_profile=workflow"


def test_router_plan_includes_profile_reason():
    registry = SubAgentRegistry()
    registry.register("FAQAgent", lambda: MagicMock(), description="FAQ")
    registry.register("TicketAgent", lambda: MagicMock(), description="Ticket")
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(content='["咨询退货"]'),
            AIMessage(
                content='[{"name": "FAQAgent", "score": 0.9}, {"name": "TicketAgent", "score": 0.8}]'
            ),
            AIMessage(content="整体目标\n子任务：\n- 咨询退货"),
        ]
    )
    from agent_framework.router.engine import RouterEngine

    plan = asyncio.run(RouterEngine(mock_llm, registry).route("退货政策"))
    assert plan.metadata.get("profile_reason")
    assert plan.profile == "workflow"


def _supervisor_prompts() -> DomainPrompts:
    return DomainPrompts(
        central_agent_system="x",
        aggregation="a",
        facts_prompt="f",
        decomposition_prompt="d",
        dependency_system="ds",
        dependency_user="du",
        agent_routing="ar",
        supervisor_system="sup",
    )


def test_supervisor_iter_request_stream_emits_handoff_events(monkeypatch):
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
    monkeypatch.setattr(
        "agent_framework.orchestration.supervisor.orchestrator.current_trace_add_event",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "agent_framework.orchestration.supervisor.orchestrator.record_handoff",
        lambda *a, **kw: None,
    )

    async def fake_astream(*args, **kwargs):
        yield {"echo_agent": {"messages": [AIMessage(content="echo reply", name="echo_agent")]}}
        yield {"supervisor": {"messages": [AIMessage(content="final answer")]}}

    mock_state = MagicMock()
    mock_state.values = {
        "messages": [
            AIMessage(content="echo reply", name="echo_agent"),
            AIMessage(content="final answer"),
        ]
    }

    mock_app = MagicMock()
    mock_app.astream = fake_astream
    mock_app.aget_state = AsyncMock(return_value=mock_state)

    registry = SubAgentRegistry()
    registry.register("EchoAgent", lambda: MagicMock(), description="echo")
    orch = SupervisorOrchestrator(
        MagicMock(),
        domain="demo",
        registry=registry,
        prompts=_supervisor_prompts(),
        enable_memory=False,
        transport="local",
    )
    orch.app = mock_app
    orch._handoff_node_names = {"echo_agent"}
    orch._a2a_node_names = set()

    async def _collect():
        events = []
        async for event in orch.iter_request_stream("hi", thread_id="t-stream"):
            events.append(public_event(event))
        return events

    events = asyncio.run(_collect())
    types = [e["type"] for e in events]
    assert "handoff.completed" in types
    assert types[-1] == "final"
    handoff = next(e for e in events if e["type"] == "handoff.completed")
    assert handoff["data"]["target"] == "echo_agent"


def test_product_readiness_check_script():
    import importlib.util
    import sys

    path = Path(__file__).resolve().parent.parent / "scripts" / "product_readiness_check.py"
    spec = importlib.util.spec_from_file_location("product_readiness_check", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    dimensions = mod.evaluate_dimensions()
    assert len(dimensions) == 6
    assert mod.overall_score(dimensions) >= 90
    for dim in dimensions:
        assert dim.score_pct >= dim.target_pct - 10
