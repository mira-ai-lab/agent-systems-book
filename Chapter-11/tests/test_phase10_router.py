"""Phase 10：RouterEngine + profile=auto + classification。"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from agent_framework.bootstrap.platform import create_runtime
from agent_framework.domain.agent_registry import SubAgentRegistry
from agent_framework.router.plan import AgentCandidate
from agent_framework.router.profile import resolve_auto_profile
from agent_framework.router.stages.classification import parse_classification_response
from agent_framework.router.engine import RouterEngine


def _cs_registry() -> SubAgentRegistry:
    registry = SubAgentRegistry()
    registry.register("FAQAgent", lambda: MagicMock(), description="FAQ 政策咨询")
    registry.register("TicketAgent", lambda: MagicMock(), description="工单投诉")
    return registry


def test_parse_classification_filters_unknown_agents():
    registry = _cs_registry()
    candidates = parse_classification_response(
        [
            {"name": "FAQAgent", "score": 0.9},
            {"name": "UnknownAgent", "score": 0.8},
        ],
        registry,
    )
    assert len(candidates) == 1
    assert candidates[0].name == "FAQAgent"


def test_resolve_auto_profile_multi_agent_workflow():
    candidates = [
        AgentCandidate("FAQAgent", 0.9),
        AgentCandidate("TicketAgent", 0.85),
    ]
    assert resolve_auto_profile(candidates) == "workflow"


def test_resolve_auto_profile_single_agent_adaptive():
    candidates = [AgentCandidate("FAQAgent", 0.9)]
    assert resolve_auto_profile(candidates) == "adaptive"


def test_router_engine_route_mock_llm():
    registry = _cs_registry()
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(content='["投诉物流", "咨询退货政策"]'),
            AIMessage(
                content='[{"name": "FAQAgent", "score": 0.9}, {"name": "TicketAgent", "score": 0.8}]'
            ),
            AIMessage(content="整体目标：处理投诉与退货\n子任务：\n- 咨询退货政策\n- 提交物流投诉"),
        ]
    )
    engine = RouterEngine(mock_llm, registry)
    plan = asyncio.run(engine.route("我要投诉并咨询退货政策"))
    assert plan.profile == "workflow"
    assert len(plan.candidates) == 2
    assert len(plan.steps) == 2
    assert "task_decomposition" in plan.metadata["stages"]


def test_create_runtime_auto_returns_router_orchestrator(monkeypatch):
    from agent_framework.orchestration.router_orchestrator import RouterOrchestrator

    monkeypatch.setattr(
        "agent_framework.orchestration.router_orchestrator.load_project_dotenv",
        lambda: None,
    )
    monkeypatch.setattr(
        "agent_framework.orchestration.router_orchestrator.setup_observability",
        lambda: None,
    )
    runtime = create_runtime("demo", profile="auto", enable_memory=False, llm=MagicMock())
    assert isinstance(runtime, RouterOrchestrator)


def test_router_orchestrator_delegates(monkeypatch):
    from agent_framework.orchestration.router_orchestrator import RouterOrchestrator
    from agent_framework.router.plan import RoutingPlan

    plan = RoutingPlan(
        rewritten_query="hello",
        candidates=[AgentCandidate("EchoAgent", 0.95)],
        profile="adaptive",
    )
    mock_backend = MagicMock()
    mock_backend.process_request = AsyncMock(
        return_value={"final_response": "echo ok", "trace_id": "t1", "span_id": "s1"}
    )

    monkeypatch.setattr(
        "agent_framework.orchestration.router_orchestrator.load_project_dotenv",
        lambda: None,
    )
    monkeypatch.setattr(
        "agent_framework.orchestration.router_orchestrator.setup_observability",
        lambda: None,
    )

    orch = create_runtime("demo", profile="auto", enable_memory=False, llm=MagicMock())
    assert isinstance(orch, RouterOrchestrator)
    orch._router.route = AsyncMock(return_value=plan)  # type: ignore[method-assign]
    orch._get_backend = AsyncMock(return_value=mock_backend)  # type: ignore[method-assign]

    result = asyncio.run(orch.process_request("hello", thread_id="t10"))
    assert result["final_response"] == "echo ok"
    assert result["resolved_profile"] == "adaptive"
    assert result["routing_plan"]["profile"] == "adaptive"
