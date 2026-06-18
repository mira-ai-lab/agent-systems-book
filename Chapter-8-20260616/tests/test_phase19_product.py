"""Phase 19：README 产品化 + workflow 与 Router 完全合一。"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from agent_framework.bootstrap.platform import create_orchestrator, create_runtime
from agent_framework.domain.agent_registry import SubAgentRegistry
from agent_framework.domain.domain_config import DomainConfig
from agent_framework.domain.domain_prompts import DomainPrompts
from agent_framework.domain.pipeline import PipelineConfig
from agent_framework.orchestration.fixed_graph.nodes import GraphContext, make_nodes
from agent_framework.orchestration.fixed_graph.state import CentralAgentState
from agent_framework.orchestration.router_orchestrator import RouterOrchestrator
from agent_framework.router.execution_plan_bridge import ensure_execution_plan_from_routing_plan
from agent_framework.router.plan import AgentCandidate, RoutingPlan
from agent_framework.router.profile import PROFILE_WORKFLOW


def test_create_runtime_workflow_returns_router_orchestrator(monkeypatch):
    mock_llm = MagicMock()
    monkeypatch.setattr(
        "agent_framework.orchestration.router_orchestrator.setup_observability",
        lambda: None,
    )
    monkeypatch.setattr(
        "agent_framework.orchestration.router_orchestrator.load_project_dotenv",
        lambda: None,
    )
    runtime = create_runtime("customer_service", profile="workflow", llm=mock_llm)
    assert isinstance(runtime, RouterOrchestrator)
    assert runtime.entry_profile == PROFILE_WORKFLOW


def test_create_orchestrator_is_router_workflow_alias(monkeypatch):
    mock_llm = MagicMock()
    monkeypatch.setattr(
        "agent_framework.orchestration.router_orchestrator.setup_observability",
        lambda: None,
    )
    monkeypatch.setattr(
        "agent_framework.orchestration.router_orchestrator.load_project_dotenv",
        lambda: None,
    )
    orch = create_orchestrator("customer_service", llm=mock_llm)
    assert isinstance(orch, RouterOrchestrator)
    assert orch.entry_profile == PROFILE_WORKFLOW


def test_ensure_execution_plan_synthesizes_single_step():
    plan = RoutingPlan(
        rewritten_query="查退货政策",
        profile="workflow",
        candidates=[AgentCandidate("FAQAgent", 0.92)],
    )
    execution = ensure_execution_plan_from_routing_plan(plan, user_query="查退货政策")
    assert execution["source"] == "router_engine"
    assert len(execution["subtasks"]) == 1
    assert execution["subtasks"][0]["agent"] == "FAQAgent"


def test_build_plan_rejects_task_planner_when_router_unified():
    registry = SubAgentRegistry()
    registry.register("A", lambda: MagicMock(), description="A")
    ctx = GraphContext(
        MagicMock(),
        None,
        registry=registry,
        prompts=DomainPrompts(
            central_agent_system="sys",
            aggregation="agg",
            facts_prompt="facts",
            decomposition_prompt="decomp",
            dependency_system="dep sys",
            dependency_user="dep user",
            agent_routing="route",
        ),
        domain_config=DomainConfig(),
        pipeline=PipelineConfig(allow_task_planner_decomposition=False),
    )
    planner = MagicMock()
    planner.build_execution_plan = AsyncMock()
    ctx.planner = planner
    nodes = make_nodes(ctx)
    state: CentralAgentState = {
        "user_query": "Do task",
        "pre_survey": {},
        "retrieved_memories": [],
        "logs": [],
    }
    with pytest.raises(ValueError, match="prefilled_execution_plan"):
        asyncio.run(nodes["build_plan"](state))
    planner.build_execution_plan.assert_not_called()


def test_router_workflow_always_prefills_execution_plan():
    plugin = MagicMock()
    plugin.create_prompts.return_value.with_platform_defaults.return_value = MagicMock()
    plugin.create_domain_config.return_value = MagicMock()
    plugin.build_pipeline.return_value = PipelineConfig(enable_pre_survey=False, enable_memory=False)
    plugin.supports_mode.return_value = True

    registry = SubAgentRegistry()
    registry.register("FAQAgent", lambda: MagicMock(), description="FAQ")

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(content='["咨询退货"]'),
            AIMessage(content='[{"name": "FAQAgent", "score": 0.95}]'),
            AIMessage(content="整体目标：咨询退货\n子任务：\n- 查政策"),
        ]
    )

    workflow = MagicMock()
    workflow.process_request = AsyncMock(return_value={"final_response": "ok"})

    orch = RouterOrchestrator(
        mock_llm,
        plugin,
        domain="customer_service",
        enable_memory=False,
        entry_profile=PROFILE_WORKFLOW,
    )
    orch.registry = registry
    orch._router.registry = registry
    orch._get_backend = AsyncMock(return_value=workflow)

    with patch(
        "agent_framework.orchestration.router_orchestrator.get_thread_stage_store",
        return_value=MagicMock(get_last_stage_summary=MagicMock(return_value="")),
    ):
        result = asyncio.run(orch.process_request("退货政策是什么？", thread_id="t1"))

    kwargs = workflow.process_request.await_args.kwargs
    assert "prefilled_execution_plan" in kwargs
    assert kwargs["prefilled_execution_plan"]["source"] == "router_engine"
    assert result["profile"] == PROFILE_WORKFLOW
    assert result["resolved_profile"] == PROFILE_WORKFLOW
