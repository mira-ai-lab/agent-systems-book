"""Phase 17：en locale + RoutingPlan.steps → FixedGraph execution_plan。"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage

from agent_framework.domain.agent_registry import SubAgentRegistry
from agent_framework.domain.domain_config import DomainConfig
from agent_framework.domain.domain_prompts import DomainPrompts
from agent_framework.orchestration.fixed_graph.nodes import GraphContext, make_nodes
from agent_framework.orchestration.fixed_graph.state import CentralAgentState
from agent_framework.prompts.platform_defaults import get_platform_domain_prompts
from agent_framework.router.execution_plan_bridge import (
    execution_plan_from_routing_plan,
    routing_steps_to_execution_plan,
)
from agent_framework.router.plan import AgentCandidate, RoutingPlan, RoutingStep
from agent_framework.router.prompts.loader import (
    get_classification_prompts,
    get_task_decomposition_prompts,
    load_locale,
)


def test_routing_steps_to_execution_plan_linear_deps():
    plan = routing_steps_to_execution_plan(
        [
            RoutingStep("T1", "Query weather", "WeatherAgent"),
            RoutingStep("T2", "Book hotel", "HotelAgent"),
        ],
        total_goal="Plan a trip",
    )
    assert plan["execution_order"] == ["T1", "T2"]
    assert plan["subtasks"][0]["depends_on"] == []
    assert plan["subtasks"][1]["depends_on"] == ["T1"]
    assert plan["subtasks"][0]["routing_status"] == "router_prefill"
    assert plan["source"] == "router_engine"


def test_execution_plan_from_routing_plan_none_when_empty():
    plan = RoutingPlan(rewritten_query="hello", profile="workflow")
    assert execution_plan_from_routing_plan(plan) is None


def test_build_plan_node_uses_prefilled_execution_plan():
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
    )
    prefilled = routing_steps_to_execution_plan(
        [RoutingStep("T1", "Do task", "A")],
        total_goal="Goal",
    )
    planner = MagicMock()
    planner.build_execution_plan = AsyncMock()
    ctx.planner = planner
    nodes = make_nodes(ctx)
    state: CentralAgentState = {
        "user_query": "Do task",
        "prefilled_execution_plan": prefilled,
        "pre_survey": {},
        "retrieved_memories": [],
        "logs": [],
    }
    result = asyncio.run(nodes["build_plan"](state))
    planner.build_execution_plan.assert_not_called()
    assert result["execution_plan"]["source"] == "router_engine"
    assert result["subtasks"][0]["agent"] == "A"


def test_router_en_locale_loads():
    data = load_locale("en")
    assert "selection" in data
    assert "platform" in data
    prompts = get_classification_prompts("en")
    assert "Available agents" in prompts["prompt_base"]
    decomp = get_task_decomposition_prompts("en")
    assert decomp["keyword_goal"] == "Overall Goal:"


def test_platform_en_domain_prompts():
    prompts = get_platform_domain_prompts("en")
    assert "multi-agent orchestration hub" in prompts["central_agent_system"].lower()


def test_router_orchestrator_passes_prefilled_plan():
    from agent_framework.orchestration.router_orchestrator import RouterOrchestrator

    from agent_framework.domain.pipeline import PipelineConfig

    plugin = MagicMock()
    plugin.create_prompts.return_value.with_platform_defaults.return_value = MagicMock()
    plugin.create_domain_config.return_value = MagicMock()
    plugin.build_pipeline.return_value = PipelineConfig(
        enable_pre_survey=False,
        enable_memory=False,
    )
    plugin.supports_mode.return_value = True

    registry = SubAgentRegistry()
    registry.register("A", lambda: MagicMock(), description="A")
    registry.register("B", lambda: MagicMock(), description="B")

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(content='["task"]'),
            AIMessage(content='[{"name": "A", "score": 0.9}, {"name": "B", "score": 0.85}]'),
            AIMessage(content="整体目标：composite\n子任务：\n- step1\n- step2"),
        ]
    )

    workflow = MagicMock()
    workflow.process_request = AsyncMock(return_value={"final_response": "ok"})

    orch = RouterOrchestrator(
        mock_llm,
        plugin,
        domain="demo",
        enable_memory=False,
    )
    orch.registry = registry
    orch._router.registry = registry
    orch._get_backend = AsyncMock(return_value=workflow)

    with patch(
        "agent_framework.orchestration.router_orchestrator.get_thread_stage_store",
        return_value=MagicMock(get_last_stage_summary=MagicMock(return_value="")),
    ):
        asyncio.run(orch.process_request("composite task", thread_id="t1"))

    kwargs = workflow.process_request.await_args.kwargs
    assert "prefilled_execution_plan" in kwargs
    assert kwargs["prefilled_execution_plan"]["source"] == "router_engine"
    assert len(kwargs["prefilled_execution_plan"]["subtasks"]) == 2
