"""Phase 28：pre_survey_mode + semantic_routing bridge pre_survey。"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from agent_framework.domain.domain_config import DomainConfig
from agent_framework.domain.domain_prompts import DomainPrompts
from agent_framework.domain.pipeline import (
    PRE_SURVEY_MODE_FULL_CH2,
    PRE_SURVEY_MODE_OFF,
    PRE_SURVEY_MODE_ROUTER_PREFILL,
    PipelineConfig,
    normalize_pre_survey_mode,
)
from agent_framework.domain.plugin_registry import get_domain_plugin
from agent_framework.orchestration.fixed_graph.nodes import GraphContext, make_nodes
from agent_framework.orchestration.fixed_graph.state import CentralAgentState
from agent_framework.router.execution_plan_bridge import enrich_execution_plan_pipeline_metadata
from agent_framework.router.plan import AgentCandidate, RoutingPlan
from agent_framework.router.pre_survey_bridge import pre_survey_from_routing_plan
from agent_framework.router.stages.semantic_routing import run_domain_decomposition
from agent_framework.router.stages.task_decomposition import run_task_decomposition


def test_normalize_pre_survey_mode():
    assert normalize_pre_survey_mode("router_prefill") == PRE_SURVEY_MODE_ROUTER_PREFILL
    assert normalize_pre_survey_mode("full_ch2") == PRE_SURVEY_MODE_FULL_CH2
    assert normalize_pre_survey_mode("off") == PRE_SURVEY_MODE_OFF
    assert normalize_pre_survey_mode("unknown") == PRE_SURVEY_MODE_ROUTER_PREFILL


def test_pipeline_runs_pre_survey_node():
    assert PipelineConfig(pre_survey_mode=PRE_SURVEY_MODE_OFF).runs_pre_survey_node is False
    assert PipelineConfig(enable_pre_survey=False).runs_pre_survey_node is False
    assert PipelineConfig(pre_survey_mode=PRE_SURVEY_MODE_FULL_CH2).runs_pre_survey_node is True


def test_travel_plugin_defaults_to_full_ch2():
    plugin = get_domain_plugin("travel")
    pipeline = plugin.build_pipeline(enable_memory=False)
    assert pipeline.resolved_pre_survey_mode == PRE_SURVEY_MODE_FULL_CH2


def test_enrich_execution_plan_pipeline_metadata():
    plan = enrich_execution_plan_pipeline_metadata(
        {"pre_survey": {"source": "router_engine"}, "source": "router_engine"},
        pre_survey_mode=PRE_SURVEY_MODE_FULL_CH2,
    )
    assert plan["pre_survey_mode"] == PRE_SURVEY_MODE_FULL_CH2
    assert plan["pre_survey_source"] == "router_engine"


def test_pre_survey_node_full_ch2_runs_llm():
    registry = MagicMock()
    ctx = GraphContext(
        MagicMock(),
        None,
        registry=registry,
        prompts=DomainPrompts(
            central_agent_system="sys",
            aggregation="agg",
            facts_prompt="facts {task}",
            decomposition_prompt="decomp",
            dependency_system="dep sys",
            dependency_user="dep user",
            agent_routing="route",
        ),
        domain_config=DomainConfig(),
        pipeline=PipelineConfig(pre_survey_mode=PRE_SURVEY_MODE_FULL_CH2),
    )
    ctx.planner = MagicMock()
    ctx.planner.run_pre_survey = AsyncMock(
        return_value={
            "given_facts": ["北京"],
            "facts_to_lookup": [],
            "facts_to_derive": [],
            "educated_guesses": [],
            "raw_text": "survey",
        }
    )
    nodes = make_nodes(ctx)
    prefilled = pre_survey_from_routing_plan(
        RoutingPlan(rewritten_query="q", events=["event-a"], profile="workflow")
    )
    state: CentralAgentState = {
        "user_query": "规划北京三日游",
        "prefilled_pre_survey": prefilled,
        "logs": [],
    }
    result = asyncio.run(nodes["pre_survey"](state))
    ctx.planner.run_pre_survey.assert_called_once()
    assert result["pre_survey"]["given_facts"] == ["北京"]


def test_build_plan_merges_full_ch2_pre_survey():
    registry = MagicMock()
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
        pipeline=PipelineConfig(
            pre_survey_mode=PRE_SURVEY_MODE_FULL_CH2,
            allow_task_planner_decomposition=False,
        ),
    )
    ctx.planner = MagicMock()
    nodes = make_nodes(ctx)
    state: CentralAgentState = {
        "user_query": "规划北京三日游",
        "thread_id": "t1",
        "pre_survey": {
            "given_facts": ["上海"],
            "facts_to_lookup": ["天气"],
            "facts_to_derive": [],
            "educated_guesses": [],
            "raw_text": "full survey",
        },
        "prefilled_execution_plan": {
            "pre_survey": {"source": "router_engine", "given_facts": ["event"]},
            "subtasks": [{"task_id": "T1", "description": "查天气", "agent": "WeatherAgent", "depends_on": []}],
            "execution_order": ["T1"],
            "total_goal": "goal",
            "source": "router_engine",
        },
        "logs": [],
    }
    result = asyncio.run(nodes["build_plan"](state))
    assert result["execution_plan"]["pre_survey"]["given_facts"] == ["上海"]
    assert result["execution_plan"]["pre_survey_source"] == "task_planner_llm"


def test_run_domain_decomposition_passes_router_pre_survey():
    bridge = pre_survey_from_routing_plan(
        RoutingPlan(
            rewritten_query="规划北京三日游",
            events=["查北京天气", "订酒店"],
            candidates=[AgentCandidate("WeatherAgent", 0.9)],
            profile="workflow",
        )
    )
    registry = get_domain_plugin("travel").create_registry()
    mock_llm = MagicMock()

    with patch(
        "agent_framework.router.stages.semantic_routing.create_domain_task_planner"
    ) as create_planner:
        planner = MagicMock()
        planner.run_decomposition = AsyncMock(
            return_value={"totalGoal": "北京三日游", "subSteps": ["查天气", "订酒店"]}
        )
        create_planner.return_value = planner

        goal, steps = asyncio.run(
            run_domain_decomposition(
                mock_llm,
                registry,
                "travel",
                "规划北京三日游",
                pre_survey=bridge,
            )
        )

    planner.run_decomposition.assert_called_once()
    call_pre_survey = planner.run_decomposition.call_args[0][1]
    assert "查北京天气" in call_pre_survey["given_facts"]
    assert goal == "北京三日游"
    assert steps == ["查天气", "订酒店"]


def test_run_task_decomposition_forwards_router_pre_survey():
    bridge = {"given_facts": ["复合行程"], "facts_to_lookup": [], "facts_to_derive": [], "educated_guesses": []}
    registry = get_domain_plugin("travel").create_registry()

    with patch(
        "agent_framework.router.stages.semantic_routing.run_domain_decomposition",
        new_callable=AsyncMock,
    ) as run_decomp:
        run_decomp.return_value = ("goal", ["查天气"])
        with patch(
            "agent_framework.router.stages.semantic_routing.build_semantic_routing_steps",
            new_callable=AsyncMock,
        ) as build_steps:
            build_steps.return_value = []
            asyncio.run(
                run_task_decomposition(
                    MagicMock(),
                    registry,
                    "规划北京三日游",
                    [],
                    domain="travel",
                    router_pre_survey=bridge,
                )
            )
            run_decomp.assert_called_once()
            assert run_decomp.call_args.kwargs["pre_survey"] == bridge
