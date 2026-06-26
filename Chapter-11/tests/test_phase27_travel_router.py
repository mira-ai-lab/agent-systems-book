"""Phase 27：Router travel 语义拆解（TaskPlanner agent_routing）。"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage

from agent_framework.domain.plugin_registry import get_domain_plugin
from agent_framework.router.config import RouterConfig
from agent_framework.router.engine import RouterEngine
from agent_framework.router.execution_plan_bridge import routing_steps_to_execution_plan
from agent_framework.router.plan import RoutingStep
from agent_framework.router.stages.semantic_routing import (
    SEMANTIC_ROUTING_DOMAINS,
    build_semantic_routing_steps,
    should_use_semantic_routing,
)
from agent_framework.router.stages.task_decomposition import run_task_decomposition


def _travel_registry():
    return get_domain_plugin("travel").create_registry()


def test_should_use_semantic_routing_travel_only():
    cfg = RouterConfig()
    assert should_use_semantic_routing("travel", cfg)
    assert not should_use_semantic_routing("demo", cfg)
    assert not should_use_semantic_routing(
        "travel",
        RouterConfig(semantic_task_routing=False),
    )
    assert "travel" in SEMANTIC_ROUTING_DOMAINS


def test_build_semantic_routing_steps_assigns_travel_agents():
    dependency = json.dumps(
        {
            "order": ["T1", "T2", "T3"],
            "depends_on": {"T2": ["T1"], "T3": ["T1", "T2"]},
        },
        ensure_ascii=False,
    )
    routing = json.dumps(
        [
            {"task_id": "T1", "description": "查询北京天气", "agent": "WeatherAgent", "depends_on": []},
            {"task_id": "T2", "description": "推荐酒店", "agent": "HotelAgent", "depends_on": ["T1"]},
            {"task_id": "T3", "description": "生成行程", "agent": "ItineraryAgent", "depends_on": ["T1", "T2"]},
        ],
        ensure_ascii=False,
    )
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(content=dependency),
            AIMessage(content=routing),
        ]
    )
    registry = _travel_registry()
    steps = asyncio.run(
        build_semantic_routing_steps(
            mock_llm,
            registry,
            "travel",
            ["查询北京天气", "推荐酒店", "生成行程"],
        )
    )
    assert [s.agent for s in steps] == ["WeatherAgent", "HotelAgent", "ItineraryAgent"]
    assert steps[2].depends_on == ("T1", "T2")


def test_run_task_decomposition_travel_uses_semantic_routing():
    decomp = """
# 目标
规划北京三日游

# 任务拆解
- 查询北京天气
- 推荐酒店
"""
    dependency = '{"order": ["T1", "T2"], "depends_on": {"T2": ["T1"]}}'
    routing = json.dumps(
        [
            {"task_id": "T1", "agent": "WeatherAgent", "description": "查询北京天气"},
            {"task_id": "T2", "agent": "HotelAgent", "description": "推荐酒店", "depends_on": ["T1"]},
        ],
        ensure_ascii=False,
    )
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(content=decomp),
            AIMessage(content=dependency),
            AIMessage(content=routing),
        ]
    )
    registry = _travel_registry()
    goal, steps = asyncio.run(
        run_task_decomposition(
            mock_llm,
            registry,
            "规划北京三日游，查天气并订酒店",
            [],
            domain="travel",
            config=RouterConfig(),
        )
    )
    assert "北京" in goal
    assert len(steps) == 2
    assert steps[0].agent == "WeatherAgent"
    assert steps[1].agent == "HotelAgent"
    assert steps[1].depends_on == ("T1",)


def test_execution_plan_bridge_preserves_semantic_depends_on():
    steps = [
        RoutingStep("T1", "查天气", "WeatherAgent"),
        RoutingStep("T2", "订酒店", "HotelAgent", depends_on=("T1",)),
    ]
    plan = routing_steps_to_execution_plan(steps, total_goal="北京三日游")
    assert plan["subtasks"][1]["depends_on"] == ["T1"]
    assert plan["subtasks"][0]["depends_on"] == []


def test_router_engine_travel_semantic_routing_stage():
    decomp = """
# 目标
规划北京三日游

# 任务拆解
- 查询北京天气
- 推荐酒店
"""
    dependency = '{"order": ["T1", "T2"], "depends_on": {}}'
    routing = json.dumps(
        [
            {"task_id": "T1", "agent": "WeatherAgent", "description": "查询北京天气"},
            {"task_id": "T2", "agent": "HotelAgent", "description": "推荐酒店"},
        ],
        ensure_ascii=False,
    )
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(content=decomp),
            AIMessage(content=dependency),
            AIMessage(content=routing),
        ]
    )
    config = RouterConfig(
        enable_history_gate=False,
        enable_interaction_rewrite=False,
        enable_extraction=False,
        enable_knowledge_routing=False,
        enable_classification=False,
        enable_instruction_build=False,
    )
    registry = _travel_registry()
    plan = asyncio.run(
        RouterEngine(
            mock_llm,
            registry,
            config=config,
            domain="travel",
        ).route("规划北京三日游", force_profile="workflow")
    )
    assert plan.profile == "workflow"
    assert len(plan.steps) == 2
    assert plan.steps[0].agent == "WeatherAgent"
    assert plan.steps[1].agent == "HotelAgent"
    assert "task_decomposition" in plan.metadata["stages"]
    assert "semantic_routing" in plan.metadata["stages"]
