"""TaskPlanner 路由与规划（Mock LLM，不调用真实 API）。"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_framework.domain.domain_config import DomainConfig
from agent_framework.domain.task_planner import TaskPlanner
from domains.travel.prompt_bundle import TravelPrompts
from domains.travel.specs import create_travel_registry_stub


def _mock_llm(responses: list[str]) -> MagicMock:
    """按调用顺序返回预设 content 的 LLM mock。"""
    llm = MagicMock()
    queue = list(responses)

    async def ainvoke(messages):
        content = queue.pop(0) if queue else ""
        msg = MagicMock()
        msg.content = content
        return msg

    llm.ainvoke = AsyncMock(side_effect=ainvoke)
    return llm


@pytest.fixture
def planner() -> TaskPlanner:
    registry = create_travel_registry_stub()
    prompts = TravelPrompts.build()
    domain_config = DomainConfig(enable_guess_agent=True)
    llm = _mock_llm([])
    return TaskPlanner(llm, registry, prompts, domain_config)


def test_route_to_agents_injects_time_anchor(planner: TaskPlanner):
    captured: list = []

    async def ainvoke(messages):
        captured.append(messages)
        msg = MagicMock()
        msg.content = json.dumps(
            [{"task_id": "T1", "agent": "FlightAgent", "params": {"date": "2026-07-01"}}],
            ensure_ascii=False,
        )
        return msg

    planner.llm.ainvoke = AsyncMock(side_effect=ainvoke)
    asyncio.run(planner.route_to_agents(["查7月1日北京飞三亚航班"], ["T1"], {"T1": []}))

    from domains.travel.plan_context import build_time_anchor

    prompt_text = captured[0][0].content
    today = build_time_anchor()["today"]
    assert today in prompt_text
    assert "{time_anchor}" not in prompt_text
    assert "禁止输出 2024" in prompt_text or "2024" in prompt_text


def test_route_to_agents_with_llm_routing(planner: TaskPlanner):
    planner.llm = _mock_llm([
        json.dumps([
            {
                "task_id": "T1",
                "description": "查北京明天天气",
                "agent": "WeatherAgent",
                "params": {"city": "北京"},
                "depends_on": [],
            }
        ], ensure_ascii=False)
    ])
    subtasks = asyncio.run(
        planner.route_to_agents(["查北京明天天气"], ["T1"], {"T1": []})
    )
    assert len(subtasks) == 1
    assert subtasks[0]["agent"] == "WeatherAgent"
    assert subtasks[0]["routing_status"] == "llm"
    assert subtasks[0]["params"]["city"] == "北京"


def test_route_to_agents_guess_fallback(planner: TaskPlanner):
    planner.llm = _mock_llm(['[{"task_id": "T1", "description": "查天气"}]'])
    subtasks = asyncio.run(
        planner.route_to_agents(["查北京明天天气"], ["T1"], {"T1": []})
    )
    assert subtasks[0]["agent"] == "WeatherAgent"
    assert subtasks[0]["routing_status"] == "guess_agent"


def test_route_to_agents_routing_fallback():
    registry = create_travel_registry_stub()
    prompts = TravelPrompts.build()
    domain_config = DomainConfig(
        routing_fallback="WeatherAgent",
        enable_guess_agent=False,
    )
    planner = TaskPlanner(_mock_llm(['[{"task_id": "T1"}]']), registry, prompts, domain_config)
    subtasks = asyncio.run(
        planner.route_to_agents(["随便问问"], ["T1"], {"T1": []})
    )
    assert subtasks[0]["agent"] == "WeatherAgent"
    assert subtasks[0]["routing_status"] == "routing_fallback"


def test_build_execution_plan_end_to_end(planner: TaskPlanner):
    decomp = """
# 目标
查询上海天气

# 任务拆解
- 查询上海明天天气预报
"""
    routing = json.dumps([
        {
            "task_id": "T1",
            "description": "查询上海明天天气预报",
            "agent": "WeatherAgent",
            "params": {"city": "上海"},
            "depends_on": [],
        }
    ], ensure_ascii=False)
    planner.llm = _mock_llm([decomp, '{"1": "T1"}', routing])

    plan = asyncio.run(
        planner.build_execution_plan(
            "上海明天天气怎么样",
            {"given_facts": [], "facts_to_lookup": ["上海天气"]},
            [],
        )
    )
    assert plan["total_goal"]
    assert len(plan["subtasks"]) == 1
    assert plan["subtasks"][0]["agent"] == "WeatherAgent"
    assert plan["execution_order"] == ["T1"]
