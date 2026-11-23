"""Routing benchmark tests."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from agent_framework.domain.task_planner import TaskPlanner
from agent_framework.optimization.decomposition.fixtures import load_decomposition_fixtures
from agent_framework.optimization.routing.evaluator import evaluate_routing_benchmark
from agent_framework.optimization.routing.prompt_optimizer import extract_agent_routing_prompt
from agent_framework.optimization.routing.scorer import score_routing
from domains.travel.prompt_bundle import TravelPrompts
from domains.travel.specs import create_travel_registry_stub


def test_score_routing_perfect_match():
    fixtures = load_decomposition_fixtures()
    case = next(item for item in fixtures.cases if item.case_id == "case-01")
    subtasks = [
        {
            "task_id": "T1",
            "description": "查询北京明天天气预报",
            "agent": "WeatherAgent",
            "routing_status": "llm",
            "params": {"city": "北京", "date": "明天"},
            "depends_on": [],
        }
    ]
    score = score_routing(subtasks, case.expect_routing)
    assert score.total >= 0.8
    assert score.agent_match_ok


def test_extract_agent_routing_prompt():
    raw = "prefix\n```\nROUTE {agent_team} {subtasks_json} {today} {time_anchor}\n```"
    assert extract_agent_routing_prompt(raw).startswith("ROUTE")


def _mock_llm_responses(responses: list[str]) -> MagicMock:
    llm = MagicMock()
    queue = list(responses)

    async def ainvoke(messages):
        msg = MagicMock()
        msg.content = queue.pop(0) if queue else ""
        return msg

    llm.ainvoke = AsyncMock(side_effect=ainvoke)
    return llm


def test_evaluate_routing_benchmark_with_mock_llm():
    fixtures = load_decomposition_fixtures()
    registry = create_travel_registry_stub()
    prompts = TravelPrompts.build(locale="zh", use_optimized=False)
    good_routing = json.dumps(
        [
            {
                "task_id": "T1",
                "description": "查询西安下周天气预报",
                "agent": "WeatherAgent",
                "params": {"city": "西安"},
                "depends_on": [],
            },
            {
                "task_id": "T2",
                "description": "推荐西安市中心酒店",
                "agent": "HotelAgent",
                "params": {"city": "西安"},
                "depends_on": [],
            },
            {
                "task_id": "T3",
                "description": "推荐西安本地特色美食餐厅",
                "agent": "RestaurantAgent",
                "params": {"location": "西安"},
                "depends_on": [],
            },
        ],
        ensure_ascii=False,
    )
    flight_hotel = json.dumps(
        [
            {
                "task_id": "T1",
                "description": "查询7月1日北京到三亚的航班",
                "agent": "FlightAgent",
                "params": {"departure": "北京", "arrival": "三亚", "date": "7月1日"},
                "depends_on": [],
            },
            {
                "task_id": "T2",
                "description": "推荐三亚海棠湾附近酒店",
                "agent": "HotelAgent",
                "params": {"city": "三亚", "preferences": "海棠湾"},
                "depends_on": [],
            },
        ],
        ensure_ascii=False,
    )
    planner = TaskPlanner(_mock_llm_responses([good_routing, flight_hotel]), registry, prompts)

    async def _run():
        return await evaluate_routing_benchmark(
            planner,
            fixtures=fixtures,
            split="dev",
        )

    report = asyncio.run(_run())
    assert report.case_count == 2
    assert report.average_score >= 0.8
