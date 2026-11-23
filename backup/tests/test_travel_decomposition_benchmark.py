"""Travel decomposition benchmark fixtures / scorer / evaluator tests."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from agent_framework.domain.task_planner import TaskPlanner
from agent_framework.optimization.decomposition.evaluator import evaluate_decomposition_benchmark
from agent_framework.optimization.decomposition.fixtures import (
    DependencyExpect,
    load_decomposition_fixtures,
)
from agent_framework.optimization.decomposition.scorer import score_decomposition
from agent_framework.optimization.routing_assignment import routing_assignment_ratio
from domains.travel.prompt_bundle import TravelPrompts
from domains.travel.specs import create_travel_registry_stub


def test_load_decomposition_fixtures():
    fixtures = load_decomposition_fixtures()
    assert fixtures.version == "1.1.0"
    assert fixtures.domain == "travel"
    assert fixtures.locale == "zh"
    assert len(fixtures.cases) == 15
    assert len(fixtures.cases_for_split("dev")) == 2
    assert len(fixtures.cases_for_split("test")) == 6
    assert fixtures.cases_for_split("dev")[0].case_id == "case-08"
    assert fixtures.routing_cases_for_split("dev")
    case_12 = next(case for case in fixtures.cases if case.case_id == "case-12")
    assert case_12.expect_dependency is not None
    assert case_12.expect_dependency.depends_on["T3"] == ["T1", "T2"]


def test_score_slot_groups_accept_synonyms():
    case = next(case for case in load_decomposition_fixtures().cases if case.case_id == "case-04")
    parsed = {
        "totalGoal": "推荐成都川菜馆",
        "subSteps": ["推荐成都人均150元左右川菜馆"],
    }
    score = score_decomposition(parsed, case.expect)
    assert score.slot_ok
    assert score.total >= 0.5


def test_score_dependency_checks_edges():
    expect = load_decomposition_fixtures().cases_for_split("dev")[0].expect
    parsed = {
        "totalGoal": "西安旅行",
        "subSteps": ["查天气", "订酒店", "找美食"],
    }
    score = score_decomposition(
        parsed,
        expect,
        execution_order=["T1", "T2", "T3"],
        depends_map={"T1": [], "T2": ["T1"], "T3": ["T1"]},
        expect_dependency=DependencyExpect(
            depends_on={"T2": ["T1"], "T3": ["T1"]},
        ),
    )
    assert score.dependency_ok


def test_score_routing_assignment_end_to_end():
    case = load_decomposition_fixtures().cases_for_split("dev")[0]
    parsed = {
        "totalGoal": "查询西安天气并推荐酒店和美食",
        "subSteps": [
            "查询西安下周天气预报",
            "推荐西安市中心酒店",
            "推荐西安本地特色美食餐厅",
        ],
    }
    routed_subtasks = [
        {"task_id": "T1", "agent": "WeatherAgent"},
        {"task_id": "T2", "agent": "HotelAgent"},
        {"task_id": "T3", "agent": "RestaurantAgent"},
    ]
    score = score_decomposition(
        parsed,
        case.expect,
        routed_subtasks=routed_subtasks,
        routed_agents=["WeatherAgent", "HotelAgent", "RestaurantAgent"],
        expect_routing=case.expect_routing,
    )
    assert score.total >= 0.8
    assert score.format_ok
    assert score.routing_assignment_ok
    assert score.agent_coverage_ok


def test_routing_assignment_ratio_flexible_when_count_differs():
    case = load_decomposition_fixtures().cases_for_split("dev")[0]
    ratio, ok, _ = routing_assignment_ratio(
        [
            {"task_id": "T1", "agent": "WeatherAgent"},
            {"task_id": "T2", "agent": "HotelAgent"},
            {"task_id": "T3", "agent": "RestaurantAgent"},
            {"task_id": "T4", "agent": "ItineraryAgent"},
        ],
        case.expect_routing,
    )
    assert ratio == 1.0
    assert ok


def test_score_detects_over_expansion():
    weather_case = next(
        case for case in load_decomposition_fixtures().cases if case.case_id == "case-01"
    )
    parsed = {
        "totalGoal": "查询北京天气并规划完整行程",
        "subSteps": [
            "查询北京明天天气",
            "推荐北京酒店",
            "规划北京三日行程",
        ],
    }
    score = score_decomposition(parsed, weather_case.expect)
    assert score.total < 0.8
    assert not score.subtask_count_ok or not score.forbidden_ok


def _mock_llm_responses(responses: list[str]) -> MagicMock:
    llm = MagicMock()
    queue = list(responses)

    async def ainvoke(messages):
        msg = MagicMock()
        msg.content = queue.pop(0) if queue else ""
        return msg

    llm.ainvoke = AsyncMock(side_effect=ainvoke)
    return llm


def test_evaluate_decomposition_benchmark_with_mock_llm():
    fixtures = load_decomposition_fixtures()
    registry = create_travel_registry_stub()
    prompts = TravelPrompts.build(locale="zh", use_optimized=False)
    parallel_dependency = json.dumps(
        {"order": ["T1", "T2", "T3"], "depends_on": {"T2": ["T1"], "T3": ["T1"]}},
        ensure_ascii=False,
    )
    two_step_dependency = json.dumps(
        {"order": ["T1", "T2"], "depends_on": {}},
        ensure_ascii=False,
    )
    xian_routing = json.dumps(
        [
            {
                "task_id": "T1",
                "description": "查询西安下周天气预报",
                "agent": "WeatherAgent",
                "depends_on": [],
            },
            {
                "task_id": "T2",
                "description": "推荐西安市中心酒店",
                "agent": "HotelAgent",
                "depends_on": ["T1"],
            },
            {
                "task_id": "T3",
                "description": "推荐西安本地特色美食餐厅",
                "agent": "RestaurantAgent",
                "depends_on": ["T1"],
            },
        ],
        ensure_ascii=False,
    )
    flight_hotel_routing = json.dumps(
        [
            {
                "task_id": "T1",
                "description": "查询7月1日北京到三亚的航班",
                "agent": "FlightAgent",
                "depends_on": [],
            },
            {
                "task_id": "T2",
                "description": "推荐三亚海棠湾附近酒店",
                "agent": "HotelAgent",
                "depends_on": [],
            },
        ],
        ensure_ascii=False,
    )
    planner = TaskPlanner(
        _mock_llm_responses(
            [
                """
# 目标
查询西安天气并推荐酒店和美食

# 任务拆解
- 查询西安下周天气预报
- 推荐西安市中心酒店
- 推荐西安本地特色美食餐厅
""",
                parallel_dependency,
                xian_routing,
                """
# 目标
查询北京飞三亚航班并推荐海棠湾酒店

# 任务拆解
- 查询7月1日北京到三亚的航班
- 推荐三亚海棠湾附近酒店
""",
                two_step_dependency,
                flight_hotel_routing,
            ]
        ),
        registry,
        prompts,
    )

    async def _run():
        return await evaluate_decomposition_benchmark(
            planner,
            registry=registry,
            fixtures=fixtures,
            split="dev",
        )

    report = asyncio.run(_run())
    assert report.case_count == 2
    assert report.version == "1.1.0"
    assert report.average_score >= 0.8
    assert report.cases[0].case_id == "case-08"
    assert report.cases[0].routed_agents == [
        "WeatherAgent",
        "HotelAgent",
        "RestaurantAgent",
    ]
    assert report.cases[0].score.routing_assignment_ok
    assert report.cases[0].score.dependency_ok
