"""Agent-B3 mini-pipeline 测试（mock runner，无需 live LLM）。"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from agent_framework.optimization.agent_mini_pipeline import (
    DEFAULT_MINI_PIPELINE_SLOTS,
    parse_mini_pipeline_slots,
)
from agent_framework.optimization.agents.mini_pipeline.fixtures import load_mini_pipeline_cases
from agent_framework.optimization.agents.mini_pipeline.scorer import score_mini_pipeline_run


def test_mini_pipeline_fixtures_train_dev_split():
    fixtures = load_mini_pipeline_cases()
    train = fixtures.cases_for_split("train")
    dev = fixtures.cases_for_split("dev")
    assert len(train) == 1
    assert len(dev) == 1
    assert train[0].case_id == "mini-sanya-trip-train"
    assert dev[0].case_id == "mini-xian-food-dev"
    assert len(train[0].steps) == 3


def test_parse_mini_pipeline_slots_default():
    assert parse_mini_pipeline_slots("default") == list(DEFAULT_MINI_PIPELINE_SLOTS)


def test_parse_mini_pipeline_slots_custom():
    slots = parse_mini_pipeline_slots("WeatherAgent, RestaurantAgent")
    assert slots == ["WeatherAgent", "RestaurantAgent"]


def test_score_mini_pipeline_run_pass():
    case = load_mini_pipeline_cases().cases_for_split("train")[0]
    score = score_mini_pipeline_run(
        {
            "final_response": "三亚天气晴朗，推荐海棠湾酒店，北京飞三亚航班如下。",
            "step_results": {
                "S1": {"agent": "WeatherAgent", "status": "completed", "score": 0.9},
                "S2": {"agent": "HotelAgent", "status": "completed", "score": 0.85},
                "S3": {"agent": "FlightAgent", "status": "completed", "score": 0.88},
            },
        },
        case,
    )
    assert score.total >= 0.8
    assert score.agents_ok
    assert score.completion_ok


def test_score_mini_pipeline_run_missing_agent():
    case = load_mini_pipeline_cases().cases_for_split("train")[0]
    score = score_mini_pipeline_run(
        {
            "final_response": "三亚天气不错。",
            "step_results": {
                "S1": {"agent": "WeatherAgent", "status": "completed", "score": 0.9},
            },
        },
        case,
    )
    assert score.total < 0.8
    assert not score.agents_ok


def test_mini_pipeline_step_to_single_agent_case():
    case = load_mini_pipeline_cases().cases_for_split("train")[0]
    step = case.steps[0]
    single = step.to_single_agent_case(case_id=case.case_id)
    assert single.agent_name == "WeatherAgent"
    assert single.tool == "get_weather_forecast"
    assert "三亚" in single.user_query


@pytest.mark.textgrad
def test_mini_pipeline_graph_forward_with_mock_bridge(monkeypatch):
    """B3 反传仍用单节点 graph，此处验证 step case 可 forward。"""
    pytest.importorskip("textgrad")
    from agent_framework.optimization.optimizers.textgrad_agent import graph as graph_module

    case = load_mini_pipeline_cases().cases_for_split("train")[0]
    step_case = case.steps[2].to_single_agent_case(case_id=case.case_id)

    class FakeBridge:
        def invoke(self, **kwargs):
            return {
                "messages": [
                    type("M", (), {"type": "ai", "content": "北京飞三亚航班如下"})(),
                    type("M", (), {"type": "tool", "name": "search_flights", "content": "{}"})(),
                ]
            }

        def format_agent_output(self, state):
            return json.dumps(
                {"final_response": "北京飞三亚航班如下", "invoked_tools": ["search_flights"]}
            )

    class FakeLossFn:
        def __call__(self, inputs):
            from textgrad import Variable

            return Variable("pipeline-loss", predecessors=[inputs[0]], role_description="fake")

    monkeypatch.setattr(graph_module, "create_agent_graph_loss_fn", lambda _e: FakeLossFn())

    from agent_framework.optimization.agents.runtime import get_agent_prompt_template
    from agent_framework.optimization.optimizers.textgrad_agent.graph import SingleAgentTextGradGraph

    template = get_agent_prompt_template("FlightAgent", locale="zh")
    graph = SingleAgentTextGradGraph(
        bridge=FakeBridge(),
        system_prompt_template=template,
        agent_name="FlightAgent",
        engine=MagicMock(),
    )

    out, loss = graph.forward_case(step_case)
    assert "search_flights" in out.value or "航班" in out.value
    assert loss.value == "pipeline-loss"
