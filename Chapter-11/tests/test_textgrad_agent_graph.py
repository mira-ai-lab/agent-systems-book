"""Tests for Agent-B1 textgrad_agent graph (mock bridge, no live LLM)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from agent_framework.optimization.agents.fixtures import load_single_agent_cases
from agent_framework.optimization.agents.scorer import score_single_agent_run
from agent_framework.optimization.optimizers.textgrad_agent.graph import SingleAgentTextGradGraph


def test_single_agent_fixtures_train_dev_split():
    fixtures = load_single_agent_cases()
    train = fixtures.cases_for_split("train", agent_name="FlightAgent")
    dev = fixtures.cases_for_split("dev", agent_name="FlightAgent")
    assert len(train) == 3
    assert len(dev) == 2
    assert train[0].case_id == "flight-beijing-sanya-jun25"
    assert dev[0].case_id == "flight-shanghai-chengdu-jun30"
    assert {c.case_id for c in train} == {
        "flight-beijing-sanya-jun25",
        "flight-guangzhou-beijing-jul05",
        "flight-shenzhen-hangzhou-jul10",
    }
    assert {c.case_id for c in dev} == {
        "flight-shanghai-chengdu-jun30",
        "flight-chengdu-xian-jul15",
    }


def test_score_single_agent_run_pass():
    case = load_single_agent_cases().cases_for_agent("FlightAgent")[0]
    state = {
        "messages": [
            type("M", (), {"type": "ai", "content": "已查询北京到三亚航班，推荐 CA1357。"})(),
            type("M", (), {"type": "tool", "name": "search_flights", "content": "{}"})(),
        ]
    }
    score = score_single_agent_run(state, case)
    assert score.tool_called_ok
    assert score.total >= 0.6


@pytest.mark.textgrad
def test_single_agent_graph_forward_with_mock_bridge(monkeypatch):
    pytest.importorskip("textgrad")
    from agent_framework.optimization.optimizers.textgrad_agent import graph as graph_module

    case = load_single_agent_cases().cases_for_agent("FlightAgent")[0]

    class FakeBridge:
        def invoke(self, **kwargs):
            return {
                "messages": [
                    type("M", (), {"type": "ai", "content": "北京飞三亚航班如下"})(),
                    type("M", (), {"type": "tool", "name": "search_flights", "content": "{}"})(),
                ]
            }

        def format_agent_output(self, state):
            return json.dumps({"final_response": "北京飞三亚航班如下", "invoked_tools": ["search_flights"]})

    class FakeLossFn:
        def __call__(self, inputs):
            from textgrad import Variable

            return Variable("agent-loss", predecessors=[inputs[0]], role_description="fake")

    monkeypatch.setattr(graph_module, "create_agent_graph_loss_fn", lambda _e: FakeLossFn())

    from agent_framework.optimization.agents.runtime import get_agent_prompt_template

    template = get_agent_prompt_template("FlightAgent", locale="zh")
    graph = SingleAgentTextGradGraph(
        bridge=FakeBridge(),
        system_prompt_template=template,
        agent_name="FlightAgent",
        engine=MagicMock(),
    )

    out, loss = graph.forward_case(case)
    assert "search_flights" in out.value or "北京" in out.value
    assert loss.value == "agent-loss"
    assert graph.variables.system_prompt.requires_grad is True
