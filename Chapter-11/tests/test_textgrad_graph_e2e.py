"""Tests for Phase B2 E2E graph optimization."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent_framework.optimization.decomposition.fixtures import load_decomposition_fixtures
from agent_framework.optimization.objective import parse_optimization_objective
from agent_framework.optimization.optimizers.textgrad_graph.e2e_loss import build_e2e_expectation_label
from agent_framework.optimization.optimizers.textgrad_graph.e2e_graph import PlannerPromptE2eGraph


def test_parse_optimization_objective():
    assert parse_optimization_objective("l1_l2") == "l1_l2"
    assert parse_optimization_objective("e2e") == "e2e"


def test_parse_optimization_objective_invalid():
    with pytest.raises(ValueError, match="不支持的 objective"):
        parse_optimization_objective("foo")


def test_build_e2e_expectation_label_contains_agents():
    case = load_decomposition_fixtures().cases_for_split("dev")[0]
    label = build_e2e_expectation_label(case)
    assert case.case_id in label
    assert "required_agents" in label


@pytest.mark.textgrad
def test_e2e_graph_forward_with_mock_bridge(monkeypatch):
    pytest.importorskip("textgrad")
    from agent_framework.optimization.optimizers.textgrad_graph import e2e_graph as e2e_graph_module

    case = load_decomposition_fixtures().cases_for_split("dev")[0]

    class FakeBridge:
        def process_request(self, **kwargs):
            return {
                "final_response": "西安天气、酒店和美食推荐如下。",
                "subtask_results": {
                    "T1": {"agent": "WeatherAgent", "status": "completed"},
                    "T2": {"agent": "HotelAgent", "status": "completed"},
                    "T3": {"agent": "RestaurantAgent", "status": "completed"},
                },
                "orchestration_mode": "fixed_graph",
                "trace_id": "t-1",
            }

        def format_e2e_output(self, result):
            import json

            return json.dumps(result, ensure_ascii=False)

    class FakeLossFn:
        def __call__(self, inputs):
            from textgrad import Variable

            return Variable(
                "e2e-loss",
                predecessors=[inputs[0]],
                role_description="fake e2e loss",
            )

    monkeypatch.setattr(
        e2e_graph_module,
        "create_e2e_graph_loss_fn",
        lambda _engine: FakeLossFn(),
    )

    from domains.travel.prompt_bundle import TravelPrompts

    prompts = TravelPrompts.build(locale="zh", use_optimized=False)
    graph = PlannerPromptE2eGraph(
        bridge=FakeBridge(),
        decomposition_prompt=prompts.decomposition_prompt,
        agent_routing=prompts.agent_routing,
        optimize_slot="decomposition",
        engine=MagicMock(),
    )

    e2e_out, loss = graph.forward_case(case)
    assert "final_response" in e2e_out.value
    assert loss.value == "e2e-loss"
    assert graph.variables.decomposition_prompt.requires_grad is True
