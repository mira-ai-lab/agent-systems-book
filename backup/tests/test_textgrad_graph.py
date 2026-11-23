"""Tests for textgrad_graph (Phase B1) planner computation graph."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent_framework.optimization.decomposition.fixtures import load_decomposition_fixtures
from agent_framework.optimization.optimizers.textgrad_graph.graph import PlannerTextGradGraph
from agent_framework.optimization.optimizers.textgrad_graph.loss import build_case_expectation_label


def test_build_case_expectation_label_contains_query():
    case = load_decomposition_fixtures().cases_for_split("dev")[0]
    label = build_case_expectation_label(case)
    assert case.case_id in label
    assert case.query in label


@pytest.mark.textgrad
def test_planner_graph_variables_requires_grad_by_slot():
    pytest.importorskip("textgrad")
    executor_llm = MagicMock()
    optimizer_llm = MagicMock()
    registry = MagicMock()
    registry.get_all_agents_text.return_value = "flight_agent, hotel_agent"
    registry.get_agent_parameters_text.return_value = ""
    registry.resolve_agent.return_value = "flight_agent"
    registry.guess_agent.return_value = None

    from domains.travel.prompt_bundle import TravelPrompts

    prompts = TravelPrompts.build(locale="zh", use_optimized=False)

    decomp_graph = PlannerTextGradGraph.create(
        executor_llm=executor_llm,
        registry=registry,
        locale="zh",
        decomposition_prompt=prompts.decomposition_prompt,
        agent_routing=prompts.agent_routing,
        optimize_slot="decomposition",
        optimizer_llm=optimizer_llm,
    )
    assert decomp_graph.variables.decomposition_prompt.requires_grad is True
    assert decomp_graph.variables.agent_routing.requires_grad is False
    assert len(decomp_graph.trainable_parameters()) == 1

    routing_graph = PlannerTextGradGraph.create(
        executor_llm=executor_llm,
        registry=registry,
        locale="zh",
        decomposition_prompt=prompts.decomposition_prompt,
        agent_routing=prompts.agent_routing,
        optimize_slot="routing",
        optimizer_llm=optimizer_llm,
    )
    assert routing_graph.variables.decomposition_prompt.requires_grad is False
    assert routing_graph.variables.agent_routing.requires_grad is True


@pytest.mark.textgrad
def test_planner_graph_forward_builds_three_step_chain(monkeypatch):
    pytest.importorskip("textgrad")
    from agent_framework.optimization.optimizers.textgrad_graph import graph as graph_module

    case = load_decomposition_fixtures().cases_for_split("dev")[0]

    class FakeBridge:
        def run_decomposition(self, *, decomposition_prompt, user_query, pre_survey):
            return {"totalGoal": "goal", "subSteps": ["查天气", "订酒店"]}

        def run_dependency_analysis(self, sub_steps):
            return ["T1", "T2"], {"T2": ["T1"]}

        def route_to_agents(self, *, agent_routing, sub_steps, execution_order, depends_map):
            return [{"task_id": "T1", "agent": "weather_agent", "routing_status": "llm"}]

        @staticmethod
        def format_pipeline_output(**kwargs):
            import json

            return json.dumps(kwargs, ensure_ascii=False)

    class FakeLossFn:
        def __call__(self, inputs):
            from textgrad import Variable

            return Variable(
                "loss",
                predecessors=[inputs[0]],
                role_description="fake loss",
            )

    fake_engine = MagicMock()
    monkeypatch.setattr(
        graph_module,
        "create_planner_graph_loss_fn",
        lambda _engine: FakeLossFn(),
    )

    from domains.travel.prompt_bundle import TravelPrompts

    prompts = TravelPrompts.build(locale="zh", use_optimized=False)
    tg_graph = PlannerTextGradGraph(
        bridge=FakeBridge(),
        decomposition_prompt=prompts.decomposition_prompt,
        agent_routing=prompts.agent_routing,
        optimize_slot="decomposition",
        engine=fake_engine,
    )

    pipeline_out, loss = tg_graph.forward_case(case)
    assert "routed_subtasks" in pipeline_out.value or "parsed" in pipeline_out.value
    assert loss.value == "loss"
    assert len(pipeline_out.predecessors) >= 1
