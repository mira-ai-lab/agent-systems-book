"""Tests for planner pipeline slot parsing and orchestration tests."""

from __future__ import annotations

import pytest

from agent_framework.optimization.objective import parse_optimization_objective
from agent_framework.optimization.planner_pipeline import parse_planner_slots


def test_parse_planner_slots_all():
    assert parse_planner_slots("all") == ["decomposition", "routing"]


def test_parse_planner_slots_single():
    assert parse_planner_slots("routing") == ["routing"]
    assert parse_planner_slots("decomposition") == ["decomposition"]


def test_parse_planner_slots_combo():
    assert parse_planner_slots("decomposition,routing") == ["decomposition", "routing"]


def test_parse_planner_slots_invalid():
    with pytest.raises(ValueError, match="不支持的 slot"):
        parse_planner_slots("foo")


def test_planner_pipeline_accepts_textgrad_graph_backend():
    from agent_framework.optimization.planner_pipeline import OptimizerBackend

    backend: OptimizerBackend = "textgrad_graph"
    assert backend == "textgrad_graph"


def test_parse_optimization_objective_e2e():
    assert parse_optimization_objective("e2e") == "e2e"


@pytest.mark.textgrad
def test_routing_prompt_variable_keeps_placeholders():
    pytest.importorskip("textgrad")
    from agent_framework.optimization.optimizers.textgrad_lib.adapter import (
        read_routing_prompt_value,
        routing_prompt_variable,
    )

    prompt = "团队: {agent_team}\n子任务: {subtasks_json}\n今天: {today}\n{time_anchor}"
    var = routing_prompt_variable(prompt)
    assert read_routing_prompt_value(var) == prompt
