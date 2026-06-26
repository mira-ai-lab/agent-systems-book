"""Structured tool_data scoring tests."""

from __future__ import annotations

from agent_framework.optimization.decomposition.fixtures import ToolDataCheck
from agent_framework.optimization.e2e.tool_data import iter_tool_payloads, score_tool_data_checks


def test_iter_tool_payloads_expands_calls():
    payloads = iter_tool_payloads(
        {
            "calls": [
                {"city": "北京"},
                {"city": "上海"},
            ],
            "count": 2,
        }
    )
    assert len(payloads) == 2


def test_score_tool_data_partial_credit():
    ok, ratio, details = score_tool_data_checks(
        {
            "T1": {"status": "completed", "tool_data": {"city": "北京"}},
            "T2": {"status": "failed", "tool_data": {"error": "x"}},
        },
        [
            ToolDataCheck(task_id="T1", field_contains={"city": ["北京"]}),
            ToolDataCheck(task_id="T2", field_contains={"city": ["杭州"]}),
        ],
    )
    assert not ok
    assert ratio == 0.5
    assert details
