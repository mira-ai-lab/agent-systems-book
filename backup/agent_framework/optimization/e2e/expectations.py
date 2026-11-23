"""Resolve end-to-end benchmark expectations from shared travel fixtures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from agent_framework.optimization.decomposition.fixtures import DecompositionBenchmarkCase, ToolDataCheck


@dataclass(frozen=True)
class E2eExpect:
    required_response_keywords: List[str] = field(default_factory=list)
    required_response_slot_groups: List[List[str]] = field(default_factory=list)
    forbidden_response_keywords: List[str] = field(default_factory=list)
    required_agents: List[str] = field(default_factory=list)
    min_completed_subtasks: int = 1
    require_final_response: bool = True
    tool_checks: List[ToolDataCheck] = field(default_factory=list)


def resolve_e2e_expect(case: DecompositionBenchmarkCase) -> E2eExpect:
    """Derive E2E expectations from planner-level fixture fields."""
    required_agents: List[str] = []
    if case.expect_routing:
        required_agents = [item.expected_agent for item in case.expect_routing.assignments]
    elif case.expect.mappable_agents:
        required_agents = list(case.expect.mappable_agents)

    min_completed = case.expect.min_subtasks
    if case.expect_routing:
        min_completed = max(min_completed, len(case.expect_routing.assignments))

    slot_groups = list(case.expect.required_slot_groups)
    keywords = list(case.expect.required_keywords)
    if not slot_groups and keywords:
        slot_groups = [[keyword] for keyword in keywords]

    return E2eExpect(
        required_response_keywords=keywords,
        required_response_slot_groups=slot_groups,
        forbidden_response_keywords=list(case.expect.forbidden_keywords),
        required_agents=required_agents,
        min_completed_subtasks=max(1, min_completed),
        tool_checks=list(case.expect.tool_checks),
    )
