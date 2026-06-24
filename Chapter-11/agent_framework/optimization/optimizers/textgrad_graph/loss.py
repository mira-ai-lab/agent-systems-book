"""Benchmark 期望标签与 graph loss 构造。"""

from __future__ import annotations

import json
from typing import Any, List

from agent_framework.optimization.decomposition.fixtures import DecompositionBenchmarkCase

from .prompts import PLANNER_GRAPH_EVAL_INSTRUCTION, PLANNER_GRAPH_ROLE_DESCRIPTIONS


def build_case_expectation_label(case: DecompositionBenchmarkCase) -> str:
    """将 fixture 期望整理为 TextLoss / MultiFieldEvaluation 的 label 文本。"""
    lines = [
        f"case_id: {case.case_id}",
        f"query: {case.query}",
        f"min_subtasks: {case.expect.min_subtasks}",
        f"max_subtasks: {case.expect.max_subtasks}",
    ]
    if case.expect.required_keywords:
        lines.append(f"required_keywords: {case.expect.required_keywords}")
    if case.expect.forbidden_keywords:
        lines.append(f"forbidden_keywords: {case.expect.forbidden_keywords}")
    if case.expect.mappable_agents:
        lines.append(f"mappable_agents: {case.expect.mappable_agents}")

    if case.expect_dependency:
        lines.append(
            "expected_dependency: "
            + json.dumps(
                {
                    "depends_on": case.expect_dependency.depends_on,
                    "execution_order": case.expect_dependency.execution_order,
                },
                ensure_ascii=False,
            )
        )

    if case.expect_routing:
        assignments = [
            {"task_id": item.task_id, "expected_agent": item.expected_agent}
            for item in case.expect_routing.assignments
        ]
        lines.append("expected_routing: " + json.dumps(assignments, ensure_ascii=False))

    return "\n".join(lines)


def create_planner_graph_loss_fn(engine):
    """创建比较 pipeline 输出与 benchmark 期望的 MultiFieldEvaluation。"""
    from agent_framework.optimization.optimizers.textgrad_lib._import import require_textgrad

    _, Variable, _, _ = require_textgrad()
    from textgrad.loss import MultiFieldEvaluation

    instruction = Variable(
        PLANNER_GRAPH_EVAL_INSTRUCTION,
        requires_grad=False,
        role_description="planner graph evaluation instruction",
    )
    return MultiFieldEvaluation(
        instruction,
        PLANNER_GRAPH_ROLE_DESCRIPTIONS,
        engine,
    )


def format_failure_cases_for_log(cases: List[DecompositionBenchmarkCase]) -> str:
    return "\n\n".join(build_case_expectation_label(case) for case in cases)
