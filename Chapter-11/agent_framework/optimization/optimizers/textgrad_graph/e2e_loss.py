"""E2E graph loss 模板与期望标签。"""

from __future__ import annotations

import json
from typing import List

from agent_framework.optimization.decomposition.fixtures import DecompositionBenchmarkCase
from agent_framework.optimization.e2e.expectations import resolve_e2e_expect

E2E_GRAPH_EVAL_INSTRUCTION = """You evaluate a full travel orchestration run against end-to-end benchmark expectations.

The run includes routing, sub-agent execution, and final_response aggregation.
Compare the actual orchestration output with the expected specification.
Identify gaps in invoked agents, completed subtasks, and final response content.
Be specific so planner prompt improvements can fix similar failures.
"""

E2E_GRAPH_ROLE_DESCRIPTIONS = [
    "actual e2e orchestration output",
    "e2e benchmark expectation specification",
]


def build_e2e_expectation_label(case: DecompositionBenchmarkCase) -> str:
    """将 E2E 期望整理为 MultiFieldEvaluation 的 label 文本。"""
    expect = resolve_e2e_expect(case)
    lines = [
        f"case_id: {case.case_id}",
        f"query: {case.query}",
        f"required_agents: {expect.required_agents}",
        f"min_completed_subtasks: {expect.min_completed_subtasks}",
        f"require_final_response: {expect.require_final_response}",
    ]
    if expect.required_response_keywords:
        lines.append(f"required_response_keywords: {expect.required_response_keywords}")
    if expect.required_response_slot_groups:
        lines.append(
            "required_response_slot_groups: "
            + json.dumps(expect.required_response_slot_groups, ensure_ascii=False)
        )
    if expect.forbidden_response_keywords:
        lines.append(f"forbidden_response_keywords: {expect.forbidden_response_keywords}")
    return "\n".join(lines)


def create_e2e_graph_loss_fn(engine):
    """创建比较 E2E 输出与 benchmark 期望的 MultiFieldEvaluation。"""
    from agent_framework.optimization.optimizers.textgrad_lib._import import require_textgrad

    _, Variable, _, _ = require_textgrad()
    from textgrad.loss import MultiFieldEvaluation

    instruction = Variable(
        E2E_GRAPH_EVAL_INSTRUCTION,
        requires_grad=False,
        role_description="e2e graph evaluation instruction",
    )
    return MultiFieldEvaluation(
        instruction,
        E2E_GRAPH_ROLE_DESCRIPTIONS,
        engine,
    )


def format_e2e_failure_cases_for_log(
    failures: List[tuple],
) -> str:
    blocks = []
    for result, case in failures:
        blocks.append(
            f"case_id={case.case_id}\n"
            f"query={case.query}\n"
            f"score={result.score.total:.3f}\n"
            f"details={'; '.join(result.score.details) or 'none'}\n"
            f"invoked_agents={result.invoked_agents}\n"
            f"response_preview={result.final_response[:200]}"
        )
    return "\n\n".join(blocks)
