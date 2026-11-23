"""E2E graph loss 模板与期望标签。"""

from __future__ import annotations

from typing import List

from agent_framework.optimization.decomposition.fixtures import DecompositionBenchmarkCase
from agent_framework.optimization.e2e.rules import build_e2e_expectation_label

E2E_GRAPH_EVAL_INSTRUCTION = """You evaluate a full travel orchestration run against end-to-end benchmark expectations.

The run includes routing, sub-agent execution, and final_response aggregation.
Compare the actual orchestration output with the expected specification.
The label includes a rule_scorer_checklist that matches dev rollback scoring — prioritize fixing
rule_scorer_failures_on_this_run when present.
Identify gaps in invoked agents, completed subtasks, and response content (final_response and subtask summaries).
Be specific so planner prompt improvements can fix similar failures.
"""

E2E_GRAPH_ROLE_DESCRIPTIONS = [
    "actual e2e orchestration output",
    "e2e benchmark expectation specification",
]

# Re-export for backward compatibility
__all__ = [
    "E2E_GRAPH_EVAL_INSTRUCTION",
    "E2E_GRAPH_ROLE_DESCRIPTIONS",
    "build_e2e_expectation_label",
    "create_e2e_graph_loss_fn",
    "format_e2e_failure_cases_for_log",
]


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
