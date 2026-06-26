"""Shared E2E rule-scorer specification (rollback + TextGrad loss alignment)."""

from __future__ import annotations

import json
from typing import List, Optional

from agent_framework.optimization.decomposition.fixtures import DecompositionBenchmarkCase

from .expectations import E2eExpect, resolve_e2e_expect


def format_e2e_rule_checklist(expect: E2eExpect) -> str:
    """Human/LLM-readable checklist matching ``score_e2e_run`` weights."""
    keyword_weight = "0.15" if expect.tool_checks else "0.25"
    lines = [
        "Rule scorer checklist (dev rollback uses these exact rules, total=1.0):",
        "- 0.10: final_response must be non-empty",
    ]

    if expect.required_response_slot_groups:
        lines.append(
            f"- {keyword_weight}: each required_response_slot_group must match in "
            "(final_response + all subtask agent_summary); groups are AND, tokens in a group are OR"
        )
        lines.append(
            "  slot_groups: "
            + json.dumps(expect.required_response_slot_groups, ensure_ascii=False)
        )
    elif expect.required_response_keywords:
        lines.append(
            f"- {keyword_weight}: each required_response_keyword must appear in "
            "(final_response + all subtask agent_summary)"
        )
        lines.append(f"  keywords: {expect.required_response_keywords}")
    else:
        lines.append(f"- {keyword_weight}: no keyword requirement (auto pass)")

    if expect.tool_checks:
        lines.append("- 0.10: structured tool_data field_contains per subtask (forbid_error=true):")
        for check in expect.tool_checks:
            lines.append(
                f"  - {check.task_id}: "
                + json.dumps(check.field_contains, ensure_ascii=False)
            )

    if expect.forbidden_response_keywords:
        lines.append(
            "- 0.15: forbidden_response_keywords must NOT appear in "
            "(final_response + all subtask agent_summary)"
        )
        lines.append(f"  forbidden: {expect.forbidden_response_keywords}")
    else:
        lines.append("- 0.15: no forbidden keywords (auto pass)")

    if expect.required_agents:
        lines.append(
            f"- 0.35: subtask_results[*].agent must include each of {expect.required_agents}"
        )
    else:
        lines.append("- 0.35: no required agents (auto pass)")

    lines.append(
        f"- 0.15: count(subtask status in completed|ok) >= {expect.min_completed_subtasks}"
    )
    return "\n".join(lines)


def build_e2e_expectation_label(
    case: DecompositionBenchmarkCase,
    *,
    rule_failures: Optional[List[str]] = None,
) -> str:
    """E2E benchmark label for MultiFieldEvaluation (aligned with rule scorer)."""
    expect = resolve_e2e_expect(case)
    lines = [
        f"case_id: {case.case_id}",
        f"query: {case.query}",
        "",
        format_e2e_rule_checklist(expect),
    ]
    if rule_failures:
        lines.append("")
        lines.append("rule_scorer_failures_on_this_run (address these first):")
        for item in rule_failures:
            lines.append(f"  - {item}")
    return "\n".join(lines)
