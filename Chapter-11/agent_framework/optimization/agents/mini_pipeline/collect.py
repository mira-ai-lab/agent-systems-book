"""Mini-pipeline 失败样本收集（供 Agent-B3 textgrad 与诊断）。"""

from __future__ import annotations

from typing import Dict, List, Mapping, Tuple

from agent_framework.optimization.agents.fixtures import SingleAgentCase

from .evaluator import MiniPipelineCaseResult, evaluate_mini_pipeline_case
from .fixtures import MiniPipelineCase
from .runtime import MiniPipelineRunner


async def collect_mini_pipeline_failures(
    runner: MiniPipelineRunner,
    cases: List[MiniPipelineCase],
    *,
    prompt_templates: Mapping[str, str],
    failure_threshold: float,
) -> List[Tuple[MiniPipelineCaseResult, MiniPipelineCase]]:
    """返回 pipeline 总分低于阈值的 (result, case)。"""
    failures: List[Tuple[MiniPipelineCaseResult, MiniPipelineCase]] = []
    for case in cases:
        result = evaluate_mini_pipeline_case(
            runner,
            case,
            prompt_templates=prompt_templates,
        )
        if result.score.total < failure_threshold:
            failures.append((result, case))
    return failures


def collect_agent_step_failures(
    case_results: List[Tuple[MiniPipelineCaseResult, MiniPipelineCase]],
    *,
    agent_name: str,
    step_failure_threshold: float,
) -> List[SingleAgentCase]:
    """从 pipeline 失败样本中提取某 Agent 的低分 step，转为 SingleAgentCase 供 graph 反传。"""
    selected: List[SingleAgentCase] = []
    seen: set[str] = set()

    for result, case in case_results:
        for step in case.steps:
            if step.agent_name != agent_name:
                continue
            step_result = (result.step_results or {}).get(step.step_id) or {}
            step_score = float(step_result.get("score") or 0.0)
            if step_score >= step_failure_threshold:
                continue
            single = step.to_single_agent_case(case_id=case.case_id)
            if single.case_id in seen:
                continue
            seen.add(single.case_id)
            selected.append(single)

    return selected
