"""子 Agent benchmark 失败样本收集。"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .evaluator import CaseEvalProgressCallback, SingleAgentCaseResult, evaluate_single_agent_case
from .fixtures import SingleAgentCase
from .runtime import AgentSyncBridge


async def collect_single_agent_failures(
    bridge: AgentSyncBridge,
    cases: List[SingleAgentCase],
    *,
    system_prompt_template: str,
    failure_threshold: float,
    on_case_evaluated: Optional[CaseEvalProgressCallback] = None,
) -> List[Tuple[SingleAgentCaseResult, SingleAgentCase]]:
    """返回得分低于阈值的 (result, case) 列表。"""
    failures: List[Tuple[SingleAgentCaseResult, SingleAgentCase]] = []
    for case in cases:
        result = await evaluate_single_agent_case(
            bridge,
            case,
            system_prompt_template=system_prompt_template,
            phase="train_collect",
            on_case_evaluated=on_case_evaluated,
        )
        if result.score.total < failure_threshold:
            failures.append((result, case))
    return failures
