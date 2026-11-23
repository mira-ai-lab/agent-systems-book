"""Mini-pipeline 规则打分。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from .fixtures import MiniPipelineCase, MiniPipelineExpect


@dataclass
class MiniPipelineScore:
    """一次 mini-pipeline 运行的总分。"""

    total: float
    response_ok: bool
    keyword_ok: bool
    agents_ok: bool
    completion_ok: bool
    completed_steps: int
    invoked_agents: List[str] = field(default_factory=list)
    details: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": round(self.total, 4),
            "response_ok": self.response_ok,
            "keyword_ok": self.keyword_ok,
            "agents_ok": self.agents_ok,
            "completion_ok": self.completion_ok,
            "completed_steps": self.completed_steps,
            "invoked_agents": list(self.invoked_agents),
            "details": list(self.details),
        }


def _invoked_agents(step_results: Dict[str, Any]) -> Set[str]:
    agents: Set[str] = set()
    for item in (step_results or {}).values():
        if not isinstance(item, dict):
            continue
        agent = str(item.get("agent") or "").strip()
        if agent:
            agents.add(agent)
    return agents


def _completed_step_count(step_results: Dict[str, Any], *, min_step_score: float) -> int:
    count = 0
    for item in (step_results or {}).values():
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "").strip().lower()
        score = float(item.get("score") or 0.0)
        if status == "completed" or score >= min_step_score:
            count += 1
    return count


def score_mini_pipeline_run(result: Dict[str, Any], case: MiniPipelineCase) -> MiniPipelineScore:
    """对 mini-pipeline 运行结果按 case.expect 打分（权重对齐 e2e scorer）。"""
    expect: MiniPipelineExpect = case.expect
    details: List[str] = []
    total = 0.0

    final_response = str(result.get("final_response") or "").strip()
    step_results = result.get("step_results") or {}
    invoked = sorted(_invoked_agents(step_results))
    completed = _completed_step_count(step_results, min_step_score=expect.min_step_score)

    response_ok = bool(final_response)
    if response_ok:
        total += 0.10
    else:
        details.append("缺少 final_response")

    if expect.response_keywords:
        missing = [kw for kw in expect.response_keywords if kw not in final_response]
        keyword_ok = not missing
        if keyword_ok:
            total += 0.25
        else:
            details.append(f"回复缺少关键词: {missing}")
    else:
        keyword_ok = True
        total += 0.25

    agents_ok = True
    if expect.required_agents:
        missing_agents = [name for name in expect.required_agents if name not in invoked]
        agents_ok = not missing_agents
        if agents_ok:
            total += 0.50
        else:
            matched = len(expect.required_agents) - len(missing_agents)
            ratio = matched / len(expect.required_agents)
            total += 0.50 * max(0.0, ratio)
            details.append(f"未执行期望 Agent 步骤: {missing_agents}")
    else:
        total += 0.50

    completion_ok = completed >= expect.min_completed_steps
    if completion_ok:
        total += 0.15
    else:
        details.append(
            f"完成步骤 {completed} < 期望最少 {expect.min_completed_steps}"
        )

    return MiniPipelineScore(
        total=min(total, 1.0),
        response_ok=response_ok,
        keyword_ok=keyword_ok,
        agents_ok=agents_ok,
        completion_ok=completion_ok,
        completed_steps=completed,
        invoked_agents=invoked,
        details=details,
    )
