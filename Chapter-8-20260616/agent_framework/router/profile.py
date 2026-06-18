"""执行 Profile：企业路由引擎 → 编排后端映射。"""

from __future__ import annotations

from typing import List, Literal, Optional

from agent_framework.orchestration.protocol import MODE_FIXED_GRAPH, MODE_SUPERVISOR, OrchestrationMode
from agent_framework.router.plan import AgentCandidate

ExecutionProfile = Literal["auto", "workflow", "adaptive", "hybrid"]

PROFILE_AUTO: ExecutionProfile = "auto"
PROFILE_WORKFLOW: ExecutionProfile = "workflow"
PROFILE_ADAPTIVE: ExecutionProfile = "adaptive"
PROFILE_HYBRID: ExecutionProfile = "hybrid"

STRONG_CANDIDATE_THRESHOLD = 0.5


def normalize_profile(profile: Optional[str]) -> ExecutionProfile:
    value = (profile or PROFILE_WORKFLOW).strip() or PROFILE_WORKFLOW
    if value not in (PROFILE_AUTO, PROFILE_WORKFLOW, PROFILE_ADAPTIVE, PROFILE_HYBRID):
        raise ValueError(
            f"不支持的 profile='{value}'，可选: auto, workflow, adaptive, hybrid"
        )
    return value  # type: ignore[return-value]


def profile_to_mode(profile: str) -> OrchestrationMode:
    if profile in (PROFILE_WORKFLOW,):
        return MODE_FIXED_GRAPH
    if profile in (PROFILE_ADAPTIVE, PROFILE_HYBRID):
        return MODE_SUPERVISOR
    raise ValueError(f"profile='{profile}' 无法直接映射为 mode，请使用 RouterOrchestrator")


def resolve_auto_profile_with_reason(candidates: List[AgentCandidate]) -> tuple[str, str]:
    """根据 classification 结果选择 workflow 或 adaptive，并返回可解释原因。"""
    strong = [
        c
        for c in candidates
        if c.score >= STRONG_CANDIDATE_THRESHOLD and c.name.lower() != "other"
    ]
    if len(strong) > 1:
        detail = ", ".join(f"{c.name}={c.score:.2f}" for c in strong)
        return (
            PROFILE_WORKFLOW,
            f"strong_agents={len(strong)}>={STRONG_CANDIDATE_THRESHOLD}: {detail}",
        )
    if len(strong) == 1:
        c = strong[0]
        return (
            PROFILE_ADAPTIVE,
            f"single_strong_agent={c.name} score={c.score:.2f}>={STRONG_CANDIDATE_THRESHOLD}",
        )
    if candidates:
        top = max(candidates, key=lambda c: c.score)
        return (
            PROFILE_ADAPTIVE,
            f"no_strong_agents threshold={STRONG_CANDIDATE_THRESHOLD}; "
            f"top={top.name} score={top.score:.2f}",
        )
    return PROFILE_ADAPTIVE, "no_candidates"


def resolve_profile_with_reason(
    candidates: List[AgentCandidate],
    force_profile: Optional[str] = None,
) -> tuple[str, str]:
    if force_profile:
        fp = force_profile.strip()
        return fp, f"forced_profile={fp}"
    return resolve_auto_profile_with_reason(candidates)


def resolve_auto_profile(candidates: List[AgentCandidate]) -> str:
    """根据 classification 结果选择 workflow（多 Agent 流水线）或 adaptive（Supervisor）。"""
    return resolve_auto_profile_with_reason(candidates)[0]
