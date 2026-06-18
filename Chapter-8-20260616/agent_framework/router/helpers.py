"""路由阶段辅助：主候选 Agent 选择。"""

from __future__ import annotations

from typing import List, Optional

from agent_framework.router.plan import AgentCandidate


def select_primary_candidate(candidates: List[AgentCandidate]) -> Optional[AgentCandidate]:
    filtered = [c for c in candidates if c.name.lower() != "other"]
    if not filtered:
        return None
    return max(filtered, key=lambda c: c.score)


def agent_skill_text(registry_agents: dict, agent_name: str) -> str:
    info = registry_agents.get(agent_name, {})
    desc = str(info.get("description") or agent_name)
    skills = info.get("skills") or []
    if not skills:
        return desc
    parts = []
    for skill in skills:
        if isinstance(skill, dict):
            parts.append(str(skill.get("name") or skill.get("description") or skill))
        else:
            parts.append(str(skill))
    skill_text = "、".join(parts) if parts else "无"
    return f"{desc}\n能力列表:{skill_text}"
