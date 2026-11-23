"""Router Engine 中间表示：RoutingPlan。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class AgentCandidate:
    name: str
    score: float


@dataclass(frozen=True)
class RoutingStep:
    step_id: str
    description: str
    agent: Optional[str] = None
    depends_on: tuple[str, ...] = ()


@dataclass
class RoutingPlan:
    """L1 路由输出，供执行 Profile / 编排层消费。"""

    rewritten_query: str
    candidates: List[AgentCandidate] = field(default_factory=list)
    events: List[str] = field(default_factory=list)
    steps: List[RoutingStep] = field(default_factory=list)
    profile: str = "adaptive"
    transport: str = "local"
    locale: str = "zh"
    history_relevant: Optional[bool] = None
    primary_agent: Optional[str] = None
    agent_instruction: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def execution_query(self) -> str:
        return (self.agent_instruction or self.rewritten_query).strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rewritten_query": self.rewritten_query,
            "candidates": [{"name": c.name, "score": c.score} for c in self.candidates],
            "events": list(self.events),
            "steps": [
                {
                    "step_id": s.step_id,
                    "description": s.description,
                    "agent": s.agent,
                    "depends_on": list(s.depends_on),
                }
                for s in self.steps
            ],
            "profile": self.profile,
            "transport": self.transport,
            "locale": self.locale,
            "history_relevant": self.history_relevant,
            "primary_agent": self.primary_agent,
            "agent_instruction": self.agent_instruction,
            "execution_query": self.execution_query,
            "metadata": dict(self.metadata),
        }
