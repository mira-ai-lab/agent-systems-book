"""领域配置：context_builder、guess、路由策略。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

ContextBuilder = Callable[[], str]
SubtaskSummaryFn = Callable[[Dict[str, Any]], Optional[str]]


def empty_context_builder() -> str:
    return ""


@dataclass
class DomainConfig:
    context_builder: ContextBuilder = empty_context_builder
    guess_fn: Optional[Callable[[str, Any], Optional[str]]] = None
    routing_fallback: Optional[str] = None
    enable_guess_agent: bool = False
    subtask_summary_fn: Optional[SubtaskSummaryFn] = None

    def build_context_block(self) -> str:
        return (self.context_builder() or "").strip()

    def guess_agent(self, description: str, registry: Any) -> Optional[str]:
        if self.guess_fn is not None:
            return self.guess_fn(description, registry)
        if hasattr(registry, "guess_agent"):
            return registry.guess_agent(description)
        return None
