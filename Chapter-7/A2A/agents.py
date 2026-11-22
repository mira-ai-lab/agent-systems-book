"""Business agents exposed by the Chapter-7 A2A services.

``TravelPlannerAgent`` reuses the Chapter-6 task planning pipeline, while
``HotelRecommendationAgent`` reuses the Chapter-6 domain agent. Both adapt the
existing Chapter-6 implementations to the stream interface consumed by the
generic A2A executor in ``server.py``.

Main capabilities:

- Adapt the Chapter-6 task planner into a stream-based A2A agent.
- Adapt the Chapter-6 hotel agent into a stream-based A2A agent.

:author: zhanghong, liuhui
:date: 2026-11-22
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from a2a_models import (
    AgentCapability,
    PlannerRequest,
    TravelPlan,
)


A2A_DIR = Path(__file__).resolve().parent
BOOK_ROOT = A2A_DIR.parent.parent
for module_dir in (A2A_DIR, BOOK_ROOT / "Chapter-5", BOOK_ROOT / "Chapter-6"):
    if module_dir.is_dir() and str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

from config import load_project_dotenv
from domain_agents import get_domain_agent
from llm import create_llm
from task_plan import create_task_plan, survey_facts


load_project_dotenv()


@dataclass(frozen=True, slots=True)
class AgentStreamItem:
    """One text fragment emitted by a business agent."""

    content: str
    is_error: bool = False


def _message_text(message: Any) -> str:
    """Extract text from a complete LangChain message."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(block.get("text") or "")
            if isinstance(block, dict)
            else str(getattr(block, "text", block))
            for block in content
        )
    return str(content or "")


class LangGraphAgentRunner:
    """Run a LangGraph agent and return its final response."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent

    async def stream(
        self,
        query: str,
        context_id: str,
    ) -> AsyncIterable[AgentStreamItem]:
        try:
            state = await self.agent.ainvoke(
                {"messages": [("user", query)]},
                {"configurable": {"thread_id": context_id}},
            )
            messages = state.get("messages") if isinstance(state, dict) else None
            text = _message_text(messages[-1]) if messages else ""
            yield AgentStreamItem(
                text or "Agent completed without response text.",
                is_error=not bool(text),
            )
        except Exception as exc:
            yield AgentStreamItem(
                f"Agent execution failed: {type(exc).__name__}: {exc}",
                is_error=True,
            )


class CapabilityRegistry:
    """Adapt discovered A2A capabilities to the Chapter-6 planner interface."""

    def __init__(self, capabilities: List[AgentCapability]) -> None:
        self.agents: Dict[str, AgentCapability] = {
            capability.name: capability for capability in capabilities
        }

    def get_all_agents_text(self) -> str:
        return "\n".join(
            f"- {agent.name}: {agent.description}"
            for agent in self.agents.values()
        )

    def get_capability_catalog_text(self) -> str:
        return "\n".join(
            f"- {agent.description}" for agent in self.agents.values()
        )

    def get_agent_parameters_text(self) -> str:
        lines: List[str] = []
        for agent in self.agents.values():
            lines.append(f"{agent.name}: {agent.description}")
            for skill in agent.skills:
                lines.append(
                    f"\t{skill.name}, "
                    f"inputSchema:{skill.input_fields}, "
                    f"outputSchema:{skill.output_fields}"
                )
        return "\n".join(lines)

    def get_agent_input_fields(self, agent_name: str) -> set[str]:
        agent = self.agents.get(agent_name)
        if agent is None:
            return set()
        return {
            field_name
            for skill in agent.skills
            for field_name in skill.input_fields
        }

    def is_parallel_safe(self, agent_name: str) -> bool:
        agent = self.agents.get(agent_name)
        return bool(agent and agent.parallel_safe)


class TravelPlannerAgent:
    """Expose Chapter-6 task decomposition and dependency analysis via A2A."""

    def __init__(self) -> None:
        self._llm: Any | None = None

    def _get_llm(self) -> Any:
        if self._llm is None:
            self._llm = create_llm()
        return self._llm

    async def stream(
        self,
        query: str,
        context_id: str,
    ) -> AsyncIterable[AgentStreamItem]:
        request = PlannerRequest.model_validate_json(query)
        llm = self._get_llm()
        try:
            pre_survey = await survey_facts(llm, request.query)
        except Exception:
            pre_survey = {
                "given_facts": [request.query.strip()],
                "facts_to_lookup": [],
                "facts_to_derive": [],
                "educated_guesses": [],
                "raw_text": request.query.strip(),
            }

        plan = await create_task_plan(
            llm,
            request.query,
            pre_survey,
            [],
            CapabilityRegistry(request.available_agents),
        )
        yield AgentStreamItem(
            TravelPlan.model_validate(plan).model_dump_json()
        )


class HotelRecommendationAgent:
    """Expose Chapter-6 ``HotelAgent`` through the shared graph runner."""

    def __init__(self) -> None:
        self._runner: LangGraphAgentRunner | None = None

    async def stream(
        self,
        query: str,
        context_id: str,
    ) -> AsyncIterable[AgentStreamItem]:
        if self._runner is None:
            self._runner = LangGraphAgentRunner(
                get_domain_agent("HotelAgent")
            )
        async for item in self._runner.stream(query, context_id):
            yield item
