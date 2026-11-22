"""Task planner reusing Chapter-6's task_plan (survey_facts + create_task_plan).

Planning stops at dependency-ordered task definitions and deliberately does not
select domain agents: Chapter-6 keeps routing as a separate stage, and Chapter-8
mirrors that with a dedicated routing node.

:author: zhanghong, liuhui, xingrui
:date: 2026-11-22
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_openai import ChatOpenAI

from travel_multi_agent.config import BOOK_ROOT  # noqa: F401 (registers Chapter-6)
from travel_multi_agent.tracing.trace_provider import span_name, trace_span

from task_plan import create_task_plan, survey_facts


class TaskPlanner:
    """Plan a request into dependency-ordered tasks (no agent assignment)."""

    def __init__(self, llm: ChatOpenAI, agent_registry: Any):
        self.llm = llm
        self.agent_registry = agent_registry

    @trace_span(name=span_name("planner.pre_survey"), attrs_args=["user_query"])
    async def run_pre_survey(self, user_query: str) -> Dict[str, Any]:
        """Run the Chapter-2 fact survey via Chapter-6's survey_facts."""
        try:
            return await survey_facts(self.llm, user_query)
        except Exception:
            return {
                "given_facts": [user_query.strip()],
                "facts_to_lookup": [],
                "facts_to_derive": [],
                "educated_guesses": [],
                "raw_text": user_query.strip(),
            }

    @trace_span(
        name=span_name("planner.build_plan"),
        attrs_args=["user_query"],
        record_result=False,
    )
    async def build_execution_plan(
        self,
        user_query: str,
        pre_survey: Dict[str, Any],
        memories: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build the dependency-safe task plan (decomposition only, no routing)."""
        return await create_task_plan(
            self.llm,
            user_query,
            pre_survey,
            memories,
            self.agent_registry,
        )