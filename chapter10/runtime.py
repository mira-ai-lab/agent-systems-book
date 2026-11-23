"""Decompose / dependency / route primitives with prompt injection.

These mirror the refactored Chapter-8 planner, but expose the two prompt
templates as injectable parameters so the optimizer can evaluate candidate
prompts without touching Chapter-4 / Chapter-6.

The pure rendering and routing-normalization helpers are shared by the async
functions below and by the synchronous bridge in ``planner_graph.py``, so
prompt formatting has a single implementation.

:author: zhanghong, liuhui
:date: 2026-11-23
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from llm import invoke_chat, invoke_json

_BOOK_ROOT = Path(__file__).resolve().parent.parent
_CHAPTER4_DIR = _BOOK_ROOT / "Chapter-4"
if str(_CHAPTER4_DIR) not in sys.path:
    sys.path.insert(0, str(_CHAPTER4_DIR))

from task_decomposition import parse_task_decomposition_response
from dependency_analysis import analyze_task_dependencies

from optimizable_prompts import AGENT_ROUTING_PROMPT, TASK_DECOMPOSITION_PROMPT


def build_task_id_map(sub_steps: List[str]) -> Dict[str, str]:
    """Map ``T1, T2, ...`` to the corresponding subtask description."""
    return {f"T{i + 1}": step for i, step in enumerate(sub_steps)}


def render_decomposition_prompt(
    template: str,
    background_info: str,
    capability_catalog: str,
    user_input: str,
) -> str:
    """Fill the decomposition template with its three slots."""
    return template.format(
        background_info=(background_info or "").strip(),
        capability_catalog=capability_catalog.strip(),
        user_input=user_input.strip(),
    )


def build_routing_subtasks(
    execution_order: List[str],
    sub_steps: List[str],
    depends_map: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """Build the subtask list rendered into the routing prompt."""
    id_to_desc = dict(zip(execution_order, sub_steps))
    return [
        {
            "task_id": task_id,
            "description": id_to_desc.get(task_id, ""),
            "depends_on": depends_map.get(task_id, []),
        }
        for task_id in execution_order
    ]


def render_routing_prompt(
    template: str,
    agent_team: str,
    pre_survey: Dict[str, Any],
    subtasks: List[Dict[str, Any]],
    depends_map: Dict[str, List[str]],
) -> str:
    """Fill the routing template with its four slots."""
    return template.format(
        agent_team=agent_team,
        pre_survey_json=json.dumps(
            {key: value for key, value in pre_survey.items() if key != "raw_text"},
            ensure_ascii=False,
            indent=2,
        ),
        subtasks_json=json.dumps(subtasks, ensure_ascii=False, indent=2),
        depends_on_json=json.dumps(depends_map, ensure_ascii=False, indent=2),
    )


def normalize_routing_output(
    routed: Any,
    execution_order: List[str],
    valid_agents: set,
    allowed_fields: Dict[str, set],
) -> List[Dict[str, Any]]:
    """Validate the model's routing list and drop unsupported params."""
    if not isinstance(routed, list):
        routed = routed.get("subtasks", []) if isinstance(routed, dict) else []
    by_id = {
        str(item.get("task_id")): item
        for item in routed
        if isinstance(item, dict) and item.get("task_id")
    }

    records: List[Dict[str, Any]] = []
    for task_id in execution_order:
        item = by_id.get(task_id)
        agent = str(item.get("agent") or "").strip() if item else ""
        params = (
            dict(item.get("params"))
            if item and isinstance(item.get("params"), dict)
            else {}
        )
        if agent in valid_agents:
            params = {
                key: value
                for key, value in params.items()
                if key in allowed_fields.get(agent, set())
            }
        else:
            agent = ""
            params = {}
        records.append({"task_id": task_id, "agent": agent, "params": params})
    return records


async def decompose(
    llm: ChatOpenAI,
    query: str,
    capability_catalog: str,
    *,
    background_info: str = "",
    prompt_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Decompose one query using Chapter-4's template (or an injected candidate)."""
    template = prompt_override or TASK_DECOMPOSITION_PROMPT
    prompt = render_decomposition_prompt(
        template, background_info, capability_catalog, query
    )
    response = await invoke_chat(llm, prompt)
    return parse_task_decomposition_response(response)


async def dependency_analysis(
    llm: ChatOpenAI,
    sub_steps: List[str],
    agent_parameters_text: str,
) -> tuple[List[str], Dict[str, List[str]]]:
    """Return a dependency-safe execution order and depends_on map."""
    id_to_task = build_task_id_map(sub_steps)
    if len(id_to_task) <= 1:
        return list(id_to_task), {task_id: [] for task_id in id_to_task}
    return await analyze_task_dependencies(llm, id_to_task, agent_parameters_text)


async def route_detailed(
    llm: ChatOpenAI,
    registry: Any,
    sub_steps: List[str],
    execution_order: List[str],
    depends_map: Dict[str, List[str]],
    pre_survey: Dict[str, Any],
    *,
    prompt_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Route subtasks to agents; returns per-task records with agent and params."""
    template = prompt_override or AGENT_ROUTING_PROMPT
    agent_team = (
        registry.get_all_agents_text() + "\n" + registry.get_agent_parameters_text()
    )
    subtasks_for_prompt = build_routing_subtasks(
        execution_order, sub_steps, depends_map
    )
    prompt = render_routing_prompt(
        template, agent_team, pre_survey, subtasks_for_prompt, depends_map
    )
    routed = await invoke_json(llm, prompt)
    valid_agents = set(registry.get_agent_names())
    allowed_fields = {
        name: registry.get_agent_input_fields(name) for name in valid_agents
    }
    return normalize_routing_output(
        routed, execution_order, valid_agents, allowed_fields
    )


class Planner:
    """Minimal planner wrapper binding an LLM, a registry, and two prompt slots."""

    def __init__(
        self,
        llm: ChatOpenAI,
        registry: Any,
        *,
        decomposition_prompt: Optional[str] = None,
        agent_routing: Optional[str] = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.decomposition_prompt = decomposition_prompt or TASK_DECOMPOSITION_PROMPT
        self.agent_routing = agent_routing or AGENT_ROUTING_PROMPT

    async def decompose(self, query: str, *, background_info: str = "") -> Dict[str, Any]:
        return await decompose(
            self.llm,
            query,
            self.registry.get_capability_catalog_text(),
            background_info=background_info,
            prompt_override=self.decomposition_prompt,
        )

    async def dependency_analysis(self, sub_steps: List[str]):
        return await dependency_analysis(
            self.llm,
            sub_steps,
            self.registry.get_agent_parameters_text(),
        )

    async def route_detailed(
        self,
        sub_steps: List[str],
        execution_order: List[str],
        depends_map: Dict[str, List[str]],
        pre_survey: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        return await route_detailed(
            self.llm,
            self.registry,
            sub_steps,
            execution_order,
            depends_map,
            pre_survey,
            prompt_override=self.agent_routing,
        )
