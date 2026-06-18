"""Agent classification（name + score）。"""

from __future__ import annotations

import json
from typing import Any, List

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from agent_framework.domain.agent_registry import SubAgentRegistry
from agent_framework.domain.parsing import parse_json_from_llm
from agent_framework.router.plan import AgentCandidate
from agent_framework.router.prompts.loader import get_classification_prompts
from agent_framework.router.skills_format import format_agent_skills
from agent_framework.tracing.trace_provider import span_name, trace_span


def build_agent_catalog(registry: SubAgentRegistry, *, locale: str = "zh") -> str:
    prompts = get_classification_prompts(locale)
    template = prompts["agent_template"]
    blocks: List[str] = []
    for name in registry.get_agent_names():
        info = registry.agents.get(name, {})
        blocks.append(
            template.format(
                name=name,
                description=str(info.get("description") or name),
                skills=format_agent_skills(info, locale=locale),
            )
        )
    return "\n".join(blocks)


def parse_classification_response(
    raw: Any,
    registry: SubAgentRegistry,
) -> List[AgentCandidate]:
    if isinstance(raw, dict):
        if "name" in raw and "score" in raw:
            raw = [raw]
        else:
            raw = []
    if not isinstance(raw, list):
        return []

    known = set(registry.get_agent_names())
    candidates: List[AgentCandidate] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        try:
            score = float(item.get("score", 0))
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(1.0, score))
        if name != "other" and name not in known:
            continue
        candidates.append(AgentCandidate(name=name, score=score))

    if not candidates:
        fallback = registry.get_agent_names()
        if fallback:
            candidates.append(AgentCandidate(name=fallback[0], score=0.5))
        else:
            candidates.append(AgentCandidate(name="other", score=1.0))
    return candidates


@trace_span(
    name=span_name("router.classification"),
    attrs_args=["query"],
    record_result=False,
)
async def run_classification(
    llm: ChatOpenAI,
    registry: SubAgentRegistry,
    query: str,
    *,
    locale: str = "zh",
    note: str = "",
) -> List[AgentCandidate]:
    prompts = get_classification_prompts(locale)
    agent_names = ", ".join(registry.get_agent_names()) or "other"
    catalog = build_agent_catalog(registry, locale=locale)
    prompt = prompts["prompt_base"].format(
        agent_names=agent_names,
        note=note or prompts.get("note", ""),
        agent_catalog=catalog,
        query=query.strip(),
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    text = str(response.content or "").strip()
    try:
        parsed = parse_json_from_llm(text)
    except (ValueError, json.JSONDecodeError):
        parsed = []
    return parse_classification_response(parsed, registry)
