"""Runtime helpers for travel TaskPlanner benchmark / optimization."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Mapping, Optional

from langchain_openai import ChatOpenAI

from agent_framework.domain.task_planner import TaskPlanner
from agent_framework.optimization.prompt_store import apply_prompt_overrides, load_optimized_prompts


def _travel_prompts_from_locale(locale: str = "zh"):
    from dataclasses import fields

    from agent_framework.domain.locale_loader import domain_prompts_from_locale
    from domains.travel.prompt_bundle import TravelPrompts

    base = domain_prompts_from_locale("travel", locale)
    return TravelPrompts(**{field.name: getattr(base, field.name) for field in fields(base)})


def build_travel_prompts(
    *,
    locale: str = "zh",
    prompt_overrides: Optional[Mapping[str, str]] = None,
    use_optimized: bool = True,
) -> "DomainPrompts":
    from agent_framework.domain.domain_prompts import DomainPrompts

    prompts: DomainPrompts = _travel_prompts_from_locale(locale)
    if use_optimized:
        prompts = apply_prompt_overrides(prompts, locale=locale)
    if prompt_overrides:
        valid = {key: str(value) for key, value in prompt_overrides.items() if str(value).strip()}
        if valid:
            prompts = replace(prompts, **valid)
    return prompts


def build_planner(
    llm: ChatOpenAI,
    registry: Any,
    *,
    locale: str = "zh",
    prompt_overrides: Optional[Mapping[str, str]] = None,
    use_optimized: bool = False,
) -> TaskPlanner:
    prompts = build_travel_prompts(
        locale=locale,
        prompt_overrides=prompt_overrides,
        use_optimized=use_optimized,
    )
    return TaskPlanner(llm, registry, prompts)


def build_decomposition_planner(
    decomposition_prompt: str,
    llm: ChatOpenAI,
    registry: Any,
    *,
    locale: str = "zh",
    agent_routing: Optional[str] = None,
) -> TaskPlanner:
    overrides: Dict[str, str] = {"decomposition_prompt": decomposition_prompt}
    if agent_routing is not None:
        overrides["agent_routing"] = agent_routing
    return build_planner(llm, registry, locale=locale, prompt_overrides=overrides, use_optimized=False)


def build_routing_planner(
    agent_routing: str,
    llm: ChatOpenAI,
    registry: Any,
    *,
    locale: str = "zh",
    decomposition_prompt: Optional[str] = None,
) -> TaskPlanner:
    overrides: Dict[str, str] = {"agent_routing": agent_routing}
    if decomposition_prompt is not None:
        overrides["decomposition_prompt"] = decomposition_prompt
    return build_planner(llm, registry, locale=locale, prompt_overrides=overrides, use_optimized=False)


def merge_with_saved_prompts(
    *,
    locale: str,
    decomposition_prompt: Optional[str] = None,
    agent_routing: Optional[str] = None,
) -> Dict[str, str]:
    merged = dict(load_optimized_prompts(locale))
    if decomposition_prompt:
        merged["decomposition_prompt"] = decomposition_prompt
    if agent_routing:
        merged["agent_routing"] = agent_routing
    return merged
