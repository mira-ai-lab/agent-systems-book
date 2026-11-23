"""Build travel orchestrators for end-to-end benchmark evaluation."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Optional, Protocol

from langchain_openai import ChatOpenAI

from agent_framework.domain.pipeline import PRE_SURVEY_MODE_FULL_CH2, PipelineConfig
from agent_framework.domain.plugin_registry import get_domain_plugin
from agent_framework.optimization.planner_runtime import build_travel_prompts
from agent_framework.orchestration.fixed_graph.orchestrator import LangGraphOrchestrator
from agent_framework.orchestration.router_orchestrator import RouterOrchestrator
from agent_framework.router.profile import PROFILE_WORKFLOW


class E2eOrchestrator(Protocol):
    async def process_request(
        self,
        user_query: str,
        thread_id: str = "default",
        timeout_sec: Optional[float] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        ...


def build_e2e_orchestrator(
    llm: ChatOpenAI,
    *,
    locale: str = "zh",
    profile: str = "workflow",
    enable_memory: bool = False,
    enable_guess_agent: bool = True,
    prompt_overrides: Optional[Mapping[str, str]] = None,
    use_optimized: bool = True,
) -> E2eOrchestrator:
    """Create a travel orchestrator for E2E evaluation.

    profile:
      - ``workflow``: RouterOrchestrator + Fixed Graph（产品路径）
      - ``legacy``: LangGraphOrchestrator，图内完整 TaskPlanner
    """
    normalized_profile = (profile or "workflow").strip().lower()
    if normalized_profile not in ("workflow", "legacy"):
        raise ValueError("profile 仅支持 workflow 或 legacy")

    def _create_eval_prompts(*, locale: str = "zh"):
        prompts = build_travel_prompts(
            locale=locale,
            prompt_overrides=prompt_overrides,
            use_optimized=use_optimized,
        )
        return prompts.with_platform_defaults(locale)

    if normalized_profile == "legacy":
        from domains.travel.registry import create_travel_registry, travel_domain_config

        pipeline = PipelineConfig(
            enable_pre_survey=True,
            pre_survey_mode=PRE_SURVEY_MODE_FULL_CH2,
            enable_memory=enable_memory,
            allow_task_planner_decomposition=True,
        )
        return LangGraphOrchestrator(
            llm=llm,
            domain="travel",
            registry=create_travel_registry(),
            prompts=_create_eval_prompts(locale=locale),
            domain_config=travel_domain_config(enable_guess_agent=enable_guess_agent),
            pipeline=pipeline,
            enable_memory=enable_memory,
            enable_guess_agent=enable_guess_agent,
            locale=locale,
        )

    plugin = get_domain_plugin("travel")
    eval_plugin = replace(plugin, create_prompts=_create_eval_prompts)
    return RouterOrchestrator(
        llm,
        eval_plugin,
        domain="travel",
        enable_memory=enable_memory,
        enable_guess_agent=enable_guess_agent,
        locale=locale,
        entry_profile=PROFILE_WORKFLOW,
    )
