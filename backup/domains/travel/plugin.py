"""旅行领域插件（注册到 agent_framework.domain.plugin_registry）。"""

from __future__ import annotations

from agent_framework.domain.a2a_spec import A2AEndpoint
from agent_framework.domain.domain_config import DomainConfig
from agent_framework.domain.domain_prompts import DomainPrompts
from agent_framework.domain.pipeline import PipelineConfig
from agent_framework.domain.plugin import DomainPlugin
from agent_framework.orchestration.protocol import MODE_FIXED_GRAPH, MODE_SUPERVISOR


def _create_registry():
    from domains.travel.registry import create_travel_registry

    return create_travel_registry()


def _create_prompts(*, locale: str = "zh") -> DomainPrompts:
    from domains.travel.prompt_bundle import TravelPrompts

    return TravelPrompts.build(locale=locale)


def _create_domain_config(*, enable_guess_agent: bool = False, **_: object) -> DomainConfig:
    from domains.travel.registry import travel_domain_config

    return travel_domain_config(enable_guess_agent=enable_guess_agent)


def _default_pipeline(*, enable_memory: bool = True) -> PipelineConfig:
    from agent_framework.domain.pipeline import PRE_SURVEY_MODE_FULL_CH2

    return PipelineConfig(
        enable_pre_survey=True,
        pre_survey_mode=PRE_SURVEY_MODE_FULL_CH2,
        enable_memory=enable_memory,
    )


def _create_travel_a2a_endpoints() -> tuple[A2AEndpoint, ...]:
    import os

    hotel_url = os.getenv("TRAVEL_A2A_HOTEL_URL", "").strip()
    if not hotel_url:
        return ()
    return (
        A2AEndpoint(
            node_name="hotel_agent",
            url=hotel_url,
            description="酒店推荐（A2A 远程）",
            registry_agent="HotelAgent",
        ),
    )


TRAVEL_PLUGIN = DomainPlugin(
    name="travel",
    display_name="旅行规划",
    create_registry=_create_registry,
    create_prompts=_create_prompts,
    create_domain_config=_create_domain_config,
    default_pipeline=_default_pipeline,
    supported_modes=(MODE_FIXED_GRAPH, MODE_SUPERVISOR),
    create_a2a_endpoints=_create_travel_a2a_endpoints,
)
