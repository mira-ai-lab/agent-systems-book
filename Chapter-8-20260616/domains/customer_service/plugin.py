"""客服领域插件。"""

from __future__ import annotations

from agent_framework.domain.domain_config import DomainConfig
from agent_framework.domain.domain_prompts import DomainPrompts
from agent_framework.domain.pipeline import PipelineConfig
from agent_framework.domain.plugin import DomainPlugin
from agent_framework.orchestration.protocol import MODE_FIXED_GRAPH, MODE_SUPERVISOR


def _create_registry():
    from domains.customer_service.registry import create_customer_service_registry

    return create_customer_service_registry()


def _create_prompts(*, locale: str = "zh") -> DomainPrompts:
    from domains.customer_service.prompt_bundle import CustomerServicePrompts

    return CustomerServicePrompts.build(locale=locale)


def _create_domain_config(*, enable_guess_agent: bool = False, **_: object) -> DomainConfig:
    from domains.customer_service.registry import customer_service_domain_config

    return customer_service_domain_config(enable_guess_agent=enable_guess_agent)


def _default_pipeline(*, enable_memory: bool = True) -> PipelineConfig:
    return PipelineConfig(enable_pre_survey=False, enable_memory=enable_memory)


CUSTOMER_SERVICE_PLUGIN = DomainPlugin(
    name="customer_service",
    display_name="智能客服",
    create_registry=_create_registry,
    create_prompts=_create_prompts,
    create_domain_config=_create_domain_config,
    default_pipeline=_default_pipeline,
    supported_modes=(MODE_FIXED_GRAPH, MODE_SUPERVISOR),
)
