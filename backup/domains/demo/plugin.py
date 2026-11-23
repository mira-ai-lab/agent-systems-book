"""最小领域插件模板（Plugin 开发文档参考实现）。"""

from __future__ import annotations

from agent_framework.domain.agent_registry import SubAgentRegistry
from agent_framework.domain.domain_config import DomainConfig
from agent_framework.domain.pipeline import PipelineConfig
from agent_framework.domain.plugin import DomainPlugin
from domains.demo.prompt_bundle import DemoPrompts
from agent_framework.infra.agent_runtime import build_agent
from agent_framework.orchestration.protocol import MODE_FIXED_GRAPH, MODE_SUPERVISOR


def _create_registry() -> SubAgentRegistry:
    registry = SubAgentRegistry()
    registry.register(
        "EchoAgent",
        lambda: build_agent([], "你是 Echo 示例 Agent，复述用户要点并简短回答。"),
        description="示例：复述用户请求（无外部工具）",
        requires_tool=False,
    )
    return registry


def _create_prompts(*, locale: str = "zh") -> DemoPrompts:
    return DemoPrompts.build(locale=locale)


def _create_domain_config(**_: object) -> DomainConfig:
    return DomainConfig(routing_fallback="EchoAgent")


def _default_pipeline(*, enable_memory: bool = True) -> PipelineConfig:
    return PipelineConfig(enable_pre_survey=False, enable_memory=enable_memory)


DEMO_PLUGIN = DomainPlugin(
    name="demo",
    display_name="最小插件模板",
    create_registry=_create_registry,
    create_prompts=_create_prompts,
    create_domain_config=_create_domain_config,
    default_pipeline=_default_pipeline,
    supported_modes=(MODE_FIXED_GRAPH, MODE_SUPERVISOR),
)
