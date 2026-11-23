"""领域插件协议：registry / prompts / domain_config 工厂。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from agent_framework.domain.a2a_spec import A2AEndpoint
from agent_framework.domain.agent_registry import SubAgentRegistry
from agent_framework.domain.domain_config import DomainConfig
from agent_framework.domain.domain_prompts import DomainPrompts
from agent_framework.domain.pipeline import PipelineConfig
from agent_framework.orchestration.protocol import MODE_FIXED_GRAPH, OrchestrationMode

RegistryFactory = Callable[[], SubAgentRegistry]
PromptsFactory = Callable[..., DomainPrompts]
DomainConfigFactory = Callable[..., DomainConfig]
PipelineFactory = Callable[..., PipelineConfig]


@dataclass(frozen=True)
class DomainPlugin:
    """可注册到平台的一个业务领域（旅行、客服、金融等）。"""

    name: str
    create_registry: RegistryFactory
    create_prompts: PromptsFactory
    create_domain_config: DomainConfigFactory
    display_name: str = ""
    is_sample: bool = False
    default_pipeline: Optional[PipelineFactory] = None
    supported_modes: Tuple[OrchestrationMode, ...] = (MODE_FIXED_GRAPH,)
    a2a_endpoints: Tuple[A2AEndpoint, ...] = ()
    create_a2a_endpoints: Optional[Callable[[], Tuple[A2AEndpoint, ...]]] = None

    def build_pipeline(self, *, enable_memory: bool = True) -> PipelineConfig:
        if self.default_pipeline is not None:
            return self.default_pipeline(enable_memory=enable_memory)
        return PipelineConfig(enable_memory=enable_memory)

    def supports_mode(self, mode: str) -> bool:
        return mode in self.supported_modes

    def resolved_a2a_endpoints(self) -> Tuple[A2AEndpoint, ...]:
        if self.create_a2a_endpoints is not None:
            return self.create_a2a_endpoints()
        return self.a2a_endpoints
