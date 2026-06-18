"""子 Agent 工厂：委托 SubAgentRegistry 惰性创建实例。"""

from __future__ import annotations

from typing import Any, List, Optional

from agent_framework.domain.agent_registry import SubAgentRegistry


class SubAgentFactory:
    """兼容层；编排器通过 use_registry() 注入领域 registry。"""

    _registry: Optional[SubAgentRegistry] = None

    @classmethod
    def use_registry(cls, registry: SubAgentRegistry) -> None:
        cls._registry = registry

    @classmethod
    def _resolve_registry(cls) -> SubAgentRegistry:
        if cls._registry is None:
            raise RuntimeError(
                "SubAgentFactory 未注入 registry；"
                "编排器初始化时应调用 SubAgentFactory.use_registry()"
            )
        return cls._registry

    @classmethod
    def get_agent(cls, agent_name: str) -> Any:
        return cls._resolve_registry().get_agent(agent_name)

    @classmethod
    def get_all_agent_names(cls) -> List[str]:
        return cls._resolve_registry().get_agent_names()
