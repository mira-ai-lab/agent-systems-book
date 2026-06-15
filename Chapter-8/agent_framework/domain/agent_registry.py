"""子智能体注册表：元数据 + 工厂 creator 合一，供 TaskPlanner 与编排层使用。"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from agent_framework.tracing import get_logger, log_info

logger = get_logger(__name__)

AgentCreator = Callable[[], Any]
GuessRule = Tuple[Sequence[str], str]


class SubAgentRegistry:
    """子智能体注册表：register() 同时登记元数据与 creator。"""

    def __init__(self) -> None:
        self.agents: Dict[str, Dict[str, Any]] = {}
        self._creators: Dict[str, AgentCreator] = {}
        self._instances: Dict[str, Any] = {}
        self._guess_rules: List[GuessRule] = []

    def register(
        self,
        name: str,
        creator: AgentCreator,
        *,
        description: str = "",
        requires_tool: bool = False,
        skills: Optional[List[Dict[str, Any]]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """注册子 Agent：creator 用于惰性实例化，meta/skills 用于 Planner prompt。"""
        info = dict(meta or {})
        info.setdefault("name", name)
        info.setdefault("description", description)
        info.setdefault("requires_tool", requires_tool)
        info.setdefault("skills", skills or [])
        self.agents[name] = info
        self._creators[name] = creator

    def register_guess_rules(self, rules: Sequence[GuessRule]) -> None:
        """注册启发式路由规则：(关键词元组, agent_name)。"""
        self._guess_rules.extend(list(rules))

    def guess_agent(self, description: str) -> Optional[str]:
        desc = description.lower()
        for keywords, agent_name in self._guess_rules:
            if any(k in desc for k in keywords):
                if self.has_agent(agent_name):
                    return agent_name
        return None

    def has_agent(self, name: Optional[str]) -> bool:
        return bool(name and name in self.agents)

    def resolve_agent(self, name: Optional[str]) -> Optional[str]:
        """校验 agent 名称是否已注册；无效时返回 None。"""
        if self.has_agent(name):
            return name
        return None

    def get_agent(self, agent_name: str) -> Any:
        """惰性创建并缓存子 Agent 实例。"""
        if agent_name not in self._instances:
            creator = self._creators.get(agent_name)
            if not creator:
                raise ValueError(f"未知的子智能体: {agent_name}")
            log_info(logger, "agent.create", agent=agent_name)
            self._instances[agent_name] = creator()
        return self._instances[agent_name]

    def get_agent_names(self) -> List[str]:
        return list(self.agents.keys())

    def get_all_agents_text(self) -> str:
        return "\n".join(f"- {a['name']}: {a['description']}" for a in self.agents.values())

    def get_agent_parameters_text(self) -> str:
        lines = []
        for info in self.agents.values():
            lines.append(info["name"])
            for skill in info.get("skills", []):
                lines.append(
                    f"\t{skill['name']}, inputSchema:{skill['inputSchema']}, "
                    f"outputSchema:{skill['outputSchema']}"
                )
        return "\n".join(lines)

    def requires_tool(self, agent_name: str) -> bool:
        info = self.agents.get(agent_name)
        return bool(info and info.get("requires_tool", False))


def create_travel_registry() -> SubAgentRegistry:
    """构建旅行领域默认子 Agent 注册表（委托 domains.travel）。"""
    from domains.travel.registry import create_travel_registry as _create

    return _create()
