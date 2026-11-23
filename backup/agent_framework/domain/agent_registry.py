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
        self._instances: Dict[tuple[str, str], Any] = {}
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
        info.setdefault("metadata_only", False)
        self.agents[name] = info
        self._creators[name] = creator

    def register_metadata(
        self,
        name: str,
        *,
        description: str = "",
        skills: Optional[List[Dict[str, Any]]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """仅登记元数据（供 Router classification / Supervisor 发现），无本地 creator。"""
        info = dict(meta or {})
        info.setdefault("name", name)
        info.setdefault("description", description)
        info.setdefault("requires_tool", False)
        info.setdefault("skills", skills or [])
        info["metadata_only"] = True
        self.agents[name] = info

    def unregister(self, name: str) -> bool:
        if name not in self.agents:
            return False
        self.agents.pop(name, None)
        self._creators.pop(name, None)
        self._instances.pop(name, None)
        return True

    def is_metadata_only(self, name: str) -> bool:
        info = self.agents.get(name)
        return bool(info and info.get("metadata_only"))

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
        """惰性创建并缓存子 Agent 实例（按 locale 分 cache）。"""
        from agent_framework.i18n.agent_locale_context import get_agent_locale

        if self.is_metadata_only(agent_name):
            raise ValueError(
                f"子智能体 '{agent_name}' 仅为元数据注册（动态/A2A），不可本地实例化"
            )
        locale = get_agent_locale()
        cache_key = (agent_name, locale)
        if cache_key not in self._instances:
            creator = self._creators.get(agent_name)
            if not creator:
                raise ValueError(f"未知的子智能体: {agent_name}")
            log_info(logger, "agent.create", agent=agent_name, locale=locale)
            self._instances[cache_key] = creator()
        return self._instances[cache_key]

    def get_agent_names(self) -> List[str]:
        return list(self.agents.keys())

    def list_agent_metadata(self) -> List[Dict[str, Any]]:
        return [dict(info) for info in self.agents.values()]

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

    def subset_excluding(self, exclude: set[str]) -> SubAgentRegistry:
        """复制注册表，排除指定 Agent 名（用于 mixed 模式下 A2A 替代本地）。"""
        out = SubAgentRegistry()
        for name in self.get_agent_names():
            if name in exclude:
                continue
            info = self.agents[name]
            if self.is_metadata_only(name):
                out.register_metadata(
                    name,
                    description=str(info.get("description", "")),
                    skills=info.get("skills"),
                    meta=dict(info),
                )
                continue
            out.register(
                name,
                self._creators[name],
                description=str(info.get("description", "")),
                requires_tool=bool(info.get("requires_tool", False)),
                skills=info.get("skills"),
                meta=dict(info),
            )
        out.register_guess_rules(self._guess_rules)
        return out
