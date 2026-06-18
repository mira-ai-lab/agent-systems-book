"""运行时动态 Agent 注册（叠加在领域静态 Registry 之上）。"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agent_framework.domain.a2a_spec import A2AEndpoint
from agent_framework.domain.agent_registry import SubAgentRegistry
from agent_framework.orchestration.supervisor.agent_names import registry_agent_to_node_name

SHARED_DOMAIN = "__shared__"


@dataclass
class DynamicAgentRecord:
    name: str
    description: str = ""
    skills: List[Dict[str, Any]] = field(default_factory=list)
    source: str = "metadata"  # metadata | a2a
    a2a_url: str = ""
    a2a_node_name: str = ""
    registry_agent: Optional[str] = None
    alias_of: Optional[str] = None
    scope: str = "domain"  # domain | shared

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "skills": list(self.skills),
            "source": self.source,
            "a2a_url": self.a2a_url,
            "a2a_node_name": self.a2a_node_name,
            "registry_agent": self.registry_agent,
            "alias_of": self.alias_of,
            "scope": self.scope,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DynamicAgentRecord":
        name = str(data.get("name") or "").strip()
        if not name:
            raise ValueError("动态 Agent 记录缺少 name")
        source = str(data.get("source") or "metadata").strip().lower()
        if source not in ("metadata", "a2a"):
            source = "metadata"
        scope = str(data.get("scope") or "domain").strip().lower()
        if scope not in ("domain", "shared"):
            scope = "domain"
        return cls(
            name=name,
            description=str(data.get("description") or ""),
            skills=list(data.get("skills") or []),
            source=source,
            a2a_url=str(data.get("a2a_url") or ""),
            a2a_node_name=str(data.get("a2a_node_name") or ""),
            registry_agent=(str(data.get("registry_agent") or "").strip() or None),
            alias_of=(str(data.get("alias_of") or "").strip() or None),
            scope=scope,
        )


class DynamicAgentStore:
    """进程内 per-domain 动态 Agent 覆盖层。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: Dict[str, Dict[str, DynamicAgentRecord]] = {}

    def list_agents(self, domain: str) -> List[DynamicAgentRecord]:
        dom = domain.strip()
        with self._lock:
            return list(self._records.get(dom, {}).values())

    def get(self, domain: str, name: str) -> Optional[DynamicAgentRecord]:
        dom = domain.strip()
        with self._lock:
            return self._records.get(dom, {}).get(name.strip())

    def register(self, domain: str, record: DynamicAgentRecord) -> DynamicAgentRecord:
        dom = domain.strip()
        name = record.name.strip()
        if not name:
            raise ValueError("agent name 不能为空")
        scope = (record.scope or "domain").strip().lower()
        if scope == "shared":
            dom = SHARED_DOMAIN
        elif not dom:
            raise ValueError("domain 不能为空")
        if record.source == "a2a" and not (record.a2a_url or "").strip():
            raise ValueError("source='a2a' 时必须提供 a2a_url")
        normalized = DynamicAgentRecord(
            name=name,
            description=record.description.strip(),
            skills=list(record.skills or []),
            source=record.source,
            a2a_url=record.a2a_url.strip(),
            a2a_node_name=(record.a2a_node_name or registry_agent_to_node_name(name)).strip(),
            registry_agent=(record.registry_agent or "").strip() or None,
            alias_of=(record.alias_of or "").strip() or None,
            scope=scope,
        )
        with self._lock:
            bucket = self._records.setdefault(dom, {})
            bucket[name] = normalized
        return normalized

    def unregister(self, domain: str, name: str) -> bool:
        dom = domain.strip()
        agent = name.strip()
        with self._lock:
            bucket = self._records.get(dom)
            if not bucket or agent not in bucket:
                return False
            del bucket[agent]
            if not bucket:
                self._records.pop(dom, None)
            return True

    def clear_domain(self, domain: str) -> None:
        dom = domain.strip()
        with self._lock:
            self._records.pop(dom, None)


_store: DynamicAgentStore | None = None
_store_persist_enabled: bool | None = None


def get_dynamic_agent_store() -> DynamicAgentStore:
    global _store, _store_persist_enabled
    if _store is None:
        from agent_framework.domain.dynamic_registry_persist import (
            PersistedDynamicAgentStore,
            should_persist_dynamic_agents,
        )

        _store_persist_enabled = should_persist_dynamic_agents()
        if _store_persist_enabled:
            _store = PersistedDynamicAgentStore()
        else:
            _store = DynamicAgentStore()
    return _store


def reset_dynamic_agent_store() -> None:
    """测试用：重置为内存 Store（不读写 JSON）。"""
    global _store, _store_persist_enabled
    _store = DynamicAgentStore()
    _store_persist_enabled = False


def _merge_skill_lists(
    base_skills: Optional[List[Dict[str, Any]]],
    extra_skills: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = list(base_skills or [])
    seen = {str(item.get("name") or idx) for idx, item in enumerate(merged) if isinstance(item, dict)}
    for item in extra_skills or []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("name") or len(merged))
        if key in seen:
            continue
        merged.append(dict(item))
        seen.add(key)
    return merged


def resolve_static_agent_reference(
    domain: str,
    base_registry: SubAgentRegistry,
    record: DynamicAgentRecord,
) -> Optional[tuple[str, Dict[str, Any]]]:
    """解析 alias_of / registry_agent 指向的静态 Agent 元数据。"""
    ref = (record.alias_of or record.registry_agent or "").strip()
    if not ref:
        return None
    if base_registry.has_agent(ref):
        return ref, dict(base_registry.agents[ref])
    from agent_framework.domain.plugin_registry import ensure_domains_loaded, get_domain_plugin, list_domains

    ensure_domains_loaded()
    for item in list_domains():
        reg = get_domain_plugin(item["name"]).create_registry()
        if reg.has_agent(ref):
            return ref, dict(reg.agents[ref])
    return None


def list_dynamic_agents_for_domain(domain: str) -> List[DynamicAgentRecord]:
    """领域本地 + 跨域 shared 动态 Agent。"""
    dom = domain.strip()
    store = get_dynamic_agent_store()
    records = list(store.list_agents(dom))
    shared = store.list_agents(SHARED_DOMAIN)
    seen = {record.name for record in records}
    for record in shared:
        if record.name not in seen:
            records.append(record)
            seen.add(record.name)
    return records


def merge_dynamic_agents(
    domain: str,
    base_registry: SubAgentRegistry,
) -> Tuple[SubAgentRegistry, Tuple[A2AEndpoint, ...]]:
    """将动态 Agent（含 shared）叠加到静态 Registry，并返回额外 A2A 端点。"""
    records = list_dynamic_agents_for_domain(domain)
    if not records:
        return base_registry, ()

    merged = SubAgentRegistry()
    for name in base_registry.get_agent_names():
        info = base_registry.agents[name]
        merged.register(
            name,
            base_registry._creators[name],
            description=str(info.get("description", "")),
            requires_tool=bool(info.get("requires_tool", False)),
            skills=info.get("skills"),
            meta={**dict(info), "source": "static"},
        )
    merged.register_guess_rules(base_registry._guess_rules)

    extra_a2a: List[A2AEndpoint] = []
    for record in records:
        ref = resolve_static_agent_reference(domain, base_registry, record)
        skills = list(record.skills or [])
        description = (record.description or record.name).strip() or record.name
        meta: Dict[str, Any] = {
            "source": "dynamic",
            "dynamic_source": record.source,
            "scope": record.scope,
        }
        if record.alias_of:
            meta["alias_of"] = record.alias_of
        if ref:
            ref_name, ref_info = ref
            meta["references"] = ref_name
            skills = _merge_skill_lists(ref_info.get("skills"), skills)
            if not record.description.strip():
                description = str(ref_info.get("description") or description)
        if record.source == "a2a":
            merged.register_metadata(
                record.name,
                description=description,
                skills=skills,
                meta=meta,
            )
            extra_a2a.append(
                A2AEndpoint(
                    node_name=record.a2a_node_name,
                    url=record.a2a_url,
                    description=description,
                    registry_agent=record.registry_agent or record.alias_of or record.name,
                )
            )
        else:
            merged.register_metadata(
                record.name,
                description=description,
                skills=skills,
                meta=meta,
            )
    return merged, tuple(extra_a2a)


def resolve_domain_registry_and_a2a(
    domain: str,
    plugin: Any,
) -> Tuple[SubAgentRegistry, Tuple[A2AEndpoint, ...]]:
    merged, dynamic_a2a = merge_dynamic_agents(domain, plugin.create_registry())
    static_a2a = plugin.resolved_a2a_endpoints()
    if not dynamic_a2a:
        return merged, static_a2a
    return merged, static_a2a + dynamic_a2a
