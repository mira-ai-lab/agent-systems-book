"""平台 Agent Registry 产品层：跨 domain 汇总静态 + 动态 Agent。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent_framework.domain.dynamic_registry import (
    SHARED_DOMAIN,
    get_dynamic_agent_store,
    list_dynamic_agents_for_domain,
    merge_dynamic_agents,
)
from agent_framework.domain.plugin_registry import ensure_domains_loaded, get_domain_plugin, list_domains
from agent_framework.router.prompts.loader import get_domain_classification_prompts


def _format_agent_line(name: str, info: Dict[str, Any]) -> str:
    tags: List[str] = []
    source = str(info.get("source") or "static")
    if source == "dynamic":
        tags.append("dynamic")
    if str(info.get("scope") or "") == "shared":
        tags.append("shared")
    ref = info.get("alias_of") or info.get("references")
    if ref:
        tags.append(f"alias→{ref}")
    prefix = f"[{','.join(tags)}] " if tags else ""
    desc = str(info.get("description") or name)
    return f"{prefix}{name}: {desc}"


def list_platform_agent_entries(
    *,
    domain: Optional[str] = None,
    scope: Optional[str] = None,
    source: Optional[str] = None,
    include_shared: bool = True,
) -> List[Dict[str, Any]]:
    """汇总所有已注册领域的 Agent 元数据（静态 + 动态 + shared）。"""
    ensure_domains_loaded()
    entries: List[Dict[str, Any]] = []
    store = get_dynamic_agent_store()

    for item in list_domains():
        dom = item["name"]
        plugin = get_domain_plugin(dom)
        for meta in plugin.create_registry().list_agent_metadata():
            entries.append(
                {
                    **meta,
                    "domain": dom,
                    "registry_scope": "static",
                    "scope": "domain",
                    "source": "static",
                }
            )
        for record in store.list_agents(dom):
            entries.append(
                {
                    **record.to_dict(),
                    "domain": dom,
                    "registry_scope": "dynamic",
                    "source": "dynamic",
                }
            )

    if include_shared:
        for record in store.list_agents(SHARED_DOMAIN):
            entries.append(
                {
                    **record.to_dict(),
                    "domain": SHARED_DOMAIN,
                    "registry_scope": "dynamic",
                    "source": "dynamic",
                }
            )

    return filter_platform_agent_entries(
        entries,
        domain=domain,
        scope=scope,
        source=source,
    )


def filter_platform_agent_entries(
    entries: List[Dict[str, Any]],
    *,
    domain: Optional[str] = None,
    scope: Optional[str] = None,
    source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    dom_filter = (domain or "").strip()
    scope_filter = (scope or "").strip().lower()
    source_filter = (source or "").strip().lower()

    filtered: List[Dict[str, Any]] = []
    for entry in entries:
        entry_dom = str(entry.get("domain") or "")
        entry_scope = str(entry.get("scope") or "domain").lower()
        entry_source = str(entry.get("source") or entry.get("registry_scope") or "static").lower()

        if dom_filter:
            visible = entry_dom == dom_filter or entry_scope == "shared" or entry_dom == SHARED_DOMAIN
            if not visible:
                continue
        if scope_filter == "domain" and entry_scope != "domain":
            continue
        if scope_filter == "shared" and entry_scope != "shared":
            continue
        if source_filter in ("static", "dynamic") and entry_source != source_filter:
            continue
        filtered.append(entry)
    return filtered


def summarize_domain_agents(domain: str, *, max_agents: int = 8) -> str:
    """供跨域 domain catalog 使用的 Agent 摘要（含 dynamic/shared/alias）。"""
    plugin = get_domain_plugin(domain)
    merged, _ = merge_dynamic_agents(domain, plugin.create_registry())
    names = merged.get_agent_names()[:max_agents]
    parts = [_format_agent_line(name, merged.agents.get(name, {})) for name in names]
    if not parts:
        return "无子 Agent"
    return "；".join(parts)


def summarize_shared_agents(*, max_agents: int = 8) -> str:
    store = get_dynamic_agent_store()
    records = store.list_agents(SHARED_DOMAIN)[:max_agents]
    if not records:
        return ""
    lines = []
    for record in records:
        ref = record.alias_of or record.registry_agent
        suffix = f" (alias→{ref})" if ref else ""
        lines.append(f"[shared,dynamic] {record.name}: {record.description or record.name}{suffix}")
    return "；".join(lines)


def build_domain_catalog(*, locale: str = "zh") -> str:
    """LLM 跨域推断用 catalog：静态 + 动态 + shared Agent。"""
    ensure_domains_loaded()
    prompts = get_domain_classification_prompts(locale)
    template = prompts["domain_template"]
    blocks: List[str] = []
    for item in list_domains():
        plugin = get_domain_plugin(item["name"])
        blocks.append(
            template.format(
                name=plugin.name,
                display_name=plugin.display_name or plugin.name,
                agents=summarize_domain_agents(plugin.name),
            )
        )
    shared = summarize_shared_agents()
    if shared:
        blocks.append(f"[跨域共享 Agent]\n{shared}")
    return "\n".join(blocks)


def list_effective_agent_names(domain: str) -> List[str]:
    plugin = get_domain_plugin(domain)
    merged, _ = merge_dynamic_agents(domain, plugin.create_registry())
    return merged.get_agent_names()


def list_dynamic_agent_records(domain: str) -> List[Dict[str, Any]]:
    return [record.to_dict() for record in list_dynamic_agents_for_domain(domain)]
