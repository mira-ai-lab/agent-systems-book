"""加载 domains/{domain}/locales 与 agents/locales → 话术字段。"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any, Iterable, Sequence

from agent_framework.domain.domain_prompts import DomainPrompts
from agent_framework.i18n.locale import normalize_locale
from agent_framework.tracing import get_logger, log_info

logger = get_logger(__name__)

_DOMAIN_PROMPT_FIELDS = (
    "central_agent_system",
    "aggregation",
    "facts_prompt",
    "decomposition_prompt",
    "dependency_system",
    "dependency_user",
    "agent_routing",
    "supervisor_system",
    "multi_task_title",
    "single_task_title",
    "aggregation_skip_hint",
    "memory_aggregation_instruction",
)


def _domain_root(domain: str) -> Path | None:
    try:
        mod = import_module(f"domains.{domain}")
        return Path(mod.__file__).resolve().parent
    except ModuleNotFoundError:
        return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _merge_locale_dict(
    primary: dict[str, Any],
    fallback: dict[str, Any],
    *,
    domain: str,
    locale: str,
    kind: str,
    fields: Sequence[str],
) -> dict[str, Any]:
    merged = dict(fallback)
    missing: list[str] = []
    for field in fields:
        raw = primary.get(field)
        if raw is not None and str(raw).strip():
            merged[field] = raw
        else:
            if field not in primary or not str(primary.get(field) or "").strip():
                missing.append(field)
    if missing and locale != "zh":
        log_info(
            logger,
            "locale.missing_keys",
            domain=domain,
            locale=locale,
            kind=kind,
            keys=missing,
            fallback="zh",
        )
    return merged


def reset_locale_loader_cache() -> None:
    load_domain_locale_payload.cache_clear()
    load_agent_locale_payload.cache_clear()


@lru_cache(maxsize=64)
def load_domain_locale_payload(domain: str, locale: str) -> dict[str, Any]:
    loc = normalize_locale(locale)
    root = _domain_root(domain)
    if root is None:
        raise FileNotFoundError(f"未知领域: {domain}")
    primary_path = root / "locales" / f"{loc}.json"
    zh_path = root / "locales" / "zh.json"
    if not primary_path.is_file():
        if loc != "zh" and zh_path.is_file():
            log_info(
                logger,
                "locale.missing_file",
                domain=domain,
                locale=loc,
                path=str(primary_path),
                fallback="zh",
            )
            return _read_json(zh_path)
        raise FileNotFoundError(f"领域 locale 不存在: {domain}/{loc} ({primary_path})")
    primary = _read_json(primary_path)
    if loc == "zh" or not zh_path.is_file():
        return primary
    zh_data = _read_json(zh_path)
    return _merge_locale_dict(
        primary,
        zh_data,
        domain=domain,
        locale=loc,
        kind="domain",
        fields=_DOMAIN_PROMPT_FIELDS,
    )


def domain_prompts_from_locale(domain: str, locale: str = "zh") -> DomainPrompts:
    data = load_domain_locale_payload(domain, locale)
    kwargs = {field: str(data.get(field) or "") for field in _DOMAIN_PROMPT_FIELDS}
    return DomainPrompts(**kwargs)


def _agent_locale_path(domain: str, locale: str) -> Path | None:
    root = _domain_root(domain)
    if root is None:
        return None
    path = root / "agents" / "locales" / f"{locale}.json"
    return path if path.is_file() else None


@lru_cache(maxsize=64)
def load_agent_locale_payload(domain: str, locale: str) -> dict[str, Any]:
    loc = normalize_locale(locale)
    root = _domain_root(domain)
    if root is None:
        raise FileNotFoundError(f"未知领域: {domain}")
    primary_path = root / "agents" / "locales" / f"{loc}.json"
    zh_path = root / "agents" / "locales" / "zh.json"
    if not primary_path.is_file():
        if loc != "zh" and zh_path.is_file():
            log_info(
                logger,
                "locale.missing_file",
                domain=domain,
                locale=loc,
                path=str(primary_path),
                fallback="zh",
                kind="agent",
            )
            return _read_json(zh_path)
        raise FileNotFoundError(f"子 Agent locale 不存在: {domain}/{loc} ({primary_path})")
    primary = _read_json(primary_path)
    if loc == "zh" or not zh_path.is_file():
        return primary
    zh_data = _read_json(zh_path)
    keys = sorted({*zh_data.keys(), *primary.keys()} - {"fragments"})
    return _merge_locale_dict(
        primary,
        zh_data,
        domain=domain,
        locale=loc,
        kind="agent",
        fields=keys,
    )


def agent_system_prompt(domain: str, agent_name: str, locale: str = "zh") -> str:
    data = load_agent_locale_payload(domain, locale)
    block = data.get(agent_name)
    if not isinstance(block, dict):
        raise KeyError(f"子 Agent locale 缺少 {domain}/{agent_name}")
    prompt = str(block.get("system_prompt") or "").strip()
    if not prompt:
        raise KeyError(f"子 Agent system_prompt 为空: {domain}/{agent_name}")
    return prompt


def agent_fragment(domain: str, name: str, locale: str = "zh") -> str:
    data = load_agent_locale_payload(domain, locale)
    fragments = data.get("fragments") or {}
    if not isinstance(fragments, dict):
        raise KeyError(f"子 Agent fragments 无效: {domain}")
    text = str(fragments.get(name) or "").strip()
    if not text:
        raise KeyError(f"子 Agent fragment 缺少 {domain}/{name}")
    return text
