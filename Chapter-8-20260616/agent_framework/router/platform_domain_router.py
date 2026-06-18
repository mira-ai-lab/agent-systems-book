"""平台级跨领域路由：未指定 domain 时推断目标领域。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from agent_framework.config import DEFAULT_DOMAIN, create_llm
from agent_framework.domain.agent_catalog import build_domain_catalog
from agent_framework.domain.parsing import parse_json_from_llm
from agent_framework.domain.plugin import DomainPlugin
from agent_framework.domain.plugin_registry import ensure_domains_loaded, get_domain_plugin, list_domains
from agent_framework.router.prompts.loader import get_domain_classification_prompts
from agent_framework.tracing.trace_provider import span_name, trace_span


@dataclass(frozen=True)
class DomainCandidate:
    name: str
    score: float


def parse_domain_classification_response(
    raw: object,
    *,
    known_domains: set[str],
) -> List[DomainCandidate]:
    if isinstance(raw, dict):
        if "name" in raw and "score" in raw:
            raw = [raw]
        else:
            raw = []
    if not isinstance(raw, list):
        return []

    candidates: List[DomainCandidate] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        try:
            score = float(item.get("score", 0))
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(1.0, score))
        if name != "other" and name not in known_domains:
            continue
        candidates.append(DomainCandidate(name=name, score=score))
    return candidates


def select_domain(candidates: List[DomainCandidate]) -> Optional[str]:
    filtered = [c for c in candidates if c.name.lower() != "other"]
    if not filtered:
        return None
    best = max(filtered, key=lambda c: c.score)
    return best.name if best.score >= 0.3 else None


@trace_span(
    name=span_name("router.domain_classification"),
    attrs_args=["query"],
    record_result=False,
)
async def classify_domain(
    llm: ChatOpenAI,
    query: str,
    *,
    locale: str = "zh",
    note: str = "",
) -> List[DomainCandidate]:
    ensure_domains_loaded()
    known = {item["name"] for item in list_domains()}
    prompts = get_domain_classification_prompts(locale)
    domain_names = ", ".join(sorted(known)) or "other"
    catalog = build_domain_catalog(locale=locale)
    prompt = prompts["prompt_base"].format(
        domain_names=domain_names,
        note=note or prompts.get("note", ""),
        domain_catalog=catalog,
        query=query.strip(),
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    text = str(response.content or "").strip()
    try:
        parsed = parse_json_from_llm(text)
    except (ValueError, json.JSONDecodeError):
        parsed = []
    return parse_domain_classification_response(parsed, known_domains=known)


async def resolve_request_domain(
    query: str,
    domain: Optional[str],
    *,
    locale: str = "zh",
    llm: Optional[ChatOpenAI] = None,
) -> tuple[str, Optional[List[DomainCandidate]]]:
    """解析请求领域：显式 domain > DEFAULT_DOMAIN > LLM 跨域推断。"""
    explicit = (domain or "").strip()
    if explicit:
        get_domain_plugin(explicit)
        return explicit, None

    fallback = (DEFAULT_DOMAIN or "").strip()
    if fallback:
        get_domain_plugin(fallback)
        return fallback, None

    client = llm or create_llm()
    candidates = await classify_domain(client, query, locale=locale)
    selected = select_domain(candidates)
    if not selected:
        raise ValueError(
            "无法推断 domain。请显式指定 domain，或设置 DEFAULT_DOMAIN 环境变量。"
            "可用领域见 GET /v1/domains。"
        )
    return selected, candidates
