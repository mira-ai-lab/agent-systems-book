"""企业路由引擎统一 SDK 入口：丢 query 即可。"""

from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI

from agent_framework.bootstrap.platform import create_runtime
from agent_framework.config import DEFAULT_DOMAIN
from agent_framework.domain.plugin_registry import get_domain_plugin
from agent_framework.i18n.locale import normalize_locale
from agent_framework.router.config import RouterConfig
from agent_framework.router.observability import enrich_routing_observability


async def _resolve_route_domain(
    query: str,
    domain: Optional[str],
    *,
    locale: str,
    llm: Optional[ChatOpenAI] = None,
) -> str:
    explicit = (domain or "").strip()
    if explicit:
        get_domain_plugin(explicit)
        return explicit

    fallback = (DEFAULT_DOMAIN or "").strip()
    if fallback:
        get_domain_plugin(fallback)
        return fallback

    from agent_framework.router.platform_domain_router import resolve_request_domain

    selected, _ = await resolve_request_domain(query, None, locale=locale, llm=llm)
    return selected


async def route(
    query: str,
    *,
    domain: Optional[str] = None,
    profile: str = "auto",
    locale: str = "zh",
    user_id: str = "default",
    thread_id: str = "default",
    conversation_history: Optional[str] = None,
    timeout_sec: Optional[float] = None,
    transport: Optional[str] = None,
    llm: Optional[ChatOpenAI] = None,
    router_config: Optional[RouterConfig] = None,
) -> Dict[str, Any]:
    """统一路由入口：解析 domain → 创建 runtime → 执行请求。

    推荐用法::

        result = await route("退货政策是什么？", domain="customer_service")
        # 或仅传 query（需 DEFAULT_DOMAIN 或 LLM 跨域推断）
        result = await route("退货政策是什么？")
    """
    loc = normalize_locale(locale)
    resolved_domain = await _resolve_route_domain(query, domain, locale=loc, llm=llm)
    runtime = create_runtime(
        resolved_domain,
        profile=profile,
        llm=llm,
        user_id=user_id,
        transport=transport,
        locale=loc,
        router_config=router_config,
    )

    request_kwargs: Dict[str, Any] = {}
    if timeout_sec is not None:
        request_kwargs["timeout_sec"] = timeout_sec
    if conversation_history and profile == "auto":
        request_kwargs["conversation_history"] = conversation_history

    result = await runtime.process_request(query, thread_id=thread_id, **request_kwargs)
    enrich_routing_observability(result, domain=resolved_domain, resolved_domain=resolved_domain)
    result["locale"] = loc
    return result
