"""Router 结果可观测字段：SDK / HTTP / SSE 统一口径。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def knowledge_matches_from_result(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    routing_plan = result.get("routing_plan") or {}
    metadata = routing_plan.get("metadata") or {}
    matches = metadata.get("knowledge_matches")
    if not matches:
        return []
    return list(matches)


def enrich_routing_observability(
    result: Dict[str, Any],
    *,
    domain: str,
    resolved_domain: Optional[str] = None,
) -> Dict[str, Any]:
    """补齐 resolved_domain / knowledge_matches 等顶层字段，便于 Debug 面板消费。"""
    resolved = (resolved_domain if resolved_domain is not None else domain).strip()
    result["domain"] = domain
    result["resolved_domain"] = resolved
    result["knowledge_matches"] = knowledge_matches_from_result(result)
    return result
