"""Registry 联邦：合并本地 catalog 与远程 cluster `/v1/agents`。"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

from agent_framework.domain.agent_catalog import (
    filter_platform_agent_entries,
    list_platform_agent_entries,
)
from agent_framework.tracing import get_logger, log_info

logger = get_logger(__name__)

DEFAULT_PROBE_TIMEOUT_SEC = 5.0
AGENT_CARD_PATHS = (
    "/.well-known/agent.json",
    "/.well-known/agent-card.json",
)
CLUSTER_HEALTH_PATHS = ("/health", "/ready")


def parse_federation_urls(raw: Optional[str] = None) -> List[str]:
    value = raw if raw is not None else os.getenv("REGISTRY_FEDERATION_URLS", "")
    return [item.strip().rstrip("/") for item in (value or "").split(",") if item.strip()]


def federation_api_key() -> str:
    return os.getenv("REGISTRY_FEDERATION_API_KEY", "").strip()


def federation_cluster_name(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.netloc:
        return parsed.netloc
    return base_url.strip("/") or "unknown"


def normalize_federated_agent(
    entry: Dict[str, Any],
    *,
    cluster: str,
    base_url: str,
) -> Dict[str, Any]:
    normalized = dict(entry)
    normalized.setdefault("name", "")
    normalized["origin"] = "federated"
    normalized["federation_cluster"] = cluster
    normalized["federation_base_url"] = base_url
    if not normalized.get("source"):
        normalized["source"] = "federated"
    return normalized


def tag_local_agents(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tagged: List[Dict[str, Any]] = []
    for entry in entries:
        item = dict(entry)
        item["origin"] = "local"
        tagged.append(item)
    return tagged


async def fetch_remote_registry(
    base_url: str,
    *,
    api_key: str = "",
    timeout_sec: float = DEFAULT_PROBE_TIMEOUT_SEC,
) -> Dict[str, Any]:
    headers: Dict[str, str] = {}
    if api_key:
        headers["X-API-Key"] = api_key
    url = f"{base_url.rstrip('/')}/v1/agents"
    async with httpx.AsyncClient(timeout=timeout_sec) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"远程 Registry 响应无效: {base_url}")
    return payload


async def probe_cluster_health(
    base_url: str,
    *,
    api_key: str = "",
    timeout_sec: float = DEFAULT_PROBE_TIMEOUT_SEC,
) -> Dict[str, Any]:
    headers: Dict[str, str] = {}
    if api_key:
        headers["X-API-Key"] = api_key
    last_error = ""
    async with httpx.AsyncClient(timeout=timeout_sec) as client:
        for path in CLUSTER_HEALTH_PATHS:
            url = f"{base_url.rstrip('/')}{path}"
            try:
                response = await client.get(url, headers=headers)
                if response.status_code < 500:
                    return {
                        "reachable": True,
                        "probe_url": url,
                        "status_code": response.status_code,
                    }
            except Exception as exc:
                last_error = str(exc)
    return {"reachable": False, "error": last_error or "unreachable"}


async def probe_a2a_endpoint(
    a2a_url: str,
    *,
    timeout_sec: float = DEFAULT_PROBE_TIMEOUT_SEC,
) -> Dict[str, Any]:
    """A2A 端点健康探测占位：agent card 或根路径可达性。"""
    base = (a2a_url or "").strip().rstrip("/")
    if not base:
        return {"reachable": False, "error": "empty a2a_url"}
    candidates = [f"{base}{path}" for path in AGENT_CARD_PATHS] + [f"{base}/"]
    last_error = ""
    async with httpx.AsyncClient(timeout=timeout_sec) as client:
        for url in candidates:
            try:
                response = await client.get(url)
                if response.status_code < 500:
                    return {
                        "reachable": True,
                        "probe_url": url,
                        "status_code": response.status_code,
                    }
            except Exception as exc:
                last_error = str(exc)
    return {"reachable": False, "error": last_error or "unreachable"}


async def _attach_a2a_health(entries: List[Dict[str, Any]]) -> None:
    for entry in entries:
        a2a_url = str(entry.get("a2a_url") or "").strip()
        if not a2a_url:
            continue
        entry["a2a_health"] = await probe_a2a_endpoint(a2a_url)


async def list_federation_clusters(
    *,
    include_health: bool = False,
    federation_urls: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    api_key = federation_api_key()
    clusters: List[Dict[str, Any]] = []
    for base_url in federation_urls or parse_federation_urls():
        cluster = federation_cluster_name(base_url)
        item: Dict[str, Any] = {
            "cluster": cluster,
            "base_url": base_url,
            "status": "configured",
        }
        if include_health:
            item["health"] = await probe_cluster_health(base_url, api_key=api_key)
        clusters.append(item)
    return clusters


async def list_federated_agents(
    *,
    domain: Optional[str] = None,
    scope: Optional[str] = None,
    source: Optional[str] = None,
    include_health: bool = False,
    federation_urls: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """合并本地 Agent 目录与远程 federation cluster catalog。"""
    local = tag_local_agents(
        list_platform_agent_entries(domain=domain, scope=scope, source=source)
    )
    federated: List[Dict[str, Any]] = []
    federation_meta: List[Dict[str, Any]] = []
    api_key = federation_api_key()

    for base_url in federation_urls or parse_federation_urls():
        cluster = federation_cluster_name(base_url)
        meta: Dict[str, Any] = {
            "cluster": cluster,
            "base_url": base_url,
        }
        try:
            payload = await fetch_remote_registry(base_url, api_key=api_key)
            remote_agents = payload.get("agents") or []
            if not isinstance(remote_agents, list):
                raise ValueError("agents 字段必须是 list")
            normalized = [
                normalize_federated_agent(item, cluster=cluster, base_url=base_url)
                for item in remote_agents
                if isinstance(item, dict)
            ]
            normalized = filter_platform_agent_entries(
                normalized,
                domain=domain,
                scope=scope,
                source=source,
            )
            federated.extend(normalized)
            meta.update({"status": "ok", "count": len(normalized), "remote_count": len(remote_agents)})
            log_info(
                logger,
                "registry.federation.fetch.ok",
                cluster=cluster,
                count=len(normalized),
            )
        except Exception as exc:
            meta.update({"status": "error", "count": 0, "error": str(exc)})
            log_info(
                logger,
                "registry.federation.fetch.error",
                cluster=cluster,
                error=str(exc),
            )
        if include_health:
            meta["health"] = await probe_cluster_health(base_url, api_key=api_key)
        federation_meta.append(meta)

    agents = local + federated
    if include_health:
        await _attach_a2a_health(agents)

    return {
        "agents": agents,
        "count": len(agents),
        "local_count": len(local),
        "federated_count": len(federated),
        "federation": federation_meta,
        "filters": {
            "domain": domain,
            "scope": scope,
            "source": source,
            "federated": True,
            "health": include_health,
        },
    }
