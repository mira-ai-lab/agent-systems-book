"""Shared A2A endpoint configuration for Chapter-8 services.

Domain-agent identities come from the ``DomainAgentName`` enum in
``travel_multi_agent.agents.registry``; ports and URLs are keyed by that enum.
Override them via ``A2A_<NAME>_PORT`` / ``A2A_<NAME>_URL`` (e.g.
``A2A_WEATHER_PORT``), where ``<NAME>`` is the uppercase enum member name.

:author: zhanghong, liuhui, xingrui
:date: 2026-11-22
"""

from __future__ import annotations

import os

from travel_multi_agent.agents.registry import DomainAgentName

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORTS = {
    DomainAgentName.WEATHER: 9031,
    DomainAgentName.HOTEL: 9032,
    DomainAgentName.RESTAURANT: 9033,
    DomainAgentName.FLIGHT: 9034,
    DomainAgentName.ITINERARY: 9035,
}
assert set(DEFAULT_PORTS) == set(DomainAgentName), "ports must cover every domain agent"


def _coerce(agent_name: DomainAgentName | str) -> DomainAgentName:
    """Normalize an agent name or enum member into a ``DomainAgentName``."""
    if isinstance(agent_name, DomainAgentName):
        return agent_name
    return DomainAgentName(agent_name)


def server_host() -> str:
    """Return the host on which local A2A server processes bind."""
    return (os.getenv("A2A_SERVER_HOST") or DEFAULT_HOST).strip()


def service_port(agent_name: DomainAgentName | str) -> int:
    """Return the configured local port for one A2A service."""
    key = _coerce(agent_name)
    raw = os.getenv(f"A2A_{key.name}_PORT") or str(DEFAULT_PORTS[key])
    return int(raw)


def service_base_url(agent_name: DomainAgentName | str) -> str:
    """Return the client endpoint for one A2A service."""
    key = _coerce(agent_name)
    configured = (os.getenv(f"A2A_{key.name}_URL") or "").strip()
    if configured:
        return configured.rstrip("/") + "/"
    return f"http://{server_host()}:{service_port(key)}/"
