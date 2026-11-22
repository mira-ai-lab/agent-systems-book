"""Shared endpoint configuration for Chapter-7 A2A services.

The module is used by ``server.py``, ``supervisor.py``, ``process_manager.py``,
and the Notebook. Explicit client URLs take precedence so the calling code
remains unchanged when a locally started service is replaced by an
independently hosted endpoint.

Main capabilities:

- Resolve service ports, base URLs, and advertised URLs.
- Configure local service endpoints for the Notebook.

:author: zhanghong, liuhui
:date: 2026-11-22
"""

from __future__ import annotations

import os
import socket
from typing import Mapping


ServiceName = str

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORTS: dict[ServiceName, int] = {
    "planner": 9011,
    "hotel": 9012,
}
SERVICE_NAMES = tuple(DEFAULT_PORTS)
TASK_AGENT_SERVICES = ("hotel",)


def _service_env_prefix(service: ServiceName) -> str:
    """Return the normalized environment prefix for one service."""
    return f"A2A_{service.upper()}_"


def _service_url_env(service: ServiceName) -> str:
    return _service_env_prefix(service) + "URL"


def _service_public_url_env(service: ServiceName) -> str:
    return _service_env_prefix(service) + "PUBLIC_URL"


def server_host() -> str:
    """Return the host on which local A2A server processes bind."""
    return (os.getenv("A2A_SERVER_HOST") or DEFAULT_HOST).strip()


def service_port(service: ServiceName) -> int:
    """Return the configured local port for one A2A service."""
    if service not in DEFAULT_PORTS:
        raise ValueError(f"Unknown A2A service: {service}")
    env_name = _service_env_prefix(service) + "PORT"
    raw_value = (os.getenv(env_name) or str(DEFAULT_PORTS[service])).strip()
    port = int(raw_value)
    if not 1 <= port <= 65535:
        raise ValueError(f"{env_name} must be between 1 and 65535")
    return port


def service_base_url(service: ServiceName) -> str:
    """Return the client endpoint for one A2A service.

    ``A2A_PLANNER_URL`` and ``A2A_HOTEL_URL`` are intended for remote
    deployments. When they are absent, the endpoint is derived from the local
    bind host and port.
    """
    configured = (os.getenv(_service_url_env(service)) or "").strip()
    if configured:
        return configured.rstrip("/") + "/"
    return f"http://{server_host()}:{service_port(service)}/"


def advertised_url(host: str, port: int) -> str:
    """Return a reachable URL for a locally bound service."""
    advertised_host = host.strip()
    if advertised_host in {"0.0.0.0", "::", ""}:
        advertised_host = DEFAULT_HOST
    return f"http://{advertised_host}:{port}/"


def find_free_port(host: str = DEFAULT_HOST) -> int:
    """Return an available TCP port on the requested local interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def configure_local_services(
    *,
    ports: Mapping[str, int] | None = None,
    host: str = DEFAULT_HOST,
) -> None:
    """Synchronize server ports and client URLs for one Notebook session."""
    selected_ports = {
        service: int(port)
        for service, port in DEFAULT_PORTS.items()
    }
    selected_ports.update(
        {
            service: int(port)
            for service, port in (ports or {}).items()
        }
    )

    normalized_host = host.strip()
    os.environ["A2A_SERVER_HOST"] = normalized_host
    for service, port in selected_ports.items():
        if not 1 <= port <= 65535:
            raise ValueError(
                f"Port for {service} must be between 1 and 65535"
            )
        base_url = f"http://{normalized_host}:{port}/"
        os.environ[_service_env_prefix(service) + "PORT"] = str(port)
        os.environ[_service_url_env(service)] = base_url
        os.environ[_service_public_url_env(service)] = base_url
