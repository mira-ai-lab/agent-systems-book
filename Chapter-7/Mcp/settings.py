"""Shared configuration for the Chapter-7 MCP server and clients.

Main capabilities:

- Resolve the server host, port, and Streamable HTTP endpoint.
- Provide reusable configuration helpers for the server, client, and Notebook.

:author: zhanghong, liuhui
:date: 2026-11-22
"""

from __future__ import annotations

import os
import socket

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MCP_PATH = "/mcp"


def server_host() -> str:
    """Return the host address on which the MCP server binds."""
    return (os.getenv("HOTEL_MCP_HOST") or DEFAULT_HOST).strip()


def server_port() -> int:
    """Return the port on which the MCP server binds."""
    raw = (os.getenv("HOTEL_MCP_PORT") or str(DEFAULT_PORT)).strip()
    return int(raw)


def mcp_url() -> str:
    """Return the Streamable HTTP endpoint used by MCP clients.

    ``HOTEL_MCP_URL`` takes precedence because a client may connect to a
    remote server instead of the locally bound host and port.
    """
    configured = (os.getenv("HOTEL_MCP_URL") or "").strip()
    if configured:
        return configured
    return f"http://{server_host()}:{server_port()}{MCP_PATH}"


def find_free_port(host: str = DEFAULT_HOST) -> int:
    """Return an available TCP port on the requested host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def configure_server(
    *,
    host: str | None = None,
    port: int | None = None,
    url: str | None = None,
) -> None:
    """Set process-local server configuration and synchronize the client URL.

    The notebook uses this function when it selects a dynamic port. Explicit
    ``url`` takes precedence; otherwise the URL is derived from host, port and
    ``MCP_PATH``.
    """
    if host is not None:
        os.environ["HOTEL_MCP_HOST"] = host.strip()
    if port is not None:
        normalized_port = int(port)
        if not 1 <= normalized_port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        os.environ["HOTEL_MCP_PORT"] = str(normalized_port)
    if url is not None:
        os.environ["HOTEL_MCP_URL"] = url.strip()
    elif host is not None or port is not None:
        os.environ["HOTEL_MCP_URL"] = (
            f"http://{server_host()}:{server_port()}{MCP_PATH}"
        )
