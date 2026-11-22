"""Publish Chapter-8 business agents as A2A services (reuse Chapter-7 infra).

:author: zhanghong, liuhui, xingrui
:date: 2026-11-22
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from travel_multi_agent.config import BOOK_ROOT

CHAPTER7_A2A_DIR = BOOK_ROOT / "Chapter-7" / "A2A"
if str(CHAPTER7_A2A_DIR) not in sys.path:
    sys.path.insert(0, str(CHAPTER7_A2A_DIR))

from server import (  # noqa: F401  # Chapter-7 generic A2A server helpers
    GenericAgentExecutor,
    build_agent_app,
    load_agent_card,
)


async def get_agent(agent_name: str) -> Any:
    """Map an Agent Card name to a Ch8 business agent wrapped as a stream runner."""
    from travel_multi_agent.agents.registry import create_travel_registry
    from agents import LangGraphAgentRunner

    registry = create_travel_registry()
    agent = await registry.get_agent(agent_name)
    return LangGraphAgentRunner(agent)


async def main() -> None:
    """Run one Ch8 A2A agent service for the given Agent Card."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Run one Ch8 A2A agent service")
    parser.add_argument("--agent-card", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--public-url", default=None)
    args = parser.parse_args()

    advertised_url = args.public_url or f"http://{args.host}:{args.port}/"
    card = load_agent_card(args.agent_card, advertised_url)
    agent = await get_agent(card.name)
    app = build_agent_app(card, agent)

    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
    server = uvicorn.Server(config)
    try:
        await server.serve()
    finally:
        from travel_multi_agent.mcp_tools import close_all_sessions

        await close_all_sessions()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
