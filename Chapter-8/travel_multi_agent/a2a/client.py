"""A2A discovery and client registry reused from Chapter-7.

Chapter-8 domain agents are started as separate A2A services. The central agent
loads their Agent Cards from ``agents/agent_cards``, registers each card by its
``name``, and builds one persistent A2A client per endpoint.

:author: zhanghong, liuhui, xingrui
:date: 2026-11-22
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from travel_multi_agent.config import BOOK_ROOT

CHAPTER7_A2A_DIR = BOOK_ROOT / "Chapter-7" / "A2A"
if str(CHAPTER7_A2A_DIR) not in sys.path:
    sys.path.insert(0, str(CHAPTER7_A2A_DIR))

from a2a_client import A2ACallResult, A2AClient, A2AClientError, discover_agent_card


class A2ARegistry:
    """Map domain-agent Agent Card names to their A2A endpoints and clients."""

    def __init__(self, cards: Dict[str, Dict[str, Any]]) -> None:
        self.cards = cards
        self._clients: Dict[str, A2AClient] = {}

    @classmethod
    def from_directory(cls, cards_dir: Path) -> "A2ARegistry":
        """Load every ``*.json`` Agent Card in ``cards_dir`` keyed by card name."""
        cards: Dict[str, Dict[str, Any]] = {}
        for path in sorted(Path(cards_dir).glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            name = payload.get("name")
            if name:
                cards[name] = payload
        return cls(cards)

    def names(self) -> List[str]:
        """Return the registered Agent Card names."""
        return list(self.cards)

    def endpoint_for(self, agent_name: str) -> str:
        """Return the advertised endpoint for one Agent Card."""
        card = self.cards.get(agent_name)
        if card is None:
            raise A2AClientError(f"No Agent Card registered for {agent_name}")
        interfaces = card.get("supportedInterfaces") or []
        if not interfaces or not interfaces[0].get("url"):
            raise A2AClientError(f"Agent Card {agent_name} has no endpoint URL")
        return interfaces[0]["url"].rstrip("/") + "/"

    def client_for(self, agent_name: str) -> A2AClient:
        """Return (and cache) the persistent A2A client for one Agent Card."""
        if agent_name not in self._clients:
            self._clients[agent_name] = A2AClient(self.endpoint_for(agent_name))
        return self._clients[agent_name]

    async def close(self) -> None:
        """Close every underlying HTTP connection."""
        for client in self._clients.values():
            await client.close()
        self._clients.clear()


__all__ = [
    "A2ACallResult",
    "A2AClient",
    "A2AClientError",
    "A2ARegistry",
    "discover_agent_card",
]