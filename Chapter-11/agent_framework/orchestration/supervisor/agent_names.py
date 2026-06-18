"""Registry Agent 名 ↔ Supervisor handoff 节点名（snake_case）。"""

from __future__ import annotations

import re


def registry_agent_to_node_name(agent_name: str) -> str:
    """WeatherAgent → weather_agent；FAQAgent → faq_agent。"""
    stem = agent_name[:-5] if agent_name.endswith("Agent") else agent_name
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", stem)
    if not parts:
        return f"{stem.lower()}_agent"
    slug = "_".join(p.lower() for p in parts)
    return slug if slug.endswith("_agent") else f"{slug}_agent"
