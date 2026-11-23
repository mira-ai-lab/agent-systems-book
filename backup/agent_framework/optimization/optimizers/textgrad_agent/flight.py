"""FlightAgent system_prompt 的 textgrad graph 优化（Agent-B1 薄封装，逻辑在 optimize.py）。"""

from __future__ import annotations

from .optimize import (
    FLIGHT_AGENT_NAME,
    default_flight_agent_report_path,
    optimize_flight_agent_prompt_graph,
)

__all__ = [
    "FLIGHT_AGENT_NAME",
    "default_flight_agent_report_path",
    "optimize_flight_agent_prompt_graph",
]
