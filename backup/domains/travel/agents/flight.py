"""FlightAgent — 航班查询。"""

from __future__ import annotations

from typing import Any, Dict

from langchain_core.tools import tool

from domains.travel.agents.base import build_agent
from domains.travel.agents.prompt_loader import travel_agent_prompt
from domains.travel.infra.travel_api import (
    fetch_flights_from_api,
    fetch_flights_from_variflight_api,
    require_non_empty,
)


@tool
async def search_flights(departure: str, arrival: str, date: str) -> Dict[str, Any]:
    """查询出发地到目的地的航班信息。"""
    ok1, err1 = require_non_empty(departure, "departure")
    if not ok1:
        return {"error": err1}

    ok2, err2 = require_non_empty(arrival, "arrival")
    if not ok2:
        return {"error": err2}

    try:
        result = await fetch_flights_from_variflight_api(departure, arrival, date, limit=10)
        if not result.get("error"):
            return {
                "departure": departure,
                "arrival": arrival,
                "date": date,
                "flights": result.get("flights", []),
                "data_source": "variflight",
            }
    except Exception:
        pass

    try:
        result = await fetch_flights_from_api(departure, arrival, date, limit=10)
        if not result.get("error"):
            return {
                "departure": departure,
                "arrival": arrival,
                "date": date,
                "flights": result.get("flights", []),
                "data_source": result.get("data_source"),
            }
    except Exception as e:
        return {"error": f"flight_query_failed: {str(e)}"}

    return {"error": "无法获取航班信息"}


def _system_prompt() -> str:
    return travel_agent_prompt("FlightAgent")


def create_flight_agent() -> Any:
    return build_agent([search_flights], _system_prompt())
