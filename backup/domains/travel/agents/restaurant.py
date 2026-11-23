"""RestaurantAgent — 美食推荐。"""

from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_core.tools import tool

from domains.travel.agents.base import build_agent
from domains.travel.agents.prompt_loader import travel_agent_prompt
from domains.travel.infra.travel_api import fetch_restaurants_from_api, require_non_empty

def _system_prompt() -> str:
    return travel_agent_prompt("RestaurantAgent")


@tool
async def recommend_restaurant(
    location: str,
    cuisine: Optional[str] = None,
    budget_cny_per_person: Optional[int] = None,
) -> Dict[str, Any]:
    """根据位置、菜系、预算推荐餐厅。"""
    ok, err = require_non_empty(location, "location")
    if not ok:
        return {"error": err}

    try:
        result = await fetch_restaurants_from_api(location, cuisine=cuisine, limit=10)
        if result.get("error"):
            return {"error": result["error"]}

        restaurants = result.get("restaurants", [])
        if budget_cny_per_person:
            restaurants = [
                r
                for r in restaurants
                if not r.get("avg_price_cny") or r["avg_price_cny"] <= budget_cny_per_person
            ]

        return {
            "location": location,
            "cuisine": cuisine,
            "budget_cny_per_person": budget_cny_per_person,
            "restaurants": restaurants[:10],
            "count": len(restaurants),
            "data_source": result.get("data_source"),
        }
    except Exception as e:
        return {"error": f"restaurant_query_failed: {str(e)}"}


def create_restaurant_agent() -> Any:
    return build_agent([recommend_restaurant], _system_prompt())
