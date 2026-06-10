"""RestaurantAgent — 美食推荐。"""

from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_core.tools import tool

from travel_multi_agent.agents.base import build_agent
from travel_multi_agent.infra.travel_api import fetch_restaurants_from_api, require_non_empty

SYSTEM_PROMPT = """你是专业的美食推荐助手。

职责：
1. 只能使用 recommend_restaurant 工具查询餐厅
2. location 参数必填，cuisine 和 budget 可选
3. 返回餐厅列表后，根据用户偏好推荐最合适的3-5家
4. 提供每家餐厅的特色菜和推荐理由
5. 非美食相关问题，回复：我只能协助餐厅推荐

注意：
- cuisine 可以是：本地菜、海鲜、川菜、粤菜、日料、西餐等
- 如果用户有特殊要求（如"适合聚餐"、"环境好"），在推荐时考虑
"""


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
    return build_agent([recommend_restaurant], SYSTEM_PROMPT)
