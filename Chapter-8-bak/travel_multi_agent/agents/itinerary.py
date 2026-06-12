"""ItineraryAgent — 行程规划。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from travel_multi_agent.agents.base import build_agent
from travel_multi_agent.infra.travel_api import (
    build_itinerary_from_candidates,
    fetch_attractions_from_api,
    fetch_hotels_from_api,
    fetch_restaurants_from_api,
    require_non_empty,
)


@tool
async def plan_itinerary(
    departure_city: str,
    destination_city: str,
    days: int,
    weather_summary: Optional[str] = None,
    attraction_list: Optional[List[Dict[str, Any]]] = None,
    preferences: Optional[str] = None,
) -> Dict[str, Any]:
    """综合各种信息生成详细的每日行程安排。"""
    ok1, err1 = require_non_empty(departure_city, "departure_city")
    if not ok1:
        return {"error": err1}

    ok2, err2 = require_non_empty(destination_city, "destination_city")
    if not ok2:
        return {"error": err2}

    try:
        if not attraction_list:
            attr_result = await fetch_attractions_from_api(destination_city, limit=15)
            attraction_list = attr_result.get("attractions", [])

        rest_result = await fetch_restaurants_from_api(destination_city, limit=10)
        restaurant_list = rest_result.get("restaurants", [])

        hotel_result = await fetch_hotels_from_api(destination_city, limit=5)
        hotel_list = hotel_result.get("hotels", [])

        itinerary = build_itinerary_from_candidates(
            departure_city=departure_city,
            destination_city=destination_city,
            days=days,
            preferences=preferences,
            attractions=attraction_list,
            restaurants=restaurant_list,
            hotels=hotel_list,
        )

        if weather_summary:
            itinerary["weather_summary"] = weather_summary

        return itinerary
    except Exception as e:
        return {"error": f"itinerary_planning_failed: {str(e)}"}


def _system_prompt() -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    return f"""你是专业的行程规划助手。

当前日期（本地时间）：{today}

职责：
1. 只能使用 plan_itinerary 工具生成行程
2. departure_city、destination_city、days 三个参数必填
3. 综合考虑天气、景点、交通、住宿等因素
4. 生成详细的每日行程（上午、下午、晚上）
5. 提供交通建议、住宿推荐、注意事项
6. 非行程规划问题，回复：我只能协助行程规划

注意：
- 行程安排要合理，考虑景点间的距离和游览时间
- 每天安排2-3个主要景点，避免过于紧凑
- 预留用餐时间和休息时间
- 考虑天气因素给出出行建议
- 用户说「今天」时必须对应当前日期 {today}，不要臆造其他日期
"""


def create_itinerary_agent() -> Any:
    return build_agent([plan_itinerary], _system_prompt())
