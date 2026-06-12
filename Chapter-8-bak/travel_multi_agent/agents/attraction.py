"""AttractionAgent — 景点推荐。"""

from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_core.tools import tool

from travel_multi_agent.agents.base import build_agent
from travel_multi_agent.infra.travel_api import fetch_attractions_from_api, require_non_empty

SYSTEM_PROMPT = """你是专业的景点推荐助手。

职责：
1. 只能使用 recommend_attractions 工具查询景点
2. city 参数必填，preferences 可选
3. 返回景点列表后，根据用户偏好推荐最合适的3-5个
4. 提供每个景点的简要介绍和推荐理由
5. 非景点相关问题，回复：我只能协助景点推荐

注意：
- preferences 可以是：历史文化、自然风光、现代建筑、亲子游、拍照打卡等
- 如果用户有特殊要求（如"必去XXX"），在推荐时优先考虑
"""


@tool
async def recommend_attractions(
    city: str,
    preferences: Optional[str] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """根据城市和偏好推荐旅游景点。"""
    ok, err = require_non_empty(city, "city")
    if not ok:
        return {"error": err}

    try:
        result = await fetch_attractions_from_api(city, limit=limit)
        if result.get("error"):
            return {"error": result["error"]}

        attractions = result.get("attractions", [])
        return {
            "city": city,
            "preferences": preferences,
            "attractions": attractions[:limit],
            "count": len(attractions),
            "data_source": result.get("data_source"),
        }
    except Exception as e:
        return {"error": f"attraction_query_failed: {str(e)}"}


def create_attraction_agent() -> Any:
    return build_agent([recommend_attractions], SYSTEM_PROMPT)
