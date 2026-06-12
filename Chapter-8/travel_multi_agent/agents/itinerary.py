"""ItineraryAgent — 候选 POI 拉取 + 确定性逐日行程骨架生成。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from travel_multi_agent.agents.base import build_agent
from travel_multi_agent.agents.prompt_fragments import (
    MULTI_ENTITY_TOOL_RULES,
    agent_time_anchor_block,
)
from travel_multi_agent.infra.travel_api import (
    build_itinerary_from_candidates,
    fetch_attractions_from_api,
    require_non_empty,
)


def _normalize_pois(raw: Any) -> List[Dict[str, Any]]:
    """将 fetch_candidate_pois 返回值或 POI 列表规范化为 build 函数可用的 dict 列表。"""
    if raw is None:
        return []
    if isinstance(raw, dict):
        if "candidate_pois" in raw:
            raw = raw["candidate_pois"]
        elif "attractions" in raw:
            raw = raw["attractions"]
        else:
            return []
    if not isinstance(raw, list):
        return []
    return [p for p in raw if isinstance(p, dict) and (p.get("name") or p.get("title"))]


@tool
async def fetch_candidate_pois(
    city: str,
    preferences: Optional[str] = None,
    limit: int = 15,
) -> Dict[str, Any]:
    """拉取指定城市的候选兴趣点（景点 POI），供 plan_itinerary 生成逐日骨架。"""
    ok, err = require_non_empty(city, "city")
    if not ok:
        return {"error": err}

    try:
        result = await fetch_attractions_from_api(city, limit=limit)
        if result.get("error"):
            return {"error": result["error"]}

        pois = result.get("attractions", [])
        return {
            "city": city,
            "preferences": preferences,
            "candidate_pois": pois[:limit],
            "count": len(pois),
            "data_source": result.get("data_source"),
        }
    except Exception as e:
        return {"error": f"poi_query_failed: {str(e)}"}


@tool
async def plan_itinerary(
    city: str,
    days: int = 3,
    candidate_pois: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """根据候选 POI 确定性生成逐日行程骨架；未传 POI 时自动拉取。"""
    ok, err = require_non_empty(city, "city")
    if not ok:
        return {"error": err}

    n_days = max(1, min(int(days or 3), 14))
    pois = _normalize_pois(candidate_pois)

    try:
        if not pois:
            attr_result = await fetch_attractions_from_api(city, limit=15)
            if attr_result.get("error"):
                return {"error": attr_result["error"]}
            pois = _normalize_pois(attr_result.get("attractions"))

        itinerary = build_itinerary_from_candidates(
            departure_city=city,
            destination_city=city,
            days=n_days,
            attractions=pois,
            restaurants=[],
            hotels=[],
        )
        itinerary["city"] = city
        itinerary["poi_count"] = len(pois)
        itinerary["skeleton_note"] = "plan 字段为基于候选 POI 的确定性逐日骨架。"
        return itinerary
    except Exception as e:
        return {"error": f"itinerary_planning_failed: {str(e)}"}


def _system_prompt() -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    return f"""你是专业的行程规划助手。

当前日期（本地时间）：{today}

职责：
1. 多城任务：对每个城市先 fetch_candidate_pois(city)，再 plan_itinerary(city, days, candidate_pois=...)
2. plan_itinerary 只需 city、days；candidate_pois 可传入上一步工具返回的 candidate_pois 列表
3. 若未传 candidate_pois，plan_itinerary 会自动拉取该城 POI
4. 拿到 plan 骨架后，用中文润色并补充交通/注意事项（可结合任务描述中的天气、酒店等信息）
5. 非行程规划问题，回复：我只能协助行程规划

注意：
- plan 骨架中的 POI 名称不要臆改
- 每天 2–3 个 POI，避免过满
{agent_time_anchor_block()}
{MULTI_ENTITY_TOOL_RULES}
"""


def create_itinerary_agent() -> Any:
    return build_agent([fetch_candidate_pois, plan_itinerary], _system_prompt())
