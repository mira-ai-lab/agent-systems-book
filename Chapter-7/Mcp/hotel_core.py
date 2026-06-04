"""酒店 POI 查询核心逻辑（无 LangChain 依赖，供 MCP / 工具层共用）。"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from mcp_paths import bootstrap_paths
from travel_common import fetch_hotels_from_api, norm_text, require_non_empty

bootstrap_paths()

_DISTRICT_RE = re.compile(r"[\u4e00-\u9fff]{1,12}(?:区|县)")
_POI_HINT = re.compile(
    r"区|县|景区|景点|古城|公园|广场|山|湖|寺|庙|步行街|地铁|高铁|火车|机场|"
    r"希尔顿|万豪|洲际|如家|全季|亚朵|汉庭|民宿"
)
_SUBJECTIVE_HINT = re.compile(r"安静|吵闹|性价比|舒适|干净|卫生|便宜|奢华|亲子|早餐|贴心|便利|方便")


def poi_search_keyword(preferences: Optional[str]) -> str:
    pref = norm_text(preferences)
    if not pref:
        return "酒店"
    if _POI_HINT.search(pref) or _DISTRICT_RE.search(pref):
        return pref if "酒店" in pref else f"{pref} 酒店"
    if _SUBJECTIVE_HINT.search(pref) or "," in pref or "，" in pref:
        return "酒店"
    return pref if "酒店" in pref else f"{pref} 酒店"


def _valid_hotel_poi(h: Dict[str, Any]) -> bool:
    name = (h.get("name") or "").strip()
    if not name:
        return False
    if name.endswith("市") and not h.get("address") and not h.get("district"):
        return False
    return True


async def recommend_hotel_impl(
    city: str,
    preferences: Optional[str] = None,
    budget_cny_per_night_max: Optional[int] = None,
) -> Dict[str, Any]:
    """查询酒店候选列表，返回 dict（与 LangChain @tool 返回值一致）。"""
    ok, err = require_non_empty(city, "city")
    if not ok:
        return {"error": err}

    pref = norm_text(preferences)
    keyword = poi_search_keyword(pref)

    hotels: List[Dict[str, Any]] = []
    source = "none"
    try:
        res = await fetch_hotels_from_api(city, limit=10, keyword=keyword)
        if not res.get("error"):
            source = res.get("data_source", "api")
            for h in res.get("hotels") or []:
                if not isinstance(h, dict):
                    continue
                item = {k: v for k, v in h.items() if k != "raw"}
                if not _valid_hotel_poi(item):
                    continue
                if budget_cny_per_night_max:
                    price = item.get("avg_price_cny")
                    if price and price > budget_cny_per_night_max:
                        continue
                hotels.append(item)
    except Exception as exc:
        return {"error": f"hotel_query_failed: {exc}"}

    return {
        "city": city,
        "search_query": keyword,
        "preferences": pref or None,
        "budget_cny_per_night_max": int(budget_cny_per_night_max) if budget_cny_per_night_max else None,
        "hotels": hotels,
        "count": len(hotels),
        "data_source": source,
        "note": "请根据用户诉求从 hotels 中挑选一家最合适的酒店向用户说明。",
    }
