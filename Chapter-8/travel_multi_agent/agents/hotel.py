"""HotelAgent — 酒店推荐。"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from travel_multi_agent.agents.base import build_agent
from travel_multi_agent.infra.travel_api import fetch_hotels_from_api, norm_text, require_non_empty

_DISTRICT_RE = re.compile(r"[\u4e00-\u9fff]{1,12}(?:区|县)")
_POI_HINT = re.compile(
    r"区|县|景区|景点|古城|公园|广场|山|湖|寺|庙|步行街|地铁|高铁|火车|机场|"
    r"希尔顿|万豪|洲际|如家|全季|亚朵|汉庭|民宿"
)
_SUBJECTIVE_HINT = re.compile(r"安静|吵闹|性价比|舒适|干净|卫生|便宜|奢华|亲子|早餐|贴心|便利|方便")

SYSTEM_PROMPT = """你是酒店推荐助手，只能通过工具 recommend_hotel 查酒店。

规则：
1. city 必填；有预算时传入 budget_cny_per_night_max。
2. preferences 只填「地图能搜的关键词」：区名、景点、地标、酒店品牌（如 西湖、平城区、希尔顿）。
   不要填主观感受（安静、舒适、性价比）——这些留给你读 hotels 后再判断。
3. 工具返回 hotels 后，结合用户全部诉求，推荐 3–5 家最合适的酒店（按匹配度排序）。
   每家需包含：名称、地址/位置、参考价格、评分（如有）、1 句推荐理由。
   若用户明确要求「只推荐一家」，则只输出 1 家。
4. 若 hotels 为空或不足 3 家，如实说明并给出已有选项或调整建议（放宽预算/扩大范围）。
5. 非酒店问题，回复：我只能协助酒店推荐。
"""


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


@tool
async def recommend_hotel(
    city: str,
    preferences: Optional[str] = None,
    budget_cny_per_night_max: Optional[int] = None,
) -> Dict[str, Any]:
    """查酒店列表。preferences 传区名/景点/品牌；安静等主观词由模型读 hotels 后判断。"""
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
                if isinstance(h, dict):
                    item = {k: v for k, v in h.items() if k != "raw"}
                    if _valid_hotel_poi(item):
                        if budget_cny_per_night_max:
                            price = h.get("avg_price_cny")
                            if price and price > budget_cny_per_night_max:
                                continue
                        hotels.append(item)
    except Exception as e:
        return {"error": f"hotel_query_failed: {str(e)}"}

    return {
        "city": city,
        "search_query": keyword,
        "preferences": pref or None,
        "budget_cny_per_night_max": budget_cny_per_night_max,
        "hotels": hotels,
        "count": len(hotels),
        "data_source": source,
        "note": "请结合 preferences 与 hotels 推荐 3–5 家最合适的，并说明各自特点。",
    }


def create_hotel_agent() -> Any:
    return build_agent([recommend_hotel], SYSTEM_PROMPT)
