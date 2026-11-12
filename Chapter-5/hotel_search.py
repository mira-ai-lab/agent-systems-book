"""Chapter-5 酒店搜索与结果标准化。

更新时间：2026-11-13

本模块在地图 POI 搜索之上提供酒店领域能力，包括搜索词构造、百度/高德结果
转换和酒店候选查询。
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

from map_api import amap_place_text_search, baidu_place_v2_search, norm_text


def require_non_empty(value: Optional[str], field: str) -> Tuple[bool, str]:
    """校验必填文本字段，并返回校验状态和错误信息。"""
    if norm_text(value):
        return True, ""
    return False, f"{field} is required."


def _normalize_baidu_hotel(item: Dict[str, Any]) -> Dict[str, Any]:
    """将百度 Place API 结果转换为统一酒店字段。"""
    location = item.get("location") or {}
    latitude = location.get("lat")
    longitude = location.get("lng")
    detail = item.get("detail_info") or {}

    rating_value = detail.get("overall_rating")
    try:
        rating = (
            float(rating_value)
            if rating_value is not None and str(rating_value).strip()
            else None
        )
    except (TypeError, ValueError):
        rating = None

    price = detail.get("price")
    try:
        avg_price = (
            int(float(str(price).replace("￥", "").replace("元", "").strip()))
            if price is not None and str(price).strip()
            else None
        )
    except (TypeError, ValueError):
        avg_price = None

    location_text = None
    if longitude is not None and latitude is not None:
        location_text = f"{longitude},{latitude}"
    return {
        "name": item.get("name"),
        "district": item.get("area") or item.get("city") or item.get("address"),
        "address": item.get("address"),
        "tel": item.get("telephone"),
        "location": location_text,
        "rating": rating,
        "avg_price_cny": avg_price,
        "type": (
            detail.get("type")
            or item.get("tag")
            or detail.get("tag")
            or item.get("type")
        ),
        "raw": item,
    }


def _normalize_amap_hotel(poi: Dict[str, Any]) -> Dict[str, Any]:
    """将高德 POI 结果转换为统一酒店字段。"""
    business = poi.get("business") or {}
    rating_value = business.get("rating")
    cost_value = business.get("cost")
    try:
        rating = (
            float(rating_value)
            if rating_value is not None and str(rating_value).strip()
            else None
        )
    except (TypeError, ValueError):
        rating = None
    try:
        avg_price = (
            int(float(cost_value))
            if cost_value is not None and str(cost_value).strip()
            else None
        )
    except (TypeError, ValueError):
        avg_price = None
    return {
        "name": poi.get("name"),
        "district": poi.get("adname") or poi.get("address"),
        "address": poi.get("address"),
        "tel": poi.get("tel"),
        "location": poi.get("location"),
        "rating": rating,
        "avg_price_cny": avg_price,
        "type": poi.get("type"),
    }


async def search_hotels(
    city: str,
    *,
    limit: int = 10,
    keyword: Optional[str] = None,
) -> Dict[str, Any]:
    """查询指定城市的酒店候选列表。

    优先使用已配置的百度 Place API，失败或未配置时回退到高德 POI 搜索。
    ``keyword`` 可包含区县、景点、地标或酒店品牌。
    """
    query = norm_text(keyword) or "酒店"
    if "酒店" not in query:
        query = f"{query} 酒店"
    page_size = max(1, min(int(limit or 10), 20))

    if norm_text(os.getenv("BAIDU_MAP_AK")):
        baidu_result = await baidu_place_v2_search(
            query=query,
            region=city,
            scope=2,
            page_size=page_size,
            filter_="industry_type:hotel|sort_name:total_score|sort_rule:0",
        )
        if not baidu_result.get("error"):
            hotels = [
                _normalize_baidu_hotel(item)
                for item in (baidu_result.get("results") or [])[:page_size]
                if isinstance(item, dict)
            ]
            return {
                "hotels": hotels,
                "data_source": "baidu_place_v2",
                "search_query": query,
            }

    amap_result = await amap_place_text_search(
        region=city,
        keyword=query,
        limit=limit,
    )
    if amap_result.get("error"):
        return amap_result
    hotels = [
        _normalize_amap_hotel(poi)
        for poi in (amap_result.get("pois") or [])[:page_size]
        if isinstance(poi, dict)
    ]
    return {
        "hotels": hotels,
        "data_source": "amap_place_text",
        "search_query": query,
    }
