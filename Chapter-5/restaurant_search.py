"""Restaurant search and normalization for map POI providers.

This module builds the restaurant-domain layer on top of the generic Baidu and
Amap adapters. It owns cuisine keyword construction, provider normalization,
and restaurant candidate validation.

:author: liuhui
:date: 2026-11-16
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from map_api import (
    amap_place_text_search,
    baidu_place_v2_search,
    normalize_amap_poi,
    normalize_baidu_poi,
)


class Restaurant(BaseModel):
    """Unified restaurant fields returned by any map provider."""

    name: str = Field(..., min_length=1, description="Restaurant name")
    cuisine: str | None = None
    avg_price_cny: float | None = None
    rating: float | None = None
    district: str | None = None
    address: str | None = None
    tel: str | None = None
    location: str | None = None
    type: str | None = None


class RestaurantSearchResult(BaseModel):
    """Normalized restaurant search response returned to upper layers."""

    restaurants: list[Restaurant]
    data_source: str
    search_query: str | None = None
    error: str | None = None
    message: str | None = None


_CITY_CUISINE_MAP: Dict[str, str] = {
    "上海": "本帮菜",
    "苏州": "苏帮菜",
    "杭州": "杭帮菜",
}

_CITY_CORE_DISTRICT: Dict[str, str] = {
    "上海": "黄浦 静安",
    "苏州": "姑苏 平江路",
    "杭州": "西湖 上城",
}

_REJECT_NAMES = frozenset({"上海市", "苏州市", "杭州市", "南宁市"})
_REJECT_KEYWORDS = (
    "棋牌",
    "足浴",
    "KTV",
    "洗浴",
    "养生会所",
    "休闲会所",
    "游戏场所",
)


def normalize_restaurant_cuisine(
    location: str,
    cuisine: Optional[str],
) -> str:
    """Map broad local-cuisine wording to a map-searchable cuisine keyword.

    Args:
        location: Target city or district.
        cuisine: Optional user cuisine preference.

    Returns:
        A cuisine keyword suitable for a map POI query.
    """
    normalized_location = (location or "").strip()
    normalized_cuisine = (cuisine or "").strip()
    if any(
        keyword in normalized_cuisine
        for keyword in ("江南", "时令", "本地", "江浙", "特色")
    ):
        return _CITY_CUISINE_MAP.get(normalized_location, "江浙菜")
    if normalized_cuisine:
        return normalized_cuisine
    return _CITY_CUISINE_MAP.get(normalized_location, "餐厅")


def restaurant_search_keyword(location: str, cuisine: Optional[str]) -> str:
    """Build the Baidu/Amap restaurant search keyword.

    Args:
        location: Target city or district.
        cuisine: Optional user cuisine preference.

    Returns:
        A query combining a core district, cuisine, and ``餐厅``.
    """
    normalized_location = (location or "").strip()
    core_district = _CITY_CORE_DISTRICT.get(
        normalized_location,
        normalized_location,
    )
    cuisine_keyword = normalize_restaurant_cuisine(location, cuisine)
    return f"{core_district} {cuisine_keyword} 餐厅".strip()


def is_valid_restaurant_poi(poi: Dict[str, Any]) -> bool:
    """Reject administrative-area and non-restaurant POI candidates.

    Args:
        poi: A normalized restaurant-shaped POI dictionary.

    Returns:
        ``True`` when the candidate is suitable for restaurant recommendations.
    """
    name = str(poi.get("name") or "").strip()
    if not name or len(name) < 2:
        return False
    if name in _REJECT_NAMES:
        return False
    if name.endswith("市") and len(name) <= 5:
        return False
    if str(poi.get("type") or "").strip().lower() == "life":
        return False

    raw = poi.get("raw") or {}
    if isinstance(raw, dict):
        detail = raw.get("detail_info") or {}
        tag = str(
            detail.get("classified_poi_tag") or detail.get("tag") or ""
        ).strip()
        if any(keyword in name or keyword in tag for keyword in _REJECT_KEYWORDS):
            return False
    return True


def _to_restaurant(poi: Dict[str, Any]) -> Restaurant:
    """Convert one normalized POI dictionary into a ``Restaurant`` model."""
    return Restaurant(
        name=str(poi.get("name") or "").strip(),
        cuisine=poi.get("cuisine") or poi.get("type"),
        avg_price_cny=poi.get("avg_price_cny"),
        rating=poi.get("rating"),
        district=poi.get("district"),
        address=poi.get("address"),
        tel=poi.get("tel"),
        location=poi.get("location"),
        type=poi.get("type"),
    )


async def search_restaurants(
    location: str,
    *,
    cuisine: Optional[str] = None,
    limit: int = 10,
) -> RestaurantSearchResult:
    """Query restaurant candidates through Baidu or Amap.

    Baidu is preferred when both keys are configured; otherwise the configured
    provider is used. Results are normalized and filtered before being returned.

    Args:
        location: Target city or district.
        cuisine: Optional cuisine preference.
        limit: Maximum number of restaurant candidates to return.

    Returns:
        A ``RestaurantSearchResult`` with normalized candidates, data source,
        query text, and optional error details.
    """
    normalized_location = (location or "").strip()
    search_query = restaurant_search_keyword(normalized_location, cuisine)
    page_size = max(1, min(int(limit or 10), 20))
    baidu_key = (os.getenv("BAIDU_MAP_AK") or "").strip()
    amap_key = (os.getenv("AMAP_KEY") or "").strip()

    if not baidu_key and not amap_key:
        return RestaurantSearchResult(
            restaurants=[],
            data_source="none",
            search_query=search_query,
            error="map_key_missing",
            message="Configure AMAP_KEY or BAIDU_MAP_AK in the project .env file.",
        )

    if baidu_key:
        baidu_result = await baidu_place_v2_search(
            query=search_query,
            region=normalized_location,
            scope=2,
            page_size=min(20, page_size * 2),
            filter_="industry_type:cater|sort_name:overall_rating|sort_rule:0",
        )
        if not baidu_result.get("error"):
            restaurants = []
            for item in baidu_result.get("results") or []:
                if not isinstance(item, dict):
                    continue
                poi = normalize_baidu_poi(item)
                if is_valid_restaurant_poi(poi):
                    restaurants.append(_to_restaurant(poi))
                if len(restaurants) >= page_size:
                    break
            if restaurants or not amap_key:
                return RestaurantSearchResult(
                    restaurants=restaurants,
                    data_source="baidu_place_v2",
                    search_query=search_query,
                    error=None if restaurants else "no_valid_restaurants",
                    message=(
                        None
                        if restaurants
                        else f"未找到有效餐饮 POI（query={search_query}）"
                    ),
                )
        elif not amap_key:
            return RestaurantSearchResult(
                restaurants=[],
                data_source="baidu_place_v2",
                search_query=search_query,
                error=baidu_result.get("error"),
                message=baidu_result.get("message"),
            )

    amap_result = await amap_place_text_search(
        region=normalized_location,
        keyword=search_query,
        limit=min(20, page_size * 2),
    )
    if amap_result.get("error"):
        return RestaurantSearchResult(
            restaurants=[],
            data_source="amap_place_text",
            search_query=search_query,
            error=amap_result.get("error"),
            message=amap_result.get("message"),
        )
    restaurants = [
        _to_restaurant(normalize_amap_poi(poi))
        for poi in (amap_result.get("pois") or [])
        if isinstance(poi, dict)
        and is_valid_restaurant_poi(normalize_amap_poi(poi))
    ][:page_size]
    return RestaurantSearchResult(
        restaurants=restaurants,
        data_source="amap_place_text",
        search_query=search_query,
        error=None if restaurants else "no_valid_restaurants",
        message=None if restaurants else "未找到有效餐饮 POI",
    )
