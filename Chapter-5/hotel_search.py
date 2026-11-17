"""Hotel search and result normalization for Chapter 5.

This module builds the hotel-domain layer on top of raw map POI searches and
generic POI normalization. It owns search keyword construction, hotel model
adaptation, and candidate queries.

:author: zhanghong, liuhui
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


class Hotel(BaseModel):
    """Unified hotel fields returned by any map provider."""

    name: str = Field(..., min_length=1, description="Hotel name")
    district: str | None = None
    address: str | None = None
    tel: str | None = None
    location: str | None = None
    rating: float | None = None
    avg_price_cny: float | None = None
    type: str | None = None


class HotelSearchResult(BaseModel):
    """Normalized hotel search response returned to upper layers."""

    hotels: list[Hotel]
    data_source: str
    search_query: str | None = None
    error: str | None = None
    message: str | None = None


_HOTEL_REJECT_KEYWORDS = (
    "游泳",
    "培训",
    "大楼",
    "棋牌",
    "足浴",
    "美容",
    "美发",
    "小区",
    "公寓",
)


def is_valid_hotel_poi(poi: Dict[str, Any]) -> bool:
    """Return whether a normalized candidate is a usable hotel POI.

    Args:
        poi: Hotel-shaped dictionary returned by a map provider.

    Returns:
        ``True`` when the candidate has a plausible name and location.
    """
    name = (poi.get("name") or "").strip()
    if len(name) < 2:
        return False
    if any(keyword in name for keyword in _HOTEL_REJECT_KEYWORDS):
        return False
    if name.endswith("市") and not (poi.get("address") or poi.get("district")):
        return False
    return bool(poi.get("address") or poi.get("district") or poi.get("location"))


def _to_hotel(poi: Dict[str, Any]) -> Hotel:
    """Convert a generic normalized POI into a ``Hotel`` model."""
    return Hotel(
        name=poi.get("name") or "",
        district=poi.get("district"),
        address=poi.get("address"),
        tel=poi.get("tel"),
        location=poi.get("location"),
        rating=poi.get("rating"),
        avg_price_cny=poi.get("avg_price_cny"),
        type=poi.get("type"),
    )


async def search_hotels(
    city: str,
    *,
    limit: int = 10,
    keyword: Optional[str] = None,
) -> HotelSearchResult:
    """Query a hotel candidate list for a target city.

    The provider is selected from the configured map keys. Baidu is preferred
    when both keys are present; otherwise the single configured provider is
    used. If neither ``AMAP_KEY`` nor ``BAIDU_MAP_AK`` is configured, an error
    is returned. The ``keyword`` argument should be a map-searchable district,
    landmark, street, or hotel brand extracted by the agent.

    Args:
        city: Target city or region.
        limit: Maximum number of candidates to return.
        keyword: Optional map-searchable keyword for the hotel query.

    Returns:
        A ``HotelSearchResult`` containing normalized hotels, the active data
        source, and an optional error description.
    """
    query = (keyword or "").strip() or "酒店"
    if "酒店" not in query:
        query = f"{query} 酒店"
    page_size = max(1, min(int(limit or 10), 20))

    baidu_key = (os.getenv("BAIDU_MAP_AK") or "").strip()
    amap_key = (os.getenv("AMAP_KEY") or "").strip()

    if not baidu_key and not amap_key:
        return HotelSearchResult(
            hotels=[],
            data_source="none",
            search_query=query,
            error="map_key_missing",
            message="Configure AMAP_KEY or BAIDU_MAP_AK in the project .env file.",
        )

    if baidu_key:
        baidu_result = await baidu_place_v2_search(
            query=query,
            region=city,
            scope=2,
            page_size=page_size,
            filter_="industry_type:hotel|sort_name:total_score|sort_rule:0",
        )
        if not baidu_result.get("error"):
            hotels = [
                _to_hotel(normalize_baidu_poi(item))
                for item in (baidu_result.get("results") or [])[:page_size]
                if isinstance(item, dict)
            ]
            return HotelSearchResult(
                hotels=hotels,
                data_source="baidu_place_v2",
                search_query=query,
            )
        if not amap_key:
            return HotelSearchResult(
                hotels=[],
                data_source="baidu_place_v2",
                search_query=query,
                error=baidu_result.get("error"),
                message=baidu_result.get("message"),
            )

    amap_result = await amap_place_text_search(
        region=city,
        keyword=query,
        limit=limit,
    )
    if amap_result.get("error"):
        return HotelSearchResult(
            hotels=[],
            data_source="amap_place_text",
            search_query=query,
            error=amap_result.get("error"),
            message=amap_result.get("message"),
        )
    hotels = [
        _to_hotel(normalize_amap_poi(poi))
        for poi in (amap_result.get("pois") or [])[:page_size]
        if isinstance(poi, dict)
    ]
    return HotelSearchResult(
        hotels=hotels,
        data_source="amap_place_text",
        search_query=query,
    )
