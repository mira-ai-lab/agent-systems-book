"""Hotel search and result normalization for Chapter 5.

This module builds the hotel-domain layer on top of raw map POI searches. It
owns search keyword construction, Baidu and Amap result conversion, and hotel
candidate queries.

:author: zhanghong, liuhui
:date: 2026-11-15
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field

from map_api import amap_place_text_search, baidu_place_v2_search


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


def require_non_empty(value: Optional[str], field: str) -> Tuple[bool, str]:
    """Return whether a required text field is non-empty."""
    if (value or "").strip():
        return True, ""
    return False, f"{field} is required."


def _normalize_baidu_hotel(item: Dict[str, Any]) -> Hotel:
    """Convert a Baidu Place API result into a unified ``Hotel`` model."""
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
    return Hotel(
        name=item.get("name") or "",
        district=item.get("area") or item.get("city") or item.get("address"),
        address=item.get("address"),
        tel=item.get("telephone"),
        location=location_text,
        rating=float(rating) if rating is not None else None,
        avg_price_cny=float(avg_price) if avg_price is not None else None,
        type=(
            detail.get("type")
            or item.get("tag")
            or detail.get("tag")
            or item.get("type")
        ),
    )


def _normalize_amap_hotel(poi: Dict[str, Any]) -> Hotel:
    """Convert an Amap POI result into a unified ``Hotel`` model."""
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
    return Hotel(
        name=poi.get("name") or "",
        district=poi.get("adname") or poi.get("address"),
        address=poi.get("address"),
        tel=poi.get("tel"),
        location=poi.get("location"),
        rating=float(rating) if rating is not None else None,
        avg_price_cny=float(avg_price) if avg_price is not None else None,
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
                _normalize_baidu_hotel(item)
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
        _normalize_amap_hotel(poi)
        for poi in (amap_result.get("pois") or [])[:page_size]
        if isinstance(poi, dict)
    ]
    return HotelSearchResult(
        hotels=hotels,
        data_source="amap_place_text",
        search_query=query,
    )
