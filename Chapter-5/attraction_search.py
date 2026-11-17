"""Attraction and classic POI search for map providers.

This module owns common scenic-POI query construction, Baidu and Amap result
normalization, and attraction candidate validation. Chapter 6 consumes the
normalized result without maintaining map-provider details.

:author: liuhui
:date: 2026-11-17
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


class Attraction(BaseModel):
    """Unified attraction and classic POI fields returned by a map provider."""

    name: str = Field(..., min_length=1, description="Attraction or POI name")
    district: str | None = None
    address: str | None = None
    tel: str | None = None
    location: str | None = None
    rating: float | None = None
    avg_price_cny: float | None = None
    type: str | None = None


class AttractionSearchResult(BaseModel):
    """Normalized attraction search response returned to upper layers."""

    attractions: list[Attraction]
    data_source: str
    search_query: str | None = None
    error: str | None = None
    message: str | None = None


_POSITIVE_KEYWORDS = (
    "景区",
    "景点",
    "公园",
    "园林",
    "博物馆",
    "美术馆",
    "展览馆",
    "纪念馆",
    "故居",
    "古镇",
    "古城",
    "古街",
    "寺",
    "庙",
    "祠",
    "塔",
    "湖",
    "山",
    "湿地",
)
_REJECT_KEYWORDS = (
    "酒店",
    "宾馆",
    "民宿",
    "小区",
    "公寓",
    "写字楼",
    "公司",
    "培训",
    "医院",
    "学校",
    "商场",
)
_ADMIN_SUFFIXES = (
    "特别行政区",
    "自治州",
    "地区",
    "盟",
    "市",
    "省",
    "县",
    "区",
)


def normalize_attraction_query(preferences: Optional[str]) -> str:
    """Convert broad user preferences into a map-searchable POI query.

    Args:
        preferences: Optional natural-language attraction preference.

    Returns:
        A concise Chinese POI query suitable for Baidu or Amap.
    """
    preference = (preferences or "").strip()
    if not preference:
        return "景点"
    if any(
        keyword in preference
        for keyword in ("历史", "文化", "古迹", "博物馆", "园林", "人文")
    ):
        return "历史文化 景点"
    if any(
        keyword in preference
        for keyword in ("自然", "风光", "湿地", "湖", "山", "徒步")
    ):
        return "自然风光 景点"
    return f"{preference} 景点"


def is_valid_attraction_poi(poi: Dict[str, Any]) -> bool:
    """Return whether a candidate looks like an attraction or scenic POI.

    Args:
        poi: A normalized attraction-shaped POI dictionary.

    Returns:
        ``True`` when the candidate has a usable location and scenic semantics.
    """
    name = str(poi.get("name") or "").strip()
    if not name or len(name) < 2:
        return False
    for suffix in _ADMIN_SUFFIXES:
        if name.endswith(suffix):
            stem = name[: -len(suffix)]
            if 1 <= len(stem) <= 6:
                return False
    if not (
        str(poi.get("address") or "").strip()
        or str(poi.get("district") or "").strip()
        or str(poi.get("location") or "").strip()
    ):
        return False
    if any(keyword in name for keyword in _REJECT_KEYWORDS):
        return False

    poi_type = str(poi.get("type") or "").strip().lower()
    raw = poi.get("raw") or {}
    detail: Dict[str, Any] = {}
    if isinstance(raw, dict):
        detail = raw.get("detail_info") or {}
    tag = str(
        detail.get("classified_poi_tag") or detail.get("tag") or ""
    ).strip()
    text = f"{name} {poi_type} {tag}"
    if any(keyword in text for keyword in _REJECT_KEYWORDS):
        return False
    if tag:
        return any(keyword in tag for keyword in _POSITIVE_KEYWORDS)
    return any(keyword in text for keyword in _POSITIVE_KEYWORDS)


def _to_attraction(poi: Dict[str, Any]) -> Attraction:
    """Convert one normalized POI dictionary into an ``Attraction`` model."""
    return Attraction(
        name=str(poi.get("name") or "").strip(),
        district=poi.get("district"),
        address=poi.get("address"),
        tel=poi.get("tel"),
        location=poi.get("location"),
        rating=poi.get("rating"),
        avg_price_cny=poi.get("avg_price_cny"),
        type=poi.get("type"),
    )


async def search_attractions(
    city: str,
    *,
    preferences: Optional[str] = None,
    limit: int = 10,
) -> AttractionSearchResult:
    """Query attraction candidates through Baidu or Amap.

    Baidu is preferred when both keys are configured; otherwise the configured
    provider is used. Provider payloads are normalized and filtered before
    being returned.

    Args:
        city: Target city used to constrain the POI search.
        preferences: Optional attraction preference used to build the query.
        limit: Maximum number of attraction candidates to return.

    Returns:
        An ``AttractionSearchResult`` with normalized candidates, data source,
        query text, and optional error details.
    """
    normalized_city = (city or "").strip()
    search_query = normalize_attraction_query(preferences)
    page_size = max(1, min(int(limit or 10), 20))
    baidu_key = (os.getenv("BAIDU_MAP_AK") or "").strip()
    amap_key = (os.getenv("AMAP_KEY") or "").strip()

    if not baidu_key and not amap_key:
        return AttractionSearchResult(
            attractions=[],
            data_source="none",
            search_query=search_query,
            error="map_key_missing",
            message="Configure AMAP_KEY or BAIDU_MAP_AK in the project .env file.",
        )

    if baidu_key:
        baidu_result = await baidu_place_v2_search(
            query=search_query,
            region=normalized_city,
            scope=2,
            page_size=min(20, page_size * 3),
            filter_="industry_type:life|sort_name:overall_rating|sort_rule:0",
        )
        if not baidu_result.get("error"):
            attractions = []
            for item in baidu_result.get("results") or []:
                if not isinstance(item, dict):
                    continue
                poi = normalize_baidu_poi(item)
                if is_valid_attraction_poi(poi):
                    attractions.append(_to_attraction(poi))
                if len(attractions) >= page_size:
                    break
            if attractions or not amap_key:
                return AttractionSearchResult(
                    attractions=attractions,
                    data_source="baidu_place_v2",
                    search_query=search_query,
                    error=None if attractions else "no_valid_attractions",
                    message=(
                        None
                        if attractions
                        else f"未找到有效景点 POI（query={search_query}）"
                    ),
                )
        elif not amap_key:
            return AttractionSearchResult(
                attractions=[],
                data_source="baidu_place_v2",
                search_query=search_query,
                error=baidu_result.get("error"),
                message=baidu_result.get("message"),
            )

    amap_result = await amap_place_text_search(
        region=normalized_city,
        keyword=search_query,
        limit=min(20, page_size * 3),
    )
    if amap_result.get("error"):
        return AttractionSearchResult(
            attractions=[],
            data_source="amap_place_text",
            search_query=search_query,
            error=amap_result.get("error"),
            message=amap_result.get("message"),
        )
    attractions = []
    for item in amap_result.get("pois") or []:
        if not isinstance(item, dict):
            continue
        poi = normalize_amap_poi(item)
        if is_valid_attraction_poi(poi):
            attractions.append(_to_attraction(poi))
        if len(attractions) >= page_size:
            break
    return AttractionSearchResult(
        attractions=attractions,
        data_source="amap_place_text",
        search_query=search_query,
        error=None if attractions else "no_valid_attractions",
        message=None if attractions else "未找到有效景点 POI",
    )
