"""Map service API adapters for Chapter 5.

This module owns raw Baidu and Amap POI requests, Baidu request signing, and
parsing of raw responses. It does not contain hotel search keywords, hotel
recommendation, or agent logic.

:author: zhanghong, liuhui
:date: 2026-11-18
"""

from __future__ import annotations

import hashlib
import os
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
from dotenv import load_dotenv


_HTTP_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_ENV_FILE)


def normalize_city_name(city: Optional[str]) -> str:
    """Remove a trailing ``市`` so map regions use the short city name."""
    normalized = (city or "").strip()
    if normalized.endswith("市") and len(normalized) > 1:
        return normalized[:-1]
    return normalized


def normalize_baidu_poi(item: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one Baidu Place result into a common POI dictionary.

    Args:
        item: A raw Baidu Place API result.

    Returns:
        A dictionary with common name, address, coordinate, rating, and price
        fields while retaining the original provider payload under ``raw``.
    """
    location = item.get("location") or {}
    latitude = location.get("lat")
    longitude = location.get("lng")
    detail = item.get("detail_info") or {}

    raw_rating = detail.get("overall_rating")
    try:
        rating = (
            float(raw_rating)
            if raw_rating is not None and str(raw_rating).strip()
            else None
        )
    except (TypeError, ValueError):
        rating = None

    raw_price = detail.get("price")
    try:
        avg_price = (
            int(float(str(raw_price).replace("￥", "").replace("元", "").strip()))
            if raw_price is not None and str(raw_price).strip()
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


def normalize_amap_poi(poi: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one Amap POI into a common POI dictionary.

    Args:
        poi: A raw Amap POI dictionary.

    Returns:
        A dictionary with common location, rating, price, type, and cuisine
        fields. ``cuisine`` mirrors the provider type for restaurant use.
    """
    business = poi.get("business") or {}
    raw_rating = business.get("rating")
    raw_cost = business.get("cost")
    try:
        rating = (
            float(raw_rating)
            if raw_rating is not None and str(raw_rating).strip()
            else None
        )
    except (TypeError, ValueError):
        rating = None
    try:
        avg_price = (
            int(float(raw_cost))
            if raw_cost is not None and str(raw_cost).strip()
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
        "cuisine": poi.get("type"),
    }


async def _http_get_json(
    url: str,
    params: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Send a JSON GET request and always return a dictionary.

    Args:
        url: Target endpoint URL.
        params: Query parameters to append to the request.
        headers: Optional request headers.

    Returns:
        The parsed JSON payload as a dictionary. A non-dictionary payload is
        wrapped under the ``data`` key.
    """
    async with httpx.AsyncClient(
        timeout=_HTTP_TIMEOUT,
        follow_redirects=True,
    ) as client:
        response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            return data
        return {"data": data}


async def amap_place_text_search(
    region: str,
    keyword: str,
    *,
    limit: int = 10,
    types: Optional[str] = None,
) -> Dict[str, Any]:
    """Search for places in a region with the Amap POI text API.

    Called endpoint: Amap Web Service API "Search POI 2.0" text search
    (``https://restapi.amap.com/v5/place/text``).

    The ``AMAP_KEY`` environment variable is required. Raw POI entries are
    returned without further processing; failures are returned through the
    ``error`` field.

    Docs: https://lbs.amap.com/api/webservice/guide/api-advanced/newpoisearch
    Key: create a Web-service key at https://console.amap.com/dev/key/app

    Args:
        region: Target region or city name.
        keyword: POI keyword to search for.
        limit: Maximum number of results to request.
        types: Optional Amap POI type filter.

    Returns:
        A dictionary containing ``pois`` and ``raw`` on success, or ``error``
        when the request cannot be completed.
    """
    key = (os.getenv("AMAP_KEY") or "").strip()
    if not key:
        return {"error": "AMAP_KEY not set"}

    normalized_region = (region or "").strip()
    normalized_keyword = (keyword or "").strip()
    if not normalized_region:
        return {"error": "region is required"}
    if not normalized_keyword and not (types or "").strip():
        return {"error": "keyword or types is required"}

    params: Dict[str, Any] = {
        "key": key,                     # Web-service key created in the Amap console (required).
        "keywords": normalized_keyword, # Single keyword; text length must not exceed 80 characters.
        "region": normalized_region,    # citycode, adcode, or cityname; adds recall weight.
        "city_limit": "true",           # When true, only POIs inside the given region are returned.
        "page_size": str(max(1, min(int(limit or 10), 25))),  # Number of results per page; Amap accepts values from 1 to 25.
        "page_num": "1",                # One-based page number; the default and first page is 1.
        "output": "JSON",               # Response format; the current Amap API supports JSON only.
        "show_fields": "business",      # Response field group filter; business adds rating/cost when present.
    }
    if types:
        params["types"] = types
    data = await _http_get_json(
        "https://restapi.amap.com/v5/place/text",
        params=params,
    )
    pois = data.get("pois")
    if not isinstance(pois, list):
        return {"error": "invalid_poi_response", "raw": data}
    return {"pois": pois, "raw": data}


def _baidu_sn(
    path: str,
    params_in_order: Sequence[Tuple[str, str]],
    sk: str,
) -> str:
    """Compute the Baidu Maps WebService API SN signature.

    Args:
        path: Request path, such as ``/place/v2/search``.
        params_in_order: Query parameters in the required order.
        sk: Baidu Maps security key.

    Returns:
        The lowercase hexadecimal MD5 signature.
    """
    query = urllib.parse.urlencode(
        params_in_order,
        safe="|,:",
        quote_via=urllib.parse.quote,
    )
    whole = f"{path}?{query}{sk}"
    return hashlib.md5(
        urllib.parse.quote(whole, safe="").encode("utf-8")
    ).hexdigest()


async def baidu_place_v2_search(
    *,
    query: str,
    region: str,
    tag: Optional[str] = None,
    scope: int = 2,
    page_size: int = 10,
    page_num: int = 0,
    filter_: Optional[str] = None,
) -> Dict[str, Any]:
    """Search for places with the Baidu Maps Place API v2.

    Called endpoint: Baidu Map WebService API Place API v2 region search,
    "地点检索2.0（原检索）" (``https://api.map.baidu.com/place/v2/search``).

    The ``BAIDU_MAP_AK`` environment variable is required. When
    ``BAIDU_MAP_SK`` is configured, requests are signed with an SN signature.
    Otherwise the server-side AK is expected to use IP whitelist verification.
    Errors are returned through the ``error`` field.

    Docs:
    https://lbsyun.baidu.com/docs/webapi?title=placev3/guide/webservice-placeapiV3/interfaceDocumentV2
    Key: create a server-side AK at https://lbsyun.baidu.com/apiconsole/key

    Args:
        query: POI keyword to search for.
        region: Target region or city name.
        tag: Optional Baidu tag filter.
        scope: Detail scope requested from Baidu.
        page_size: Maximum number of results per page.
        page_num: Zero-based page number.
        filter_: Optional Baidu filter expression.

    Returns:
        A dictionary containing ``results`` and ``raw`` on success, or
        ``error`` when the request cannot be completed.
    """
    ak = (os.getenv("BAIDU_MAP_AK") or "").strip()
    if not ak:
        return {"error": "BAIDU_MAP_AK not set"}
    sk = (os.getenv("BAIDU_MAP_SK") or "").strip()

    params_in_order: List[Tuple[str, str]] = [
        ("query", (query or "").strip()),    # Single keyword; max 45 characters.
        ("region", (region or "").strip()),  # Administrative region name or cityCode; adds recall weight.
        ("output", "json"),                  # Response format: json or xml.
        ("scope", str(int(scope or 2))),     # 1 basic info; 2 detailed POI info.
        ("page_size", str(max(1, min(int(page_size or 10), 20)))),  # Default 10, max 20.
        ("page_num", str(max(0, int(page_num or 0)))),        # Zero-based page number.
        ("ak", ak),                          # Server-side developer access key.
    ]
    if tag:
        params_in_order.insert(1, ("tag", (tag or "").strip()))         # Optional category preference.
    if filter_:
        params_in_order.insert(5, ("filter", (filter_ or "").strip()))  # Industry/sort filter for scope=2.
    if sk:
        params_in_order.append(("timestamp", str(int(time.time()))))
        params_in_order.append(
            ("sn", _baidu_sn("/place/v2/search", params_in_order, sk))
        )

    async with httpx.AsyncClient(
        timeout=_HTTP_TIMEOUT,
        follow_redirects=True,
    ) as client:
        response = await client.get(
            "https://api.map.baidu.com/place/v2/search",
            params=params_in_order,
        )
        response.raise_for_status()
        data = response.json()

    if not isinstance(data, dict):
        return {"error": "invalid_baidu_response", "raw": data}
    if str(data.get("status")) not in ("0", "OK", "ok"):
        return {
            "error": "baidu_error",
            "raw": data,
            "message": data.get("message"),
        }
    results = data.get("results")
    if not isinstance(results, list):
        return {"error": "invalid_baidu_results", "raw": data}
    return {"results": results, "raw": data}
