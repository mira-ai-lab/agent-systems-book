"""Map service API adapters for Chapter 5.

This module owns raw Baidu and Amap POI requests, Baidu request signing, and
parsing of raw responses. It does not contain hotel search keywords, hotel
recommendation, or agent logic.

:author: zhanghong, liuhui
:date: 2026-11-16
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
from dotenv import load_dotenv


_HTTP_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
_BAIDU_API_LOCK = asyncio.Lock()
_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_ENV_FILE)


def normalize_city_name(city: Optional[str]) -> str:
    """Remove a trailing ``市`` so map regions use the short city name."""
    normalized = (city or "").strip()
    if normalized.endswith("市") and len(normalized) > 1:
        return normalized[:-1]
    return normalized


def require_non_empty(value: Optional[str], field: str) -> Tuple[bool, str]:
    """Return whether a required text field contains a non-empty value."""
    if (value or "").strip():
        return True, ""
    return False, f"{field} is required."


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


def parse_lng_lat(location: Any) -> tuple[float, float] | None:
    """Parse common ``lng,lat`` map coordinates.

    Args:
        location: Coordinate string or dictionary returned by a map provider.

    Returns:
        A ``(longitude, latitude)`` tuple, or ``None`` for invalid input.
    """
    if not location:
        return None
    if isinstance(location, dict):
        longitude = (
            location.get("lng")
            or location.get("lon")
            or location.get("longitude")
        )
        latitude = location.get("lat") or location.get("latitude")
    else:
        parts = str(location).split(",")
        if len(parts) != 2:
            return None
        longitude, latitude = parts
    if longitude is None or latitude is None:
        return None
    try:
        return float(str(longitude).strip()), float(str(latitude).strip())
    except (TypeError, ValueError):
        return None


def haversine_m(
    origin: tuple[float, float],
    destination: tuple[float, float],
) -> int:
    """Estimate straight-line distance in metres between two coordinates."""
    longitude_1, latitude_1 = origin
    longitude_2, latitude_2 = destination
    radius_m = 6_371_000
    phi_1 = math.radians(latitude_1)
    phi_2 = math.radians(latitude_2)
    delta_phi = math.radians(latitude_2 - latitude_1)
    delta_lambda = math.radians(longitude_2 - longitude_1)
    haversine = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi_1)
        * math.cos(phi_2)
        * math.sin(delta_lambda / 2) ** 2
    )
    return int(
        2
        * radius_m
        * math.atan2(math.sqrt(haversine), math.sqrt(1 - haversine))
    )


def select_baidu_direction_mode(distance_m: int) -> str:
    """Choose a Baidu Direction API mode based on route distance."""
    if distance_m <= 1200:
        return "walking"
    if distance_m <= 5000:
        return "riding"
    if distance_m <= 30000:
        return "transit"
    return "driving"


def _extract_baidu_route_summary(
    data: Dict[str, Any],
    mode: str,
) -> Dict[str, Any]:
    """Extract distance, duration, and step count from a route response."""
    result = data.get("result") if isinstance(data, dict) else None
    if not isinstance(result, dict):
        return {"error": "invalid_baidu_direction_result"}

    routes = result.get("routes")
    if not isinstance(routes, list) or not routes:
        routes = result.get("routes_list") or []
    route = routes[0] if routes and isinstance(routes[0], dict) else {}
    if mode == "transit" and isinstance(route.get("scheme"), list):
        scheme = route["scheme"][0] if route["scheme"] else None
        if isinstance(scheme, dict):
            route = scheme

    raw_distance = route.get("distance") or route.get("distance_m")
    raw_duration = route.get("duration") or route.get("duration_s")
    try:
        distance_m = (
            int(float(raw_distance))
            if raw_distance is not None
            else None
        )
    except (TypeError, ValueError):
        distance_m = None
    try:
        duration_min = (
            int(round(float(raw_duration) / 60))
            if raw_duration is not None
            else None
        )
    except (TypeError, ValueError):
        duration_min = None
    steps = route.get("steps")
    return {
        "distance_m": distance_m,
        "duration_min": duration_min,
        "steps_count": len(steps) if isinstance(steps, list) else None,
    }


async def baidu_direction_route(
    origin: tuple[float, float],
    destination: tuple[float, float],
    *,
    mode: str,
    region: str | None = None,
) -> Dict[str, Any]:
    """Request a route from the Baidu Direction API.

    Args:
        origin: Origin coordinate as ``(longitude, latitude)``.
        destination: Destination coordinate as ``(longitude, latitude)``.
        mode: One of ``driving``, ``walking``, ``riding``, or ``transit``.
        region: Optional city name required by some transit requests.

    Returns:
        A normalized route summary or an error dictionary.
    """
    ak = (os.getenv("BAIDU_MAP_AK") or "").strip()
    if not ak:
        return {"error": "BAIDU_MAP_AK not set"}

    api_mode = (
        mode
        if mode in {"driving", "walking", "riding", "transit"}
        else "transit"
    )
    path = f"/direction/v2/{api_mode}"
    params_in_order: List[Tuple[str, str]] = [
        ("origin", f"{origin[1]},{origin[0]}"),
        ("destination", f"{destination[1]},{destination[0]}"),
        ("ak", ak),
        ("output", "json"),
    ]
    if api_mode == "transit" and region:
        params_in_order.append(("region", normalize_city_name(region)))

    sk = (os.getenv("BAIDU_MAP_SK") or "").strip()
    if sk:
        params_in_order.extend(
            [
                ("timestamp", str(int(time.time()))),
                (
                    "sn",
                    _baidu_sn(path, params_in_order, sk),
                ),
            ]
        )

    try:
        async with _BAIDU_API_LOCK:
            async with httpx.AsyncClient(
                timeout=_HTTP_TIMEOUT,
                follow_redirects=True,
            ) as client:
                response = await client.get(
                    f"https://api.map.baidu.com{path}",
                    params=dict(params_in_order),
                )
                response.raise_for_status()
                data = response.json()
    except Exception as exc:
        return {
            "error": "baidu_direction_request_failed",
            "message": f"{type(exc).__name__}: {exc}",
        }

    if not isinstance(data, dict):
        return {"error": "invalid_baidu_direction_response", "raw": data}
    if str(data.get("status")) not in ("0", "OK", "ok"):
        return {
            "error": "baidu_direction_error",
            "status": data.get("status"),
            "message": data.get("message") or data.get("msg"),
            "raw": data,
        }
    summary = _extract_baidu_route_summary(data, api_mode)
    if summary.get("error"):
        return {**summary, "raw": data}
    return {
        "data_source": "baidu_direction_v2",
        "mode": api_mode,
        **summary,
    }
