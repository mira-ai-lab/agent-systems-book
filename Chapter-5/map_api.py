"""Chapter-5 地图服务 API 适配。

更新时间：2026-11-13

本模块只负责高德、百度地图 POI 请求、百度签名和原始返回解析，不包含酒店
搜索词、酒店推荐或 Agent 逻辑。
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


def norm_text(value: Optional[str]) -> str:
    """去除文本首尾空白；空值统一返回空字符串。"""
    return (value or "").strip()


async def _http_get_json(
    url: str,
    params: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """执行 JSON GET 请求，并统一字典返回值。"""
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
    """使用高德 POI 关键字接口查询指定城市内的地点。

    需要环境变量 ``AMAP_KEY``。返回原始 POI 列表；错误以 ``error`` 字段返回。
    """
    key = norm_text(os.getenv("AMAP_KEY"))
    if not key:
        return {"error": "AMAP_KEY not set"}

    normalized_region = norm_text(region)
    normalized_keyword = norm_text(keyword)
    if not normalized_region:
        return {"error": "region is required"}
    if not normalized_keyword and not norm_text(types):
        return {"error": "keyword or types is required"}

    params: Dict[str, Any] = {
        "key": key,
        "keywords": normalized_keyword,
        "region": normalized_region,
        "city_limit": "true",
        "page_size": str(max(1, min(int(limit or 10), 25))),
        "page_num": "1",
        "output": "JSON",
        "show_fields": "business",
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


def _md5_hex(text: str) -> str:
    """计算文本的 MD5 十六进制摘要。"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _baidu_sn(
    path: str,
    params_in_order: Sequence[Tuple[str, str]],
    sk: str,
) -> str:
    """计算百度地图 WebService API 的 SN 签名。"""
    query = urllib.parse.urlencode(
        params_in_order,
        safe="|,:",
        quote_via=urllib.parse.quote,
    )
    whole = f"{path}?{query}{sk}"
    return _md5_hex(urllib.parse.quote(whole, safe=""))


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
    """调用百度地图 Place API v2 搜索地点。

    需要环境变量 ``BAIDU_MAP_AK``；可选 ``BAIDU_MAP_SK`` 用于 SN 签名。
    """
    ak = norm_text(os.getenv("BAIDU_MAP_AK"))
    if not ak:
        return {"error": "BAIDU_MAP_AK not set"}
    sk = norm_text(os.getenv("BAIDU_MAP_SK"))

    params_in_order: List[Tuple[str, str]] = [
        ("query", norm_text(query)),
        ("region", norm_text(region)),
        ("output", "json"),
        ("scope", str(int(scope or 2))),
        ("page_size", str(max(1, min(int(page_size or 10), 20)))),
        ("page_num", str(max(0, int(page_num or 0)))),
        ("ak", ak),
    ]
    if tag:
        params_in_order.insert(1, ("tag", norm_text(tag)))
    if filter_:
        params_in_order.insert(5, ("filter", norm_text(filter_)))
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
