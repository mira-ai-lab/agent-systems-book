import hashlib
import asyncio
import os
import re
import time
import urllib.parse
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field


def norm_text(s: Optional[str]) -> str:
    return (s or "").strip()


def normalize_city_name(city: Optional[str]) -> str:
    """上海市 → 上海（地图 region / 城市表查找用）。"""
    c = norm_text(city)
    if c.endswith("市") and len(c) > 1:
        return c[:-1]
    return c


def require_non_empty(value: Optional[str], field: str) -> Tuple[bool, str]:
    if norm_text(value):
        return True, ""
    return False, f"{field} is required."


def parse_exact_date(date_str: str) -> Tuple[Optional[str], Optional[str]]:
    """
    强制使用精确日期格式 YYYY-MM-DD。
    返回 (标准化日期, 错误信息)。
    """
    s = norm_text(date_str)
    if not s:
        return None, "date is required (YYYY-MM-DD)."
    try:
        dt = datetime.strptime(s, "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d"), None
    except ValueError:
        return None, "date must be an exact YYYY-MM-DD string."


# ---- 相对日期解析（用于单个日期参数，保留原方案）----

_RELATIVE_DATE_OFFSETS = (
    ("大后天", 3),
    ("后天", 2),
    ("明天", 1),
    ("明日", 1),
    ("今天", 0),
    ("今日", 0),
    ("tomorrow", 1),
    ("today", 0),
)


def resolve_relative_date(
    date_str: str,
    ref: Optional[datetime] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """将 YYYY-MM-DD 或相对日期（今天/明天/后天等）转为标准日期。

    此函数用于工具参数级别的单个日期转换，输入通常是"今天""明天"等简单词。
    正则方案在此场景下足够可靠，无需 LLM。
    """
    s = norm_text(date_str)
    if not s:
        return None, "date is required (YYYY-MM-DD or 今天/明天/后天)."

    exact, _ = parse_exact_date(s)
    if exact:
        return exact, None

    low = s.lower()
    for keyword, offset in _RELATIVE_DATE_OFFSETS:
        if keyword in low:
            base = ref or datetime.now()
            return (base + timedelta(days=offset)).strftime("%Y-%m-%d"), None

    return None, (
        f"unsupported date expression: {date_str!r} "
        "(use YYYY-MM-DD or 今天/明天/后天)."
    )


# ---- 正则回退方案（保留作为 LLM 解析失败时的兜底）----

def resolve_trip_dates_from_query(
    query: str,
    ref: Optional[datetime] = None,
) -> List[str]:
    """从用户话术中解析出行日期列表（正则回退方案）。

    支持：下周/下礼拜/下下周、N 天、YYYY-MM-DD。
    当 LLM 解析不可用或失败时作为兜底。
    """
    ref = ref or datetime.now()
    # 正则提取精确日期，并验证合法性
    explicit_raw = re.findall(r"\d{4}-\d{2}-\d{2}", query)
    explicit_valid: List[str] = []
    for d in explicit_raw:
        try:
            datetime.strptime(d, "%Y-%m-%d")
            explicit_valid.append(d)
        except ValueError:
            pass
    if explicit_valid:
        return explicit_valid

    n_days = 3
    md = re.search(r"(\d+)\s*天", query)
    if md:
        n_days = max(1, int(md.group(1)))

    if "下下周" in query:
        days_ahead = (7 - ref.weekday()) % 7 + 7
    elif "下周" in query or "下礼拜" in query:
        days_ahead = (7 - ref.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
    else:
        days_ahead = 1

    start = ref + timedelta(days=days_ahead)
    return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n_days)]


def build_trip_date_anchor(
    query: str,
    ref: Optional[datetime] = None,
) -> Dict[str, Any]:
    """生成全链路统一的日期锚定信息（同步正则回退方案）。"""
    ref = ref or datetime.now()
    today = ref.strftime("%Y-%m-%d")
    trip_dates = resolve_trip_dates_from_query(query, ref)
    trip_range = (
        f"{trip_dates[0]} 至 {trip_dates[-1]}" if len(trip_dates) > 1 else trip_dates[0]
    )
    anchor_block = (
        f"[系统日期锚定：今天是 {today}；"
        f"本请求出行日期为 {trip_range}（{', '.join(trip_dates)}）。"
        f"所有子任务与最终回复必须使用上述日期，禁止编造 2024 等错误年份。]"
    )
    return {
        "today": today,
        "trip_dates": trip_dates,
        "trip_range": trip_range,
        "anchor_block": anchor_block,
    }


# ---- LLM 日期解析（结构化输出，优先使用）----


class TripDateExtraction(BaseModel):
    """LLM 从用户话术中提取的出行日期信息。"""
    start_date: str = Field(
        description="出行起始日期，格式 YYYY-MM-DD，根据今天的日期推算",
    )
    duration_days: int = Field(
        default=3,
        description="出行天数，默认3天",
    )
    trip_dates: List[str] = Field(
        description="每一天的出行日期列表，每个日期为 YYYY-MM-DD 格式",
    )


_DATE_EXTRACT_PROMPT = """\
今天是 {today}。

请从以下用户话术中提取出行日期信息。你需要：
1. 理解用户提到的任何相对日期表达（如"明天"、"后天"、"下周"、"下礼拜"、"大后天"、"6月5号"、"本周末"等）
2. 根据今天是 {today} 来计算出具体的 YYYY-MM-DD 日期
3. 确定出行的天数（如果用户说了"N天"、"一周"、"半个月"等，据此计算；否则默认3天）
4. 生成每一天的出行日期列表

用户话术: {query}

请严格按照 TripDateExtraction 的字段返回结构化数据。"""


async def resolve_trip_dates_from_llm(
    query: str,
    ref: Optional[datetime] = None,
    llm: Optional[Any] = None,
) -> List[str]:
    """使用 LLM 结构化输出从用户话术中解析出行日期列表。

    优势：能理解"6月5号出发玩3天"、"本周末去杭州"、"下下个礼拜"等
    正则无法覆盖的自然语言表达。

    如果 LLM 不可用或解析失败，自动回退到正则方案 resolve_trip_dates_from_query。
    """
    if llm is None:
        return resolve_trip_dates_from_query(query, ref)

    ref = ref or datetime.now()
    today = ref.strftime("%Y-%m-%d")

    try:
        structured_llm = llm.with_structured_output(TripDateExtraction)
        prompt = _DATE_EXTRACT_PROMPT.format(today=today, query=query)
        result: TripDateExtraction = await structured_llm.ainvoke(prompt)

        # 验证返回的日期格式是否合法
        validated_dates: List[str] = []
        for d in result.trip_dates:
            try:
                datetime.strptime(d, "%Y-%m-%d")
                validated_dates.append(d)
            except ValueError:
                pass

        if validated_dates:
            return validated_dates

        # LLM 返回的 trip_dates 全部不合法，尝试用 start_date + duration_days 计算
        try:
            start = datetime.strptime(result.start_date, "%Y-%m-%d")
            return [
                (start + timedelta(days=i)).strftime("%Y-%m-%d")
                for i in range(max(1, result.duration_days))
            ]
        except ValueError:
            pass

    except Exception as e:
        print(f"[日期解析] LLM 解析失败，回退到正则方案: {e}", flush=True)

    # 回退到正则方案
    return resolve_trip_dates_from_query(query, ref)


async def build_trip_date_anchor_async(
    query: str,
    ref: Optional[datetime] = None,
    llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """生成全链路统一的日期锚定信息（异步版本，优先使用 LLM 解析）。

    如果 LLM 不可用，回退到同步正则方案 build_trip_date_anchor。
    """
    ref = ref or datetime.now()
    today = ref.strftime("%Y-%m-%d")

    trip_dates = await resolve_trip_dates_from_llm(query, ref, llm)

    trip_range = (
        f"{trip_dates[0]} 至 {trip_dates[-1]}" if len(trip_dates) > 1 else trip_dates[0]
    )
    anchor_block = (
        f"[系统日期锚定：今天是 {today}；"
        f"本请求出行日期为 {trip_range}（{', '.join(trip_dates)}）。"
        f"所有子任务与最终回复必须使用上述日期，禁止编造 2024 等错误年份。]"
    )
    return {
        "today": today,
        "trip_dates": trip_dates,
        "trip_range": trip_range,
        "anchor_block": anchor_block,
    }


def stable_int(*parts: str) -> int:
    raw = "|".join([norm_text(p) for p in parts]).encode("utf-8")
    h = hashlib.sha256(raw).hexdigest()
    return int(h[:8], 16)


def pick_one(items: Sequence[Dict[str, Any]], *seed_parts: str) -> Dict[str, Any]:
    if not items:
        return {}
    idx = stable_int(*seed_parts) % len(items)
    return dict(items[idx])


def pick_many(items: Sequence[Dict[str, Any]], n: int, *seed_parts: str) -> List[Dict[str, Any]]:
    if not items:
        return []
    n = max(1, int(n or 1))
    # 通过 stable_int + 索引步进实现确定性“洗牌”
    start = stable_int(*seed_parts) % len(items)
    step = (stable_int("step", *seed_parts) % (len(items) - 1) + 1) if len(items) > 1 else 1
    out: List[Dict[str, Any]] = []
    i = start
    seen = set()
    while len(out) < min(n, len(items)) and i not in seen:
        seen.add(i)
        out.append(dict(items[i]))
        i = (i + step) % len(items)
    return out



# ---- API 适配器（优先使用真实 API，回退到模拟数据）----

_HTTP_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
_BAIDU_API_LOCK: Optional[asyncio.Lock] = None


def _baidu_api_lock() -> asyncio.Lock:
    global _BAIDU_API_LOCK
    if _BAIDU_API_LOCK is None:
        _BAIDU_API_LOCK = asyncio.Lock()
    return _BAIDU_API_LOCK
_DOTENV_LOADED = False


def _project_root_dir() -> str:
    from chapter6.paths import BOOK_ROOT

    return str(BOOK_ROOT)


def ensure_project_dotenv_loaded() -> None:
    """
    使 `.env` 加载在不同 IDE 运行/调试配置下更加健壮。
    如果存在则加载 `<project_root>/.env` 一次。
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    root = _project_root_dir()
    dotenv_path = os.path.join(root, ".env")
    try:
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path, override=False)
    finally:
        _DOTENV_LOADED = True


async def _http_get_json(url: str, params: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
        r = await client.get(url, params=params, headers=headers)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            return data
        return {"data": data}


def _advice_from_condition(condition: str) -> str:
    c = (condition or "").strip()
    if "雨" in c:
        return "带伞，注意路滑"
    if "雪" in c:
        return "注意防滑保暖"
    if "晴" in c:
        return "注意防晒补水"
    if "雾" in c:
        return "注意能见度与交通安全"
    return "注意增减衣物与出行安全"


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        return int(float(s))
    except Exception:
        return None


async def wttr_weather_by_city_and_date(city: str, date: str) -> Dict[str, Any]:
    """
    Free weather API via wttr.in.

    - If date is today: uses `format=3` (text).
    - Otherwise: uses `format=j1` (JSON) and picks the matching date.
    """
    norm_date, derr = parse_exact_date(date)
    if derr:
        return {"error": derr}
    q = norm_text(city)
    if not q:
        return {"error": "city is required."}

    # Today shortcut: cheap and stable
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
            if norm_date == today:
                r = await client.get(f"https://wttr.in/{q}", params={"format": "3", "lang": "zh"})
                r.raise_for_status()
                text = (r.text or "").strip()
                if not text:
                    return {"error": "empty_response"}
                return {"city": q, "date": norm_date, "text": text, "data_source": "wttr.in(format=3)"}

            # Forecast JSON
            r = await client.get(f"https://wttr.in/{q}", params={"format": "j1", "lang": "zh"})
            r.raise_for_status()
            data = r.json() if r.content else {}
    except Exception as e:
        return {"error": f"wttr_request_failed: {str(e)}"}

    weather = data.get("weather") if isinstance(data, dict) else None
    if not isinstance(weather, list) or not weather:
        return {"error": "invalid_wttr_response", "raw": data}

    picked = None
    for d in weather:
        if isinstance(d, dict) and (d.get("date") == norm_date):
            picked = d
            break
    if picked is None:
        return {"error": "date_out_of_range", "available_dates": [w.get("date") for w in weather if isinstance(w, dict)], "raw": data}

    # Best-effort mapping
    max_temp = _safe_int(picked.get("maxtempC"))
    min_temp = _safe_int(picked.get("mintempC"))
    hourly = picked.get("hourly") or []
    desc = ""
    if isinstance(hourly, list) and hourly:
        h0 = hourly[0] if isinstance(hourly[0], dict) else {}
        # prefer Chinese desc if present
        lang_desc = h0.get("lang_zh") or h0.get("lang_zh-cn") or h0.get("lang_zh-hans")
        if isinstance(lang_desc, list) and lang_desc and isinstance(lang_desc[0], dict):
            desc = lang_desc[0].get("value") or ""
        if not desc:
            wx = h0.get("weatherDesc")
            if isinstance(wx, list) and wx and isinstance(wx[0], dict):
                desc = wx[0].get("value") or ""

    cond = desc or "未知"
    return {
        "city": q,
        "date": norm_date,
        "forecast": {
            "condition": cond,
            "temp_high_c": max_temp,
            "temp_low_c": min_temp,
            "advice": _advice_from_condition(cond),
        },
        "data_source": "wttr.in(format=j1)",
        "raw": picked,
    }


async def amap_geocode(address: str) -> Dict[str, Any]:
    """
    高德地理编码：把“城市/地址”转成经纬度与 adcode。
    需要环境变量：AMAP_KEY
    """
    ensure_project_dotenv_loaded()
    key = norm_text(os.getenv("AMAP_KEY"))
    if not key:
        return {"error": "AMAP_KEY not set"}
    q = norm_text(address)
    if not q:
        return {"error": "address is required"}
    url = "https://restapi.amap.com/v3/geocode/geo"
    data = await _http_get_json(url, params={"key": key, "address": q, "output": "JSON"})
    geocodes = data.get("geocodes") if isinstance(data, dict) else None
    if not isinstance(geocodes, list) or not geocodes:
        return {"error": "geocode_not_found", "raw": data}
    g0 = geocodes[0] or {}
    return {
        "formatted_address": g0.get("formatted_address"),
        "location": g0.get("location"),  # "lng,lat"
        "adcode": g0.get("adcode"),
        "city": g0.get("city") or g0.get("province"),
        "raw": g0,
    }


async def amap_weather_by_city_and_date(city: str, date: str) -> Dict[str, Any]:
    """
    高德天气：按 city + 精确 date(YYYY-MM-DD) 返回预报（高德一般提供未来 4 天）。
    需要环境变量：AMAP_KEY
    """
    ensure_project_dotenv_loaded()
    key = norm_text(os.getenv("AMAP_KEY"))
    if not key:
        return {"error": "AMAP_KEY not set"}
    norm_date, derr = parse_exact_date(date)
    if derr:
        return {"error": derr}

    geo = await amap_geocode(city)
    if geo.get("error"):
        return {"error": f"geocode_failed: {geo.get('error')}", "detail": geo}
    adcode = geo.get("adcode")
    if not adcode:
        return {"error": "adcode_not_found", "detail": geo}

    url = "https://restapi.amap.com/v3/weather/weatherInfo"
    data = await _http_get_json(
        url,
        params={"key": key, "city": adcode, "extensions": "all", "output": "JSON"},
    )
    # 预期结构: forecasts[0].casts[] 包含 date/dayweather/nightweather/daytemp/nighttemp
    forecasts = data.get("forecasts") if isinstance(data, dict) else None
    casts: List[Dict[str, Any]] = []
    if isinstance(forecasts, list) and forecasts:
        fc0 = forecasts[0] or {}
        casts = fc0.get("casts") or []
    if not casts:
        return {"error": "no_forecast_data", "raw": data}

    picked: Optional[Dict[str, Any]] = None
    for c in casts:
        if (c.get("date") or "") == norm_date:
            picked = c
            break
    if picked is None:
        return {
            "error": "date_out_of_range",
            "available_dates": [c.get("date") for c in casts if isinstance(c, dict)],
            "raw": data,
        }

    day_weather = picked.get("dayweather") or ""
    night_weather = picked.get("nightweather") or ""
    cond = day_weather or night_weather or ""
    try:
        hi = int(float(picked.get("daytemp")))
    except Exception:
        hi = None
    try:
        lo = int(float(picked.get("nighttemp")))
    except Exception:
        lo = None

    return {
        "city": geo.get("city") or city.strip(),
        "date": norm_date,
        "forecast": {
            "condition": cond,
            "temp_high_c": hi,
            "temp_low_c": lo,
            "advice": _advice_from_condition(cond),
            "day": {"weather": day_weather, "wind": picked.get("daywind"), "power": picked.get("daypower")},
            "night": {"weather": night_weather, "wind": picked.get("nightwind"), "power": picked.get("nightpower")},
        },
        "data_source": "amap_weather",
        "raw": picked,
    }


async def amap_place_text_search(region: str, keyword: str, *, limit: int = 10, types: Optional[str] = None) -> Dict[str, Any]:
    """
    高德 POI 关键字搜索（v5）。
    需要环境变量：AMAP_KEY
    """
    ensure_project_dotenv_loaded()
    key = norm_text(os.getenv("AMAP_KEY"))
    if not key:
        return {"error": "AMAP_KEY not set"}
    reg = norm_text(region)
    kw = norm_text(keyword)
    if not reg:
        return {"error": "region is required"}
    if not kw and not norm_text(types):
        return {"error": "keyword or types is required"}
    url = "https://restapi.amap.com/v5/place/text"
    params: Dict[str, Any] = {
        "key": key,
        "keywords": kw,
        "region": reg,
        "city_limit": "true",
        "page_size": str(max(1, min(int(limit or 10), 25))),
        "page_num": "1",
        "output": "JSON",
        # business 字段在酒店/餐厅可用时包含评分/价格信息
        "show_fields": "business",
    }
    if types:
        params["types"] = types
    data = await _http_get_json(url, params=params)
    pois = data.get("pois") if isinstance(data, dict) else None
    if not isinstance(pois, list):
        return {"error": "invalid_poi_response", "raw": data}
    return {"pois": pois, "raw": data}


def _md5_hex(s: str) -> str:
    import hashlib as _hashlib

    return _hashlib.md5(s.encode("utf-8")).hexdigest()


def _baidu_sn(path: str, params_in_order: List[Tuple[str, str]], sk: str) -> str:
    """
    百度 WebService API 的 SN 签名（尽力而为）。
    参考：百度 SN 签名机制（需要有序参数）。
    """
    qs = urllib.parse.urlencode(params_in_order, safe="|,:", quote_via=urllib.parse.quote)
    whole = f"{path}?{qs}{sk}"
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
    """
    Baidu Map Place API v2 search.
    Docs: https://lbsyun.baidu.com/docs/webapi?title=placev3%2Fguide%2Fwebservice-placeapiV3%2FinterfaceDocumentV2
    """
    ensure_project_dotenv_loaded()
    ak = norm_text(os.getenv("BAIDU_MAP_AK"))
    if not ak:
        return {"error": "BAIDU_MAP_AK not set"}
    sk = norm_text(os.getenv("BAIDU_MAP_SK"))

    url = "https://api.map.baidu.com/place/v2/search"
    # 保持参数顺序稳定以用于 SN 签名
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
        # 根据常见示例将 filter 放在 scope 附近；顺序必须与请求匹配
        params_in_order.insert(5, ("filter", norm_text(filter_)))
    if sk:
        params_in_order.append(("timestamp", str(int(time.time()))))
        sn = _baidu_sn("/place/v2/search", params_in_order, sk)
        params_in_order.append(("sn", sn))

    async with _baidu_api_lock():
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
            r = await client.get(url, params=params_in_order)
            r.raise_for_status()
            data = r.json()

    if not isinstance(data, dict):
        return {"error": "invalid_baidu_response", "raw": data}
    # status: 0 ok
    if str(data.get("status")) not in ("0", "OK", "ok"):
        return {"error": "baidu_error", "raw": data, "message": data.get("message")}
    results = data.get("results")
    if not isinstance(results, list):
        return {"error": "invalid_baidu_results", "raw": data}
    return {"results": results, "raw": data}


def _baidu_result_to_common_poi(item: Dict[str, Any]) -> Dict[str, Any]:
    loc = item.get("location") or {}
    lat = loc.get("lat")
    lng = loc.get("lng")
    detail = item.get("detail_info") or {}
    # 百度 detail_info 中的 price / overall_rating 有时是字符串
    try:
        rating = float(detail.get("overall_rating")) if detail.get("overall_rating") is not None and str(detail.get("overall_rating")).strip() else None
    except Exception:
        rating = None
    price = detail.get("price")
    try:
        avg_price = int(float(str(price).replace("￥", "").replace("元", "").strip())) if price is not None and str(price).strip() else None
    except Exception:
        avg_price = None

    location_str = None
    if lng is not None and lat is not None:
        location_str = f"{lng},{lat}"
    return {
        "name": item.get("name"),
        "district": item.get("area") or item.get("city") or item.get("address"),
        "address": item.get("address"),
        "tel": item.get("telephone"),
        "location": location_str,
        "rating": rating,
        "avg_price_cny": avg_price,
        "type": detail.get("type") or item.get("tag") or item.get("detail_info", {}).get("tag") or item.get("type"),
        "raw": item,
    }


def _poi_to_hotel(poi: Dict[str, Any]) -> Dict[str, Any]:
    biz = poi.get("business") or {}
    rating = biz.get("rating")
    cost = biz.get("cost")
    try:
        rating_f = float(rating) if rating is not None and str(rating).strip() else None
    except Exception:
        rating_f = None
    try:
        cost_i = int(float(cost)) if cost is not None and str(cost).strip() else None
    except Exception:
        cost_i = None
    return {
        "name": poi.get("name"),
        "district": poi.get("adname") or poi.get("address"),
        "address": poi.get("address"),
        "tel": poi.get("tel"),
        "location": poi.get("location"),
        "rating": rating_f,
        "avg_price_cny": cost_i,
        "type": poi.get("type"),
    }


def _poi_to_restaurant(poi: Dict[str, Any]) -> Dict[str, Any]:
    biz = poi.get("business") or {}
    rating = biz.get("rating")
    cost = biz.get("cost")
    try:
        rating_f = float(rating) if rating is not None and str(rating).strip() else None
    except Exception:
        rating_f = None
    try:
        cost_i = int(float(cost)) if cost is not None and str(cost).strip() else None
    except Exception:
        cost_i = None
    return {
        "name": poi.get("name"),
        "cuisine": poi.get("type"),
        "avg_price_cny": cost_i,
        "rating": rating_f,
        "district": poi.get("adname"),
        "address": poi.get("address"),
        "tel": poi.get("tel"),
        "location": poi.get("location"),
    }


async def fetch_hotels_from_api(city: str, *, limit: int = 10, keyword: Optional[str] = None) -> Dict[str, Any]:
    # keyword: 原始 preferences；若已含「酒店」则视为完整搜索词，避免重复拼接
    ensure_project_dotenv_loaded()
    region = normalize_city_name(city) or norm_text(city)
    pref = norm_text(keyword)
    q = pref if (pref and "酒店" in pref) else build_hotel_search_query(city, keyword)
    page_n = max(1, min(int(limit or 10), 20))
    fetch_size = min(20, page_n * 2)

    if norm_text(os.getenv("BAIDU_MAP_AK")):
        res = await baidu_place_v2_search(
            query=q,
            region=region,
            scope=2,
            page_size=fetch_size,
            filter_="industry_type:hotel|sort_name:total_score|sort_rule:0",
        )
        if not res.get("error"):
            hotels = []
            for it in (res.get("results") or []):
                if isinstance(it, dict):
                    hotels.append(_baidu_result_to_common_poi(it))
            hotels = _filter_hotels_for_city(hotels, region)[:page_n]
            if hotels:
                # 结果过少时自动放宽检索（避免核心城区过窄导致仅 1 家）
                min_expected = min(3, page_n)
                if len(hotels) < min_expected:
                    broad_q = f"{normalize_city_name(city)} 酒店"
                    if broad_q != q:
                        res_broad = await baidu_place_v2_search(
                            query=broad_q,
                            region=region,
                            scope=2,
                            page_size=min(20, page_n * 3),
                            filter_="industry_type:hotel|sort_name:total_score|sort_rule:0",
                        )
                        if not res_broad.get("error"):
                            for it in (res_broad.get("results") or []):
                                if isinstance(it, dict):
                                    hotels.append(_baidu_result_to_common_poi(it))
                            merged = _filter_hotels_for_city(hotels, region, strict_core=False)
                            dedup: List[Dict[str, Any]] = []
                            seen = set()
                            for h in merged:
                                key = (norm_text(h.get("name")), norm_text(h.get("address")))
                                if key in seen:
                                    continue
                                seen.add(key)
                                dedup.append(h)
                            hotels = dedup[:page_n]
                return {
                    "hotels": hotels,
                    "data_source": "baidu_place_v2",
                    "search_query": q,
                }

    res2 = await amap_place_text_search(region=region, keyword=q, limit=fetch_size)
    if res2.get("error"):
        return res2
    hotels2 = []
    for p in (res2.get("pois") or []):
        if isinstance(p, dict):
            hotels2.append(_poi_to_hotel(p))
    hotels2 = _filter_hotels_for_city(hotels2, region)[: max(1, int(limit or 10))]
    return {
        "hotels": hotels2,
        "data_source": "amap_place_text",
        "search_query": q,
        "note": "已优先筛选核心城区，排除远郊结果" if hotels2 else "核心城区无结果",
    }


_ATTRACTION_POSITIVE_KW = (
    "景区", "景点", "公园", "园林", "博物馆", "美术馆", "展览馆", "纪念馆",
    "故居", "古镇", "古城", "古街", "寺", "庙", "祠", "塔", "湖", "山", "湿地",
)
_ATTRACTION_REJECT_KW = (
    "酒店", "宾馆", "民宿", "小区", "公寓", "写字楼", "公司", "培训", "医院", "学校", "商场",
)
_ADMIN_NAME_SUFFIX = ("特别行政区", "自治州", "地区", "盟", "市", "省", "县", "区")


def _normalize_attraction_query(preferences: Optional[str]) -> str:
    pref = norm_text(preferences)
    if not pref:
        return "景点"
    if any(k in pref for k in ("历史", "文化", "古迹", "博物馆", "园林", "人文")):
        return "历史文化 景点"
    if any(k in pref for k in ("自然", "风光", "湿地", "湖", "山", "徒步")):
        return "自然风光 景点"
    return f"{pref} 景点"


def _is_valid_attraction_poi(poi: Dict[str, Any]) -> bool:
    name = norm_text(poi.get("name"))
    if not name or len(name) < 2:
        return False
    for suffix in _ADMIN_NAME_SUFFIX:
        if name.endswith(suffix):
            stem = name[: -len(suffix)]
            if 1 <= len(stem) <= 6:
                return False
    if not (norm_text(poi.get("address")) or norm_text(poi.get("district")) or norm_text(poi.get("location"))):
        return False
    if any(k in name for k in _ATTRACTION_REJECT_KW):
        return False
    typ = norm_text(poi.get("type")).lower()
    raw = poi.get("raw") or {}
    detail = raw.get("detail_info") or {}
    tag = norm_text(detail.get("classified_poi_tag") or detail.get("tag") or "")
    text = f"{name} {typ} {tag}"
    if any(k in text for k in _ATTRACTION_REJECT_KW):
        return False
    if tag:
        return any(k in tag for k in _ATTRACTION_POSITIVE_KW)
    return any(k in text for k in _ATTRACTION_POSITIVE_KW)


async def fetch_attractions_from_api(
    city: str,
    *,
    preferences: Optional[str] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    获取用于行程规划的景点/POI 候选列表。
    如果已配置则优先使用百度 Place v2，否则回退到高德。
    """
    ensure_project_dotenv_loaded()
    query = _normalize_attraction_query(preferences)
    if norm_text(os.getenv("BAIDU_MAP_AK")):
        res = await baidu_place_v2_search(
            query=query,
            region=city,
            scope=2,
            page_size=min(20, max(1, int(limit or 10)) * 3),
            filter_="industry_type:life|sort_name:overall_rating|sort_rule:0",
        )
        if not res.get("error"):
            items = []
            for it in (res.get("results") or []):
                if isinstance(it, dict):
                    poi = _baidu_result_to_common_poi(it)
                    if _is_valid_attraction_poi(poi):
                        # 减少下游 LLM prompt 的负载大小
                        poi.pop("raw", None)
                        items.append(poi)
                    if len(items) >= max(1, int(limit or 10)):
                        break
            return {
                "attractions": items,
                "data_source": "baidu_place_v2",
                "search_query": query,
            }

    # 高德回退方案
    res2 = await amap_place_text_search(
        region=city,
        keyword=query,
        limit=min(20, max(1, int(limit or 10)) * 3),
    )
    if res2.get("error"):
        return res2
    items2 = []
    for p in (res2.get("pois") or []):
        if isinstance(p, dict):
            item = {
                "name": p.get("name"),
                "district": p.get("adname") or p.get("address"),
                "address": p.get("address"),
                "tel": p.get("tel"),
                "location": p.get("location"),
                "rating": (p.get("business") or {}).get("rating") if isinstance(p.get("business"), dict) else None,
                "avg_price_cny": (p.get("business") or {}).get("cost") if isinstance(p.get("business"), dict) else None,
                "type": p.get("type"),
            }
            if _is_valid_attraction_poi(item):
                items2.append(item)
            if len(items2) >= max(1, int(limit or 10)):
                break
    return {"attractions": items2, "data_source": "amap_place_text", "search_query": query}


def _poi_slot(poi: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """行程 slot 用 POI 摘要（不含 raw 等大字段）。"""
    if not isinstance(poi, dict) or not norm_text(poi.get("name")):
        return None
    return {
        k: poi.get(k)
        for k in (
            "name", "address", "district", "location", "rating",
            "avg_price_cny", "type", "tel", "cuisine",
        )
        if poi.get(k) is not None
    }


def _parse_lng_lat(location: Any) -> Optional[Tuple[float, float]]:
    """解析地图 API 常见的 "lng,lat" 坐标。"""
    if not location:
        return None
    if isinstance(location, dict):
        lng = location.get("lng") or location.get("lon") or location.get("longitude")
        lat = location.get("lat") or location.get("latitude")
    else:
        parts = str(location).split(",")
        if len(parts) != 2:
            return None
        lng, lat = parts[0], parts[1]
    if lng is None or lat is None:
        return None
    try:
        return float(str(lng).strip()), float(str(lat).strip())
    except (TypeError, ValueError):
        return None


def _haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> int:
    """按经纬度估算两点直线距离（米）。"""
    lng1, lat1 = a
    lng2, lat2 = b
    radius_m = 6_371_000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lng2 - lng1)
    h = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return int(2 * radius_m * math.atan2(math.sqrt(h), math.sqrt(1 - h)))


def _estimate_duration_min(distance_m: int, mode: str) -> int:
    """无 Directions API 时按常见城市出行速度估算耗时。"""
    speeds_kmh = {
        "walk": 4.5,
        "walking": 4.5,
        "bike": 12,
        "riding": 12,
        "drive": 24,
        "driving": 24,
        "transit": 18,
    }
    speed = speeds_kmh.get(mode, speeds_kmh["transit"])
    return max(1, int(round((distance_m / 1000) / speed * 60)))


def _select_baidu_direction_mode(distance_m: int) -> str:
    """按两点距离选择百度 Direction API 类型。"""
    if distance_m <= 1200:
        return "walking"
    if distance_m <= 5000:
        return "riding"
    if distance_m <= 30000:
        return "transit"
    return "driving"


def _coord_to_baidu_param(coord: Tuple[float, float]) -> str:
    """Baidu Direction API 使用 lat,lng；本项目 POI location 存储为 lng,lat。"""
    lng, lat = coord
    return f"{lat},{lng}"


def _extract_baidu_route_summary(data: Dict[str, Any], mode: str) -> Dict[str, Any]:
    result = data.get("result") if isinstance(data, dict) else None
    if not isinstance(result, dict):
        return {"error": "invalid_baidu_direction_result"}

    routes = result.get("routes")
    if not isinstance(routes, list) or not routes:
        routes = result.get("routes") or result.get("routes_list") or []
    if isinstance(routes, list) and routes:
        route = routes[0] if isinstance(routes[0], dict) else {}
    else:
        route = {}

    # transit 有些响应将方案放在 result.routes[0].scheme / scheme 里，做宽松兼容。
    if mode == "transit" and isinstance(route.get("scheme"), list) and route["scheme"]:
        scheme = route["scheme"][0]
        if isinstance(scheme, dict):
            route = scheme

    distance = route.get("distance") or route.get("distance_m")
    duration = route.get("duration") or route.get("duration_s")
    try:
        distance_m = int(float(distance)) if distance is not None else None
    except (TypeError, ValueError):
        distance_m = None
    try:
        duration_min = int(round(float(duration) / 60)) if duration is not None else None
    except (TypeError, ValueError):
        duration_min = None

    steps = route.get("steps")
    return {
        "distance_m": distance_m,
        "duration_min": duration_min,
        "steps_count": len(steps) if isinstance(steps, list) else None,
    }


async def baidu_direction_route(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    *,
    mode: str,
    region: Optional[str] = None,
) -> Dict[str, Any]:
    """调用百度 Direction API 获取两点路线；失败时返回 error，不抛出。"""
    ensure_project_dotenv_loaded()
    ak = norm_text(os.getenv("BAIDU_MAP_AK"))
    if not ak:
        return {"error": "BAIDU_MAP_AK not set"}

    api_mode = mode if mode in ("driving", "walking", "riding", "transit") else "transit"
    path = f"/direction/v2/{api_mode}"
    url = f"https://api.map.baidu.com{path}"
    params_in_order: List[Tuple[str, str]] = [
        ("origin", _coord_to_baidu_param(origin)),
        ("destination", _coord_to_baidu_param(destination)),
        ("ak", ak),
        ("output", "json"),
    ]
    if api_mode == "transit" and region:
        params_in_order.append(("region", normalize_city_name(region)))

    sk = norm_text(os.getenv("BAIDU_MAP_SK"))
    if sk:
        params_in_order.append(("timestamp", str(int(time.time()))))
        params_in_order.append(("sn", _baidu_sn(path, params_in_order, sk)))

    try:
        async with _baidu_api_lock():
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
                resp = await client.get(url, params=dict(params_in_order))
                resp.raise_for_status()
                data = resp.json()
    except Exception as exc:
        return {"error": "baidu_direction_request_failed", "message": f"{type(exc).__name__}: {exc}"}

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


def build_local_route_plan(
    pois: List[Dict[str, Any]],
    *,
    mode: str = "transit",
) -> Dict[str, Any]:
    """
    基于 POI 经纬度生成当日局部路线顺序和路段估算。
    当前实现使用坐标贪心排序；有地图 Directions API 时可在此替换为真实路径规划。
    """
    valid: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for poi in pois:
        if not isinstance(poi, dict) or not norm_text(poi.get("name")):
            continue
        coord = _parse_lng_lat(poi.get("location"))
        if not coord:
            skipped.append(norm_text(poi.get("name")))
            continue
        valid.append({**poi, "_coord": coord})

    if len(valid) < 2:
        return {
            "ordered_pois": [_poi_slot(p) for p in valid if _poi_slot(p)],
            "segments": [],
            "routing_status": "insufficient_coordinates",
            "skipped_without_location": skipped,
            "mode": mode,
        }

    ordered = [valid.pop(0)]
    while valid:
        current = ordered[-1]["_coord"]
        next_idx = min(
            range(len(valid)),
            key=lambda i: _haversine_m(current, valid[i]["_coord"]),
        )
        ordered.append(valid.pop(next_idx))

    segments = []
    total_distance = 0
    total_duration = 0
    for prev, curr in zip(ordered, ordered[1:]):
        distance = _haversine_m(prev["_coord"], curr["_coord"])
        duration = _estimate_duration_min(distance, mode)
        total_distance += distance
        total_duration += duration
        segments.append({
            "from": prev.get("name"),
            "to": curr.get("name"),
            "distance_m": distance,
            "duration_min_est": duration,
            "mode": mode,
            "source": "coordinate_estimate",
        })

    clean_ordered = []
    for poi in ordered:
        item = dict(poi)
        item.pop("_coord", None)
        slot = _poi_slot(item)
        if slot:
            clean_ordered.append(slot)

    return {
        "ordered_pois": clean_ordered,
        "segments": segments,
        "total_distance_m": total_distance,
        "total_duration_min_est": total_duration,
        "routing_status": "estimated_by_coordinates",
        "skipped_without_location": skipped,
        "mode": mode,
    }


async def build_local_route_plan_with_baidu(
    pois: List[Dict[str, Any]],
    *,
    region: Optional[str] = None,
    mode: str = "auto",
) -> Dict[str, Any]:
    """优先使用百度路线规划；不可用时回退到坐标估算。"""
    estimate = build_local_route_plan(pois, mode="transit" if mode == "auto" else mode)
    if estimate.get("routing_status") != "estimated_by_coordinates":
        return estimate

    coord_by_name: Dict[str, Tuple[float, float]] = {}
    for poi in pois:
        if isinstance(poi, dict) and norm_text(poi.get("name")):
            coord = _parse_lng_lat(poi.get("location"))
            if coord:
                coord_by_name[norm_text(poi.get("name"))] = coord

    baidu_segments = []
    total_distance = 0
    total_duration = 0
    used_baidu = False
    for segment in estimate.get("segments") or []:
        from_name = norm_text(segment.get("from"))
        to_name = norm_text(segment.get("to"))
        origin = coord_by_name.get(from_name)
        destination = coord_by_name.get(to_name)
        if not origin or not destination:
            baidu_segments.append(segment)
            continue

        estimated_distance = int(segment.get("distance_m") or _haversine_m(origin, destination))
        api_mode = _select_baidu_direction_mode(estimated_distance) if mode == "auto" else mode
        route = await baidu_direction_route(origin, destination, mode=api_mode, region=region)
        if route.get("error"):
            baidu_segments.append({
                **segment,
                "api_mode": api_mode,
                "baidu_error": route.get("message") or route.get("error"),
            })
            total_distance += int(segment.get("distance_m") or 0)
            total_duration += int(segment.get("duration_min_est") or 0)
            continue

        distance_m = route.get("distance_m") or segment.get("distance_m")
        duration_min = route.get("duration_min") or segment.get("duration_min_est")
        used_baidu = True
        total_distance += int(distance_m or 0)
        total_duration += int(duration_min or 0)
        baidu_segments.append({
            "from": segment.get("from"),
            "to": segment.get("to"),
            "distance_m": distance_m,
            "duration_min": duration_min,
            "mode": route.get("mode") or api_mode,
            "source": "baidu_direction_v2",
            "steps_count": route.get("steps_count"),
        })

    return {
        **estimate,
        "segments": baidu_segments,
        "total_distance_m": total_distance,
        "total_duration_min": total_duration if used_baidu else None,
        "total_duration_min_est": None if used_baidu else estimate.get("total_duration_min_est"),
        "routing_status": "baidu_direction_v2" if used_baidu else "estimated_by_coordinates",
        "mode": "auto" if mode == "auto" else mode,
    }


async def enrich_itinerary_routes_with_baidu(itinerary: Dict[str, Any]) -> Dict[str, Any]:
    """根据每日 slot POI 的经纬度刷新 local_route。"""
    plan = itinerary.get("plan")
    if not isinstance(plan, list):
        return itinerary
    for day in plan:
        if not isinstance(day, dict):
            continue
        pois: List[Dict[str, Any]] = []
        for slot in day.get("slots") or []:
            if isinstance(slot, dict) and isinstance(slot.get("poi"), dict):
                pois.append(slot["poi"])
        if pois:
            day["local_route"] = await build_local_route_plan_with_baidu(
                pois,
                region=day.get("city") or itinerary.get("destination_city"),
                mode="auto",
            )
    return itinerary


def build_itinerary_from_candidates(
    departure_city: str,
    destination_city: str,
    days: int,
    *,
    preferences: Optional[str] = None,
    must_visit: Optional[List[str]] = None,
    attractions: Optional[List[Dict[str, Any]]] = None,
    restaurants: Optional[List[Dict[str, Any]]] = None,
    hotels: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    从候选 POI 构建结构化逐日行程骨架（仅 JSON，不含自然语言攻略）。
    叙事润色由 ItineraryAgent / 聚合 LLM 完成。
    """
    d = max(1, int(days or 1))
    pref = norm_text(preferences)
    must = must_visit or []

    atts = [a for a in (attractions or []) if isinstance(a, dict) and norm_text(a.get("name"))]
    rests = [r for r in (restaurants or []) if isinstance(r, dict) and norm_text(r.get("name"))]
    hots = [h for h in (hotels or []) if isinstance(h, dict) and norm_text(h.get("name"))]

    if must:
        known = {norm_text(a.get("name")) for a in atts}
        for mv in must:
            if norm_text(mv) and norm_text(mv) not in known:
                atts.insert(0, {
                    "name": mv,
                    "type": "must_visit",
                    "source": "user_must_visit",
                })

    if not atts:
        return {
            "error": "no_attraction_candidates",
            "message": (
                "未获取到可用景点 POI，无法生成结构化行程；"
                "请确认地图 API 配置，或在 attraction_list 中传入景点。"
            ),
            "departure_city": departure_city,
            "destination_city": destination_city,
            "days": d,
            "preferences": pref,
            "must_visit": must,
            "data_source": "itinerary_builder",
            "candidates": {
                "attractions": [],
                "restaurants": [_poi_slot(r) for r in rests[:10] if _poi_slot(r)],
                "hotels": [_poi_slot(h) for h in hots[:5] if _poi_slot(h)],
            },
        }

    key = f"{departure_city}|{destination_city}|{d}|{pref}|{','.join(must)}"
    picked_atts = pick_many(atts, min(len(atts), max(3, min(12, d * 2 + 2))), key, "atts")
    picked_rests = pick_many(rests, min(len(rests), max(2, min(10, d + 2))), key, "rests") if rests else []

    days_out: List[Dict[str, Any]] = []
    for i in range(1, d + 1):
        a_m = picked_atts[(i - 1) % len(picked_atts)]
        a_a = picked_atts[i % len(picked_atts)]
        r_e = picked_rests[(i - 1) % len(picked_rests)] if picked_rests else None
        route_pois = [p for p in (a_m, a_a, r_e) if isinstance(p, dict)]
        route_plan = build_local_route_plan(route_pois)
        slots = [
            {"period": "morning", "category": "attraction", "poi": _poi_slot(a_m)},
            {"period": "afternoon", "category": "attraction", "poi": _poi_slot(a_a)},
        ]
        if r_e:
            slots.append({"period": "evening", "category": "dining", "poi": _poi_slot(r_e)})
        days_out.append({
            "day": i,
            "slots": slots,
            "local_route": route_plan,
        })

    return {
        "departure_city": departure_city,
        "destination_city": destination_city,
        "days": d,
        "preferences": pref,
        "must_visit": must,
        "data_source": "itinerary_builder",
        "transportation": {
            "outbound": {"from_city": departure_city, "to_city": destination_city},
            "local": {"suggested_modes": ["metro", "ride_hail", "walk", "bike_share"]},
            "return": {"from_city": destination_city, "to_city": departure_city},
        },
        "stay_suggestion": _poi_slot(hots[0]) if hots else None,
        "plan": days_out,
        "candidates": {
            "attractions": [_poi_slot(a) for a in picked_atts[:12] if _poi_slot(a)],
            "restaurants": [_poi_slot(r) for r in picked_rests[:10] if _poi_slot(r)],
            "hotels": [_poi_slot(h) for h in hots[:5] if _poi_slot(h)],
        },
    }


def build_multi_city_itinerary_from_context(
    *,
    departure_city: str,
    cities: List[str],
    dates: Optional[List[str]] = None,
    preferences: Optional[str] = None,
    weather_by_city_date: Optional[Dict[str, Dict[str, Any]]] = None,
    attractions_by_city: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    hotels_by_city: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    restaurants_by_city: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """
    从按城市/日期分组的上游结果生成多城市结构化行程。
    这里保持确定性，只生成可校验 JSON；自然语言润色交给聚合 LLM。
    """
    route = [normalize_city_name(c) for c in cities if norm_text(c)]
    route = list(dict.fromkeys(route))
    if not route:
        return {
            "error": "no_route_cities",
            "message": "未获取到多城市路线，无法生成结构化行程。",
            "data_source": "multi_city_itinerary_builder",
        }

    trip_dates = [str(d).strip() for d in (dates or []) if str(d).strip()]
    if not trip_dates:
        trip_dates = [f"day_{i + 1}" for i in range(len(route))]

    weather_by_city_date = weather_by_city_date or {}
    attractions_by_city = attractions_by_city or {}
    hotels_by_city = hotels_by_city or {}
    restaurants_by_city = restaurants_by_city or {}

    all_attractions = [
        a
        for items in attractions_by_city.values()
        for a in items
        if isinstance(a, dict) and norm_text(a.get("name"))
    ]
    if not all_attractions:
        return {
            "error": "no_attraction_candidates",
            "message": "未获取到可用景点 POI，无法生成多城市结构化行程。",
            "departure_city": departure_city,
            "cities": route,
            "dates": trip_dates,
            "preferences": norm_text(preferences),
            "data_source": "multi_city_itinerary_builder",
        }

    plan: List[Dict[str, Any]] = []
    for idx, date in enumerate(trip_dates):
        city = route[min(idx, len(route) - 1)]
        city_atts = [
            a for a in attractions_by_city.get(city, [])
            if isinstance(a, dict) and norm_text(a.get("name"))
        ] or all_attractions
        city_rests = [
            r for r in restaurants_by_city.get(city, [])
            if isinstance(r, dict) and norm_text(r.get("name"))
        ]
        city_hotels = [
            h for h in hotels_by_city.get(city, [])
            if isinstance(h, dict) and norm_text(h.get("name"))
        ]

        key = f"{departure_city}|{'-'.join(route)}|{date}|{norm_text(preferences)}"
        picked_atts = pick_many(city_atts, min(len(city_atts), 2), key, f"atts_{idx}")
        evening_rest = pick_many(city_rests, 1, key, f"rests_{idx}")[0] if city_rests else None
        stay = pick_many(city_hotels, 1, key, f"hotels_{idx}")[0] if city_hotels else None
        route_pois = [p for p in (*picked_atts, evening_rest) if isinstance(p, dict)]
        route_plan = build_local_route_plan(route_pois)

        slots = []
        if picked_atts:
            slots.append({"period": "morning", "category": "attraction", "poi": _poi_slot(picked_atts[0])})
        if len(picked_atts) > 1:
            slots.append({"period": "afternoon", "category": "attraction", "poi": _poi_slot(picked_atts[1])})
        if evening_rest:
            slots.append({"period": "evening", "category": "dining", "poi": _poi_slot(evening_rest)})

        plan.append({
            "day": idx + 1,
            "date": date,
            "city": city,
            "weather": weather_by_city_date.get(city, {}).get(date),
            "slots": slots,
            "local_route": route_plan,
            "stay_suggestion": _poi_slot(stay),
        })

    return {
        "departure_city": departure_city,
        "cities": route,
        "dates": trip_dates,
        "days": len(trip_dates),
        "preferences": norm_text(preferences),
        "data_source": "multi_city_itinerary_builder",
        "transportation": {
            "route": [{"from_city": departure_city if i == 0 else route[i - 1], "to_city": city} for i, city in enumerate(route)],
            "local": {"suggested_modes": ["metro", "ride_hail", "walk", "bike_share"]},
        },
        "plan": plan,
        "candidates": {
            "attractions_by_city": {
                city: [_poi_slot(a) for a in items[:8] if _poi_slot(a)]
                for city, items in attractions_by_city.items()
            },
            "hotels_by_city": {
                city: [_poi_slot(h) for h in items[:5] if _poi_slot(h)]
                for city, items in hotels_by_city.items()
            },
            "restaurants_by_city": {
                city: [_poi_slot(r) for r in items[:5] if _poi_slot(r)]
                for city, items in restaurants_by_city.items()
            },
        },
    }


async def fetch_restaurants_from_api(location: str, *, cuisine: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
    # 如果已配置则优先使用百度 Place API，否则回退到高德。
    ensure_project_dotenv_loaded()
    loc = norm_text(location)
    cuisine_kw = _normalize_restaurant_cuisine(loc, cuisine)
    core_kw = _city_core_district_keyword(loc)
    search_query = f"{core_kw} {cuisine_kw} 餐厅".strip()
    if norm_text(os.getenv("BAIDU_MAP_AK")):
        res = await baidu_place_v2_search(
            query=search_query,
            region=loc,
            scope=2,
            page_size=min(20, max(1, int(limit or 10)) * 2),
            filter_="industry_type:cater|sort_name:overall_rating|sort_rule:0",
        )
        if not res.get("error"):
            items = []
            for it in (res.get("results") or []):
                if not isinstance(it, dict):
                    continue
                poi = _baidu_result_to_common_poi(it)
                if _is_valid_restaurant_poi(poi):
                    items.append(poi)
                if len(items) >= max(1, int(limit or 10)):
                    break
            if items:
                return {"restaurants": items, "data_source": "baidu_place_v2", "search_query": search_query}
            return {
                "error": "no_valid_restaurants",
                "message": f"未找到有效餐饮 POI（query={search_query}）",
                "restaurants": [],
                "data_source": "baidu_place_v2",
            }

    # 高德回退方案
    kw2 = cuisine_kw or "餐厅"
    res2 = await amap_place_text_search(region=loc, keyword=f"{core_kw} {kw2}", limit=limit)
    if res2.get("error"):
        return res2
    items2 = []
    for p in (res2.get("pois") or []):
        if isinstance(p, dict):
            poi = _poi_to_restaurant(p)
            if _is_valid_restaurant_poi(poi):
                items2.append(poi)
    return {"restaurants": items2[: max(1, int(limit or 10))], "data_source": "amap_place_text"}


# 城市名 -> 主枢纽机场 IATA（无民航机场的城市映射至邻近枢纽）
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

_CITY_CORE_DISTRICT_NAMES: Dict[str, tuple] = {
    "上海": ("黄浦", "静安", "徐汇", "长宁", "虹口", "杨浦"),
    "苏州": ("姑苏", "吴中", "工业园"),
    "杭州": ("上城", "拱墅", "西湖"),
}

_CITY_FAR_DISTRICT_NAMES: Dict[str, tuple] = {
    "上海": ("崇明", "奉贤", "金山", "青浦", "嘉定", "临港", "横沙", "迪士尼"),
    "苏州": ("太仓", "张家港", "常熟", "昆山", "吴江", "同里", "西山", "太湖", "金庭"),
    "杭州": ("淳安", "建德", "临安", "富阳", "桐庐", "钱塘", "千岛湖", "界首", "凤坞"),
}

_SUBJECTIVE_HOTEL_PREF = re.compile(
    r"安静|吵闹|性价比|舒适|干净|卫生|便宜|奢华|亲子|早餐|贴心|便利|方便"
)

_RESTAURANT_REJECT_NAME = frozenset({"上海市", "苏州市", "杭州市", "南宁市"})
_RESTAURANT_REJECT_KW = ("棋牌", "足浴", "KTV", "洗浴", "养生会所", "休闲会所", "游戏场所")


def _normalize_restaurant_cuisine(location: str, cuisine: Optional[str]) -> str:
    loc = norm_text(location)
    c = norm_text(cuisine) or ""
    if any(x in c for x in ("江南", "时令", "本地", "江浙", "特色")):
        return _CITY_CUISINE_MAP.get(loc, "江浙菜")
    if c:
        return c
    return _CITY_CUISINE_MAP.get(loc, "餐厅")


def _city_core_district_keyword(location: str) -> str:
    loc = norm_text(location)
    return _CITY_CORE_DISTRICT.get(loc, loc)


def _hotel_location_text(poi: Dict[str, Any]) -> str:
    return f"{poi.get('district') or ''} {poi.get('address') or ''}"


def _hotel_in_far_suburb(poi: Dict[str, Any], city: str) -> bool:
    text = _hotel_location_text(poi)
    for far in _CITY_FAR_DISTRICT_NAMES.get(normalize_city_name(city), ()):
        if far in text:
            return True
    return False


def _hotel_in_core_district(poi: Dict[str, Any], city: str) -> bool:
    if _hotel_in_far_suburb(poi, city):
        return False
    text = _hotel_location_text(poi)
    core = _CITY_CORE_DISTRICT_NAMES.get(normalize_city_name(city), ())
    if not core:
        return True
    return any(c in text for c in core)


def build_hotel_search_query(city: str, preferences: Optional[str] = None) -> str:
    """地图 POI 搜索词：主观偏好（安静等）映射为核心城区 + 酒店。"""
    loc = normalize_city_name(city)
    core = _city_core_district_keyword(loc)
    pref = norm_text(preferences)
    if not pref or _SUBJECTIVE_HOTEL_PREF.search(pref):
        return f"{core} 酒店"
    if "酒店" in pref:
        return pref
    return f"{core} {pref} 酒店"


def _filter_hotels_for_city(
    hotels: List[Dict[str, Any]],
    city: str,
    *,
    strict_core: bool = True,
) -> List[Dict[str, Any]]:
    typed = [h for h in hotels if _is_hotel_poi(h)]
    pool = typed or hotels
    if strict_core:
        core_hotels = [h for h in pool if _hotel_in_core_district(h, city)]
        if core_hotels:
            return core_hotels
    return [h for h in pool if not _hotel_in_far_suburb(h, city)]


def _is_hotel_poi(poi: Dict[str, Any]) -> bool:
    name = norm_text(poi.get("name"))
    if not name:
        return False
    reject_name = ("游泳", "培训", "大楼", "棋牌", "足浴", "美容", "美发")
    if any(k in name for k in reject_name):
        return False
    typ = norm_text(poi.get("type")).lower()
    if typ and typ not in ("hotel", ""):
        return False
    raw = poi.get("raw") if isinstance(poi.get("raw"), dict) else {}
    detail = raw.get("detail_info") or {}
    tag = norm_text(detail.get("classified_poi_tag") or detail.get("tag") or "")
    if tag and not any(k in tag for k in ("酒店", "宾馆", "旅馆", "民宿", "客栈")):
        return False
    return any(k in name for k in ("酒店", "宾馆", "旅馆", "民宿", "客栈", "饭店"))


def _is_valid_restaurant_poi(poi: Dict[str, Any]) -> bool:
    name = norm_text(poi.get("name"))
    if not name or len(name) < 2:
        return False
    if name in _RESTAURANT_REJECT_NAME:
        return False
    if name.endswith("市") and len(name) <= 5:
        return False
    typ = norm_text(poi.get("type")).lower()
    if typ == "life":
        return False
    raw = poi.get("raw") or {}
    if isinstance(raw, dict):
        detail = raw.get("detail_info") or {}
        tag = norm_text(detail.get("classified_poi_tag") or detail.get("tag") or "")
        for kw in _RESTAURANT_REJECT_KW:
            if kw in name or kw in tag:
                return False
    return True


CITY_TO_IATA: Dict[str, str] = {
    "北京": "PEK",
    "上海": "PVG",
    "成都": "CTU",
    "重庆": "CKG",
    "广州": "CAN",
    "深圳": "SZX",
    "杭州": "HGH",
    "西安": "XIY",
    "昆明": "KMG",
    "武汉": "WUH",
    "南京": "NKG",
    "厦门": "XMN",
    "青岛": "TAO",
    "天津": "TSN",
    "长沙": "CSX",
    "郑州": "CGO",
    "沈阳": "SHE",
    "大连": "DLC",
    "哈尔滨": "HRB",
    "乌鲁木齐": "URC",
    "苏州": "WUX",
    "无锡": "WUX",
    "宁波": "NGB",
    "常州": "CZX",
    "温州": "WNZ",
    "珠海": "ZUH",
    "三亚": "SYX",
    "贵阳": "KWE",
    "南宁": "NNG",
    "福州": "FOC",
    "济南": "TNA",
    "合肥": "HFE",
    "南昌": "KHN",
    "海口": "HAK",
    "兰州": "LHW",
    "银川": "INC",
    "呼和浩特": "HET",
    "石家庄": "SJW",
    "太原": "TYN",
    "长春": "CGQ",
    "泉州": "JJN",
    "桂林": "KWL",
}


def _normalize_city_or_iata(value: str) -> str:
    return norm_text(value).replace("市", "").replace("机场", "").strip()


def resolve_city_to_iata(value: str) -> Optional[str]:
    """将城市名或已是 IATA 的输入解析为三字码；无法解析时返回 None。"""
    raw = _normalize_city_or_iata(value)
    if not raw:
        return None
    upper = raw.upper()
    if len(upper) == 3 and upper.isascii() and upper.isalpha():
        return upper
    return CITY_TO_IATA.get(raw)


async def fetch_flights_from_api(departure: str, arrival: str, date: str, *, limit: int = 10) -> Dict[str, Any]:
    """
    Aviationstack 航班端点。尽力而为：
    - 如果出发/到达看起来像 IATA 机场代码（3 个字母），则调用 API。
    - 否则回退（因为没有另一个数据集/API，城市->机场映射并不简单）。
    环境变量: AVIATIONSTACK_KEY
    """
    ensure_project_dotenv_loaded()
    key = norm_text(os.getenv("AVIATIONSTACK_KEY"))
    if not key:
        return {"error": "AVIATIONSTACK_KEY not set"}
    norm_date, derr = parse_exact_date(date)
    if derr:
        return {"error": derr}

    dep = resolve_city_to_iata(departure)
    arr = resolve_city_to_iata(arrival)

    if not dep or not arr:
        return {
            "error": "unable_to_resolve_airport_iata",
            "message": "请使用机场 IATA 三字码（如 PVG/PEK/CTU），或提供 CITY_TO_IATA 已收录的城市名。",
            "departure_input": departure,
            "arrival_input": arrival,
        }

    url = "https://api.aviationstack.com/v1/flights"
    data = await _http_get_json(
        url,
        params={
            "access_key": key,
            "flight_date": norm_date,
            "dep_iata": dep,
            "arr_iata": arr,
            "limit": str(max(1, min(int(limit or 10), 100))),
        },
    )
    rows = data.get("data") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return {"error": "invalid_flight_response", "raw": data}
    flights = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        flight = r.get("flight") or {}
        dep_obj = r.get("departure") or {}
        arr_obj = r.get("arrival") or {}
        flights.append(
            {
                "flight_no": (flight.get("iata") or flight.get("number") or "").strip(),
                "dep_time": (dep_obj.get("scheduled") or "")[-8:-3] if dep_obj.get("scheduled") else None,
                "arr_time": (arr_obj.get("scheduled") or "")[-8:-3] if arr_obj.get("scheduled") else None,
                "dep_iata": dep_obj.get("iata"),
                "arr_iata": arr_obj.get("iata"),
                "status": r.get("flight_status"),
                "airline": (r.get("airline") or {}).get("name"),
            }
        )
    return {"flights": flights, "data_source": "aviationstack", "raw_count": len(rows)}


async def fetch_flights_from_variflight_api(departure: str, arrival: str, date: str, *, limit: int = 10) -> Dict[str, Any]:
    """
    VariFlight (飞常准) HTTP API (MCP HTTP gateway):
    - POST https://mcp.variflight.com/api/v1/mcp/data
    - Header: X-VARIFLIGHT-KEY
    - Body: {"endpoint": "flights", "params": {"dep": "...", "arr": "...", "date": "YYYY-MM-DD"}}

    Env: X_VARIFLIGHT_KEY or VARIFLIGHT_API_KEY
    """
    ensure_project_dotenv_loaded()
    key = norm_text(os.getenv("X_VARIFLIGHT_KEY") or os.getenv("VARIFLIGHT_API_KEY"))
    if not key:
        return {"error": "VARIFLIGHT_API_KEY not set", "hint": "Set X_VARIFLIGHT_KEY or VARIFLIGHT_API_KEY in .env"}

    norm_date, derr = parse_exact_date(date)
    if derr:
        return {"error": derr}

    dep = resolve_city_to_iata(departure)
    arr = resolve_city_to_iata(arrival)

    if not dep or not arr:
        return {
            "error": "unable_to_resolve_airport_iata",
            "message": "请使用机场 IATA 三字码（如 PVG/PEK/CTU），或提供 CITY_TO_IATA 已收录的城市名。",
            "departure_input": departure,
            "arrival_input": arrival,
        }

    url = os.getenv("VARIFLIGHT_API_URL") or "https://mcp.variflight.com/api/v1/mcp/data"

    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
            r = await client.post(
                url,
                headers={"X-VARIFLIGHT-KEY": key, "Content-Type": "application/json"},
                json={"endpoint": "flights", "params": {"dep": dep, "arr": arr, "date": norm_date}},
            )
            r.raise_for_status()
            payload = r.json()
    except Exception as e:
        return {"error": "variflight_request_failed", "detail": str(e)}

    # Expected: {"code":200,"message":"Success","data":...}
    code = payload.get("code") if isinstance(payload, dict) else None
    if code not in (200, "200", None):
        return {"error": "variflight_error", "payload": payload}

    data = payload.get("data") if isinstance(payload, dict) else None
    flights_raw: Any = None
    # 启发式：data 可能是列表或包含在已知键下的列表的字典
    if isinstance(data, list):
        flights_raw = data
    elif isinstance(data, dict):
        for k in ["flights", "list", "data", "items", "result"]:
            v = data.get(k)
            if isinstance(v, list):
                flights_raw = v
                break
        if flights_raw is None:
            # 有时是按航班号键控的字典
            flights_raw = data

    flights: List[Dict[str, Any]] = []
    if isinstance(flights_raw, list):
        for item in flights_raw[: max(1, int(limit or 10))]:
            if not isinstance(item, dict):
                continue
            # 尽可能保持紧凑的通用结构，但保留原始数据
            flights.append(
                {
                    "flight_no": item.get("fnum") or item.get("flightNo") or item.get("FlightNo") or item.get("flight_no") or item.get("flight"),
                    "dep_time": item.get("dep_time") or item.get("FlightDeptimePlanDate") or item.get("deptimestd") or item.get("std"),
                    "arr_time": item.get("arr_time") or item.get("FlightArrtimePlanDate") or item.get("arrtimestd") or item.get("sta"),
                    "dep_iata": item.get("dep") or item.get("FlightDepcode") or dep,
                    "arr_iata": item.get("arr") or item.get("FlightArrcode") or arr,
                    "status": item.get("status") or item.get("FlightState") or item.get("state"),
                    "airline": item.get("airline") or item.get("FlightCompany") or item.get("airlineName"),
                    "raw": item,
                }
            )

    return {
        "departure": dep,
        "arrival": arr,
        "date": norm_date,
        "flights": flights,
        "raw": payload,
        "data_source": "variflight_http_api",
    }


async def fetch_news_from_api(headline: str, *, limit: int = 10) -> Dict[str, Any]:
    """
    GDELT Doc 2.0（无需密钥）：按查询搜索文章。
    """
    q = norm_text(headline)
    if not q:
        return {"error": "headline is required"}
    # 搜索过去 7 天以保持结果相关性
    end = datetime.utcnow()
    start = end - timedelta(days=7)
    start_dt = start.strftime("%Y%m%d%H%M%S")
    end_dt = end.strftime("%Y%m%d%H%M%S")
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    data = await _http_get_json(
        url,
        params={
            "query": q,
            "mode": "artlist",
            "format": "json",
            "maxrecords": str(max(1, min(int(limit or 10), 250))),
            "startdatetime": start_dt,
            "enddatetime": end_dt,
            "sort": "hybridrel",
        },
    )
    articles = (data.get("articles") or []) if isinstance(data, dict) else []
    out = []
    for a in articles[: max(1, int(limit or 10))]:
        if not isinstance(a, dict):
            continue
        out.append(
            {
                "source": a.get("sourceCountry") or a.get("sourcecountry") or a.get("domain"),
                "title": a.get("title"),
                "summary": a.get("summary") or "",
                "url": a.get("url"),
                "seendate": a.get("seendate"),
                "domain": a.get("domain"),
            }
        )
    return {"articles": out, "data_source": "gdelt_doc2", "raw_count": len(articles)}


def _strip_html_tags(s: str) -> str:
    """用于片段/标题的最小化标签剥离器。"""
    import re as _re

    if not s:
        return ""
    s = _re.sub(r"<script[\s\S]*?</script>", "", s, flags=_re.I)
    s = _re.sub(r"<style[\s\S]*?</style>", "", s, flags=_re.I)
    s = _re.sub(r"<[^>]+>", "", s)
    s = s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'")
    return " ".join(s.split())


def _extract_paragraph_text(html: str, *, max_chars: int = 1600) -> str:
    """
    无需额外依赖的最佳努力文章文本提取。
    策略：收集 <p> 块；如果太少，则回退到剥离文本。
    """
    import re as _re
    from html.parser import HTMLParser

    if not html:
        return ""
    html = _re.sub(r"<script[\s\S]*?</script>", "", html, flags=_re.I)
    html = _re.sub(r"<style[\s\S]*?</style>", "", html, flags=_re.I)

    class _P(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self._in_p = False
            self._buf: List[str] = []
            self.paras: List[str] = []

        def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
            if tag.lower() == "p":
                self._in_p = True
                self._buf = []

        def handle_endtag(self, tag: str) -> None:
            if tag.lower() == "p" and self._in_p:
                txt = " ".join("".join(self._buf).split())
                txt = txt.replace("\u3000", " ").strip()
                if txt:
                    self.paras.append(txt)
                self._in_p = False
                self._buf = []

        def handle_data(self, data: str) -> None:
            if self._in_p and data:
                self._buf.append(data)

    p = _P()
    try:
        p.feed(html)
    except Exception:
        pass

    text = ""
    if len(p.paras) >= 2:
        text = "\n\n".join(p.paras)
    else:
        text = _strip_html_tags(html)

    text = " ".join(text.split()) if "\n\n" not in text else text
    if max_chars and len(text) > max_chars:
        text = text[: max_chars].rstrip() + "..."
    return text


async def fetch_news_from_baidu_html(query: str, *, limit: int = 10) -> Dict[str, Any]:
    """
    通过抓取 HTML 进行百度新闻垂直搜索。
    端点 (HTML): https://www.baidu.com/s?tn=news&word=...
    注意：这不是官方 API，可能会受到速率限制/验证码保护。
    """
    q = norm_text(query)
    if not q:
        return {"error": "query is required"}

    url = "https://www.baidu.com/s"
    params = {"tn": "news", "word": q}
    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.7",
        "referer": "https://www.baidu.com/",
    }

    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, follow_redirects=True, headers=headers) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        html = r.text or ""

    low = html.lower()
    if ("验证码" in html) or ("安全验证" in html) or ("captcha" in low and "verify" in low):
        return {
            "error": "baidu_captcha",
            "message": "百度返回了安全验证/验证码页面（需要降低频率或回退到 GDELT/RSS）。",
        }

    # 提取结果。DOM 可能会变化。
    import re as _re

    title_pat = _re.compile(r"<h3[^>]*>(?P<h3>[\s\S]*?)</h3>", flags=_re.I)
    a_pat = _re.compile(r"<a[^>]*href=\"(?P<href>[^\"]+)\"[^>]*>(?P<title>[\s\S]*?)</a>", flags=_re.I)
    abs_pat = _re.compile(r"c-abstract[^>]*>(?P<abs>[\s\S]*?)</", flags=_re.I)

    # 收集更多候选，然后在内容获取/验证后过滤到 `limit` 数量。
    candidates: List[Dict[str, Any]] = []
    for m in title_pat.finditer(html):
        h3 = m.group("h3") or ""
        am = a_pat.search(h3)
        if not am:
            continue
        href = (am.group("href") or "").strip()
        title = _strip_html_tags(am.group("title") or "")
        if not title or not href:
            continue

        window = html[m.end() : m.end() + 2500]
        snip_m = abs_pat.search(window)
        snippet = _strip_html_tags(snip_m.group("abs") if snip_m else "")

        candidates.append(
            {
                "source": "baidu",
                "title": title,
                "summary": snippet or "",
                "url": href,
            }
        )
        # 收集额外数据以抵消被过滤/阻止的页面。
        if len(candidates) >= max(10, int(limit or 10) * 4):
            break

    if not candidates:
        return {"error": "no_results", "message": "未解析到结果（可能 DOM 变化或被拦截）"}

    def _is_bad_article(url_s: str, content_s: str) -> bool:
        u = (url_s or "").casefold()
        c = (content_s or "").casefold()
        # 明显的纯视频页面
        if "haokan.baidu.com" in u or "video" in u:
            return True
        # captcha / safe verify / JS-required / blocked
        bad_phrases = [
            "安全验证",
            "百度安全验证",
            "验证码",
            "网络不给力，请稍后重试",
            "enable javascript",
            "doesn't work properly without javascript",
            "verify that you're not a robot",
            "返回首页",
        ]
        if any(p.casefold() in c for p in bad_phrases):
            return True
        # 太短/非文章
        if len(_strip_html_tags(content_s)) < 80:
            return True
        return False

    needed = max(1, int(limit or 10))
    good: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, follow_redirects=True, headers=headers) as client:
        for cand in candidates:
            if len(good) >= needed:
                break
            u = norm_text(cand.get("url"))
            if not u:
                continue
            body = ""
            try:
                rr = await client.get(u)
                rr.raise_for_status()
                body = _extract_paragraph_text(rr.text or "", max_chars=1600)
            except Exception:
                body = ""
            # 如果正文为空则回退到摘要
            content = body or norm_text(cand.get("summary"))
            if _is_bad_article(u, content):
                continue
            item = dict(cand)
            item["content"] = content
            good.append(item)

    if not good:
        return {"error": "no_good_results", "message": "结果均被拦截/无正文/为视频页，建议换关键词或回退 GDELT"}
    return {"articles": good[:needed], "data_source": "baidu_news_html", "raw_count": len(good)}


async def fetch_movies_now_playing_from_api(*, limit: int = 10, language: str = "zh-CN", region: str = "CN") -> Dict[str, Any]:
    """
    TMDB 正在上映的电影。
    环境变量: TMDB_BEARER_TOKEN (v4 访问令牌)
    """
    ensure_project_dotenv_loaded()
    token = norm_text(os.getenv("TMDB_BEARER_TOKEN"))
    if not token:
        return {"error": "TMDB_BEARER_TOKEN not set"}
    if token.lower().startswith("bearer "):
        token = token.split(" ", 1)[1].strip()
        if not token:
            return {"error": "TMDB_BEARER_TOKEN is empty after stripping 'Bearer ' prefix"}
    url = "https://api.themoviedb.org/3/movie/now_playing"
    data = await _http_get_json(
        url,
        params={"language": language, "region": region, "page": "1"},
        headers={"Authorization": f"Bearer {token}", "accept": "application/json"},
    )
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        return {"error": "invalid_tmdb_response", "raw": data}
    movies = []
    for m in results[: max(1, int(limit or 10))]:
        if not isinstance(m, dict):
            continue
        movies.append(
            {
                "title": m.get("title") or m.get("original_title"),  # 标题或原始标题
                "release_date": m.get("release_date"),
                "overview": m.get("overview"),
                "popularity": m.get("popularity"),
                "vote_average": m.get("vote_average"),
            }
        )
    return {"movies": movies, "data_source": "tmdb_now_playing", "raw_count": len(results)}

