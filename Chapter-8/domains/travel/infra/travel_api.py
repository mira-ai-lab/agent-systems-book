import hashlib
import os
import re
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field


def norm_text(s: Optional[str]) -> str:
    return (s or "").strip()


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
_DOTENV_LOADED = False


def _project_root_dir() -> str:
    from agent_framework.config import BOOK_ROOT

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
    # 如果已配置则优先使用百度 Place API，否则回退到高德。
    # keyword: 区域/地标偏好，如「黄浦区」「外滩」；会拼进 POI 搜索词（默认仅「酒店」）
    ensure_project_dotenv_loaded()
    q = norm_text(keyword) or "酒店"
    if "酒店" not in q:
        q = f"{q} 酒店"
    page_n = max(1, min(int(limit or 10), 20))
    if norm_text(os.getenv("BAIDU_MAP_AK")):
        res = await baidu_place_v2_search(
            query=q,
            region=city,
            scope=2,
            page_size=page_n,
            filter_="industry_type:hotel|sort_name:total_score|sort_rule:0",
        )
        if not res.get("error"):
            hotels = []
            for it in (res.get("results") or [])[:page_n]:
                if isinstance(it, dict):
                    hotels.append(_baidu_result_to_common_poi(it))
            return {"hotels": hotels, "data_source": "baidu_place_v2", "search_query": q}
    # 关键字搜索: city + 酒店 (高德)
    res2 = await amap_place_text_search(region=city, keyword=q, limit=limit)
    if res2.get("error"):
        return res2
    hotels2 = []
    for p in (res2.get("pois") or [])[: max(1, int(limit or 10))]:
        if isinstance(p, dict):
            hotels2.append(_poi_to_hotel(p))
    return {"hotels": hotels2, "data_source": "amap_place_text", "search_query": q}


async def fetch_attractions_from_api(city: str, *, limit: int = 10) -> Dict[str, Any]:
    """
    获取用于行程规划的景点/POI 候选列表。
    如果已配置则优先使用百度 Place v2，否则回退到高德。
    """
    ensure_project_dotenv_loaded()
    if norm_text(os.getenv("BAIDU_MAP_AK")):
        res = await baidu_place_v2_search(
            query="景点",
            region=city,
            scope=2,
            page_size=min(20, max(1, int(limit or 10))),
            filter_="industry_type:life|sort_name:overall_rating|sort_rule:0",
        )
        if not res.get("error"):
            items = []
            for it in (res.get("results") or [])[: max(1, int(limit or 10))]:
                if isinstance(it, dict):
                    poi = _baidu_result_to_common_poi(it)
                    # 减少下游 LLM prompt 的负载大小
                    poi.pop("raw", None)
                    items.append(poi)
            return {"attractions": items, "data_source": "baidu_place_v2"}

    # 高德回退方案
    res2 = await amap_place_text_search(region=city, keyword="景点", limit=limit)
    if res2.get("error"):
        return res2
    items2 = []
    for p in (res2.get("pois") or [])[: max(1, int(limit or 10))]:
        if isinstance(p, dict):
            items2.append(
                {
                    "name": p.get("name"),
                    "district": p.get("adname") or p.get("address"),
                    "address": p.get("address"),
                    "tel": p.get("tel"),
                    "location": p.get("location"),
                    "rating": (p.get("business") or {}).get("rating") if isinstance(p.get("business"), dict) else None,
                    "avg_price_cny": (p.get("business") or {}).get("cost") if isinstance(p.get("business"), dict) else None,
                    "type": p.get("type"),
                }
            )
    return {"attractions": items2, "data_source": "amap_place_text"}


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
    从候选 POI 中确定性构建逐日行程骨架。
    LLM 随后可以润色措辞/添加提示。
    """
    d = max(1, int(days or 1))
    pref = norm_text(preferences)
    must = must_visit or []

    atts = [a for a in (attractions or []) if isinstance(a, dict) and norm_text(a.get("name"))]
    rests = [r for r in (restaurants or []) if isinstance(r, dict) and norm_text(r.get("name"))]
    hots = [h for h in (hotels or []) if isinstance(h, dict) and norm_text(h.get("name"))]

    # 如果必去景点不在列表中，确保将其作为伪景点加入
    if must:
        known = {norm_text(a.get("name")) for a in atts}
        for mv in must:
            if norm_text(mv) and norm_text(mv) not in known:
                atts.insert(0, {"name": mv, "address": None, "district": None, "location": None, "rating": None, "avg_price_cny": None, "type": "must_visit"})

    key = f"{departure_city}|{destination_city}|{d}|{pref}|{','.join(must)}"
    if not atts:
        # 回退到之前的模拟行为
        base = build_itinerary(departure_city, destination_city, d, preferences=pref, must_visit=must)
        base["data_source"] = "stub(mafengwo_guides)"
        return base

    picked_atts = pick_many(atts, min(len(atts), max(3, min(12, d * 2 + 2))), key, "atts")
    picked_rests = pick_many(rests, min(len(rests), max(2, min(10, d + 2))), key, "rests") if rests else []
    # 缩小负载以供下游 LLM 使用
    for x in picked_atts:
        if isinstance(x, dict):
            x.pop("raw", None)
    for x in picked_rests:
        if isinstance(x, dict):
            x.pop("raw", None)
    for x in hots:
        if isinstance(x, dict):
            x.pop("raw", None)

    days_out: List[Dict[str, Any]] = []
    for i in range(1, d + 1):
        a_m = picked_atts[(i - 1) % len(picked_atts)]
        a_a = picked_atts[(i) % len(picked_atts)]
        r_e = picked_rests[(i - 1) % len(picked_rests)] if picked_rests else None
        evening = f"晚餐：{r_e.get('name')}（可订位） + 夜景/夜市" if isinstance(r_e, dict) else "晚餐：本地热门餐厅 + 夜景/夜市"
        days_out.append(
            {
                "day": i,
                "morning": f"{a_m.get('name')}（建议早到避开人流）",
                "afternoon": f"{a_a.get('name')}（结合交通距离调整顺序）",
                "evening": evening,
                "notes": "市内交通优先地铁/打车；热门景点建议提前预约；每晚按体力留出机动时间。",
            }
        )

    transport = {
        "outbound": f"{departure_city} → {destination_city}：优先高铁/飞机（按预算与时长选择）",
        "local": "市内交通：地铁优先；景区可用网约车/公交；步行与共享单车适合短距离。",
        "return": f"{destination_city} → {departure_city}：建议预留 2-3 小时到站/到机场时间。",
    }
    stay = None
    if hots:
        stay = hots[0]
    return {
        "departure_city": departure_city,
        "destination_city": destination_city,
        "days": d,
        "preferences": pref,
        "must_visit": must,
        "transportation": transport,
        "stay_suggestion": stay,
        "plan": days_out,
        "candidates": {
            "attractions": picked_atts[: min(len(picked_atts), 12)],
            "restaurants": picked_rests[: min(len(picked_rests), 10)],
            "hotels": hots[: min(len(hots), 5)],
        },
    }


async def fetch_restaurants_from_api(location: str, *, cuisine: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
    # 如果已配置则优先使用百度 Place API，否则回退到高德。
    ensure_project_dotenv_loaded()
    kw = norm_text(cuisine) or "美食"
    if norm_text(os.getenv("BAIDU_MAP_AK")):
        res = await baidu_place_v2_search(
            query=kw,
            region=location,
            scope=2,
            page_size=min(20, max(1, int(limit or 10))),
            filter_="industry_type:cater|sort_name:overall_rating|sort_rule:0",
        )
        if not res.get("error"):
            items = []
            for it in (res.get("results") or [])[: max(1, int(limit or 10))]:
                if isinstance(it, dict):
                    items.append(_baidu_result_to_common_poi(it))
            return {"restaurants": items, "data_source": "baidu_place_v2"}

    # 高德回退方案
    kw2 = "餐厅"
    if norm_text(cuisine):
        kw2 = f"{norm_text(cuisine)}"
    res2 = await amap_place_text_search(region=location, keyword=kw2, limit=limit)
    if res2.get("error"):
        return res2
    items2 = []
    for p in (res2.get("pois") or [])[: max(1, int(limit or 10))]:
        if isinstance(p, dict):
            items2.append(_poi_to_restaurant(p))
    return {"restaurants": items2, "data_source": "amap_place_text"}


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
    def _normalize_city_or_iata(x: str) -> str:
        return norm_text(x).replace("市", "").replace("机场", "").strip()

    CITY_TO_IATA = {
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
    }

    dep_raw = _normalize_city_or_iata(departure)
    arr_raw = _normalize_city_or_iata(arrival)
    dep = dep_raw.upper()
    arr = arr_raw.upper()

    # 如果用户提供城市名，将其映射到 IATA 机场代码（尽力而为）
    if not (len(dep) == 3 and dep.isalpha()):
        dep = CITY_TO_IATA.get(dep_raw, "")
    if not (len(arr) == 3 and arr.isalpha()):
        arr = CITY_TO_IATA.get(arr_raw, "")

    if not dep or not arr:
        return {
            "error": "unable_to_resolve_airport_iata",
            "message": "请使用机场 IATA 三字码（如 PVG/PEK/CTU），或使用常见城市名（如 上海/北京/成都/广州/深圳）。",
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

    def _normalize_city_or_iata(x: str) -> str:
        return norm_text(x).replace("市", "").replace("机场", "").strip()

    # 重用常见的中国城市名 -> 机场 IATA 映射
    CITY_TO_IATA = {
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
    }

    dep_raw = _normalize_city_or_iata(departure)
    arr_raw = _normalize_city_or_iata(arrival)
    dep = dep_raw.upper()
    arr = arr_raw.upper()
    if not (len(dep) == 3 and dep.isalpha()):
        dep = (CITY_TO_IATA.get(dep_raw) or "").upper()
    if not (len(arr) == 3 and arr.isalpha()):
        arr = (CITY_TO_IATA.get(arr_raw) or "").upper()

    if not dep or not arr:
        return {
            "error": "unable_to_resolve_airport_iata",
            "message": "请使用机场 IATA 三字码（如 PVG/PEK/CTU），或使用常见城市名（如 上海/北京/成都/广州/深圳）。",
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


@dataclass
class ItineraryDay:
    day: int
    morning: str
    afternoon: str
    evening: str
    notes: str = ""


def build_itinerary(
    departure_city: str,
    destination_city: str,
    days: int,
    preferences: Optional[str] = None,
    must_visit: Optional[List[str]] = None,
) -> Dict[str, Any]:
    d = max(1, int(days or 1))
    pref = norm_text(preferences)
    must = must_visit or []
    key = f"{departure_city}|{destination_city}|{d}|{pref}|{','.join(must)}"
    attractions = [
        "城市地标打卡",
        "博物馆/展览",
        "老街慢逛",
        "城市公园",
        "本地美食街",
        "周边一日游",
        "夜景观景点",
    ]
    if must:
        attractions = must + [a for a in attractions if a not in must]
    picks = pick_many([{"name": a} for a in attractions], n=min(7, max(3, d + 2)), *[key])
    days_out: List[Dict[str, Any]] = []
    for i in range(1, d + 1):
        a1 = picks[(i - 1) % len(picks)]["name"]
        a2 = picks[(i) % len(picks)]["name"]
        a3 = picks[(i + 1) % len(picks)]["name"]
        days_out.append(
            {
                "day": i,
                "morning": f"{a1}（轻量行程）",
                "afternoon": f"{a2}（根据人流调整）",
                "evening": f"{a3} + 夜市/夜景",
                "notes": "优先选择地铁/打车；热门景点建议提前预约。",
            }
        )
    transport = {
        "outbound": f"{departure_city} → {destination_city}：优先高铁/飞机（按预算与时长选择）",
        "local": "市内交通：地铁优先；景区可用网约车/公交；步行与共享单车适合短距离。",
        "return": f"{destination_city} → {departure_city}：建议预留 2-3 小时到站/到机场时间。",
    }
    return {
        "departure_city": departure_city,
        "destination_city": destination_city,
        "days": d,
        "preferences": pref,
        "must_visit": must,
        "transportation": transport,
        "plan": days_out,
    }

