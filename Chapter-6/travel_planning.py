"""Shared travel-planning primitives for Chapter 6.

This module has one scope: turn parsed dates and validated attraction
candidates into deterministic route and itinerary structures. It does not own
POI search, tool-result quality, agent execution, or final answer aggregation.

:author: zhanghong, liuhui
:date: 2026-11-17
"""

import hashlib
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from config import CH5_DIR
from llm import invoke_structured
from map_api import (
    baidu_direction_route,
    haversine_m,
    normalize_city_name,
    parse_lng_lat,
    select_baidu_direction_mode,
)

if not CH5_DIR.is_dir():
    raise FileNotFoundError(f"Could not locate {CH5_DIR}")


def parse_exact_date(date_str: str) -> Tuple[Optional[str], Optional[str]]:
    """
    强制使用精确日期格式 YYYY-MM-DD。
    返回 (标准化日期, 错误信息)。
    """
    s = (date_str or "").strip()
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
    s = (date_str or "").strip()
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
        prompt = _DATE_EXTRACT_PROMPT.format(today=today, query=query)
        result: TripDateExtraction = await invoke_structured(
            llm,
            prompt,
            schema=TripDateExtraction,
        )

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

    Args:
        query: Original user request containing explicit or relative dates.
        ref: Optional reference time used to resolve relative expressions.
        llm: Optional model used for structured date extraction.

    Returns:
        A dictionary containing today, trip dates, the date range, and the
        prompt anchor block.

    When no model is supplied or extraction fails, the function falls back to
    the local regular-expression resolver.
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
    """Build a deterministic integer from ordered text fragments."""
    raw = "|".join([(part or "").strip() for part in parts]).encode("utf-8")
    h = hashlib.sha256(raw).hexdigest()
    return int(h[:8], 16)


def pick_many(items: Sequence[Dict[str, Any]], n: int, *seed_parts: str) -> List[Dict[str, Any]]:
    """Select up to ``n`` items in deterministic order."""
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

def _poi_slot(poi: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """行程 slot 用 POI 摘要（不含 raw 等大字段）。"""
    if not isinstance(poi, dict) or not str(poi.get("name") or "").strip():
        return None
    return {
        k: poi.get(k)
        for k in (
            "name", "address", "district", "location", "rating",
            "avg_price_cny", "type", "tel", "cuisine",
        )
        if poi.get(k) is not None
    }


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
        if not isinstance(poi, dict) or not str(poi.get("name") or "").strip():
            continue
        coord = parse_lng_lat(poi.get("location"))
        if not coord:
            skipped.append(str(poi.get("name") or "").strip())
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
            key=lambda i: haversine_m(current, valid[i]["_coord"]),
        )
        ordered.append(valid.pop(next_idx))

    segments = []
    total_distance = 0
    total_duration = 0
    for prev, curr in zip(ordered, ordered[1:]):
        distance = haversine_m(prev["_coord"], curr["_coord"])
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
        if isinstance(poi, dict) and str(poi.get("name") or "").strip():
            coord = parse_lng_lat(poi.get("location"))
            if coord:
                coord_by_name[str(poi.get("name") or "").strip()] = coord

    baidu_segments = []
    total_distance = 0
    total_duration = 0
    used_baidu = False
    for segment in estimate.get("segments") or []:
        from_name = str(segment.get("from") or "").strip()
        to_name = str(segment.get("to") or "").strip()
        origin = coord_by_name.get(from_name)
        destination = coord_by_name.get(to_name)
        if not origin or not destination:
            baidu_segments.append(segment)
            continue

        estimated_distance = int(
            segment.get("distance_m")
            or haversine_m(origin, destination)
        )
        api_mode = (
            select_baidu_direction_mode(estimated_distance)
            if mode == "auto"
            else mode
        )
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
    attractions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    从候选 POI 构建结构化逐日行程骨架（仅 JSON，不含自然语言攻略）。
    叙事润色由行程领域智能体 / 聚合 LLM 完成。
    """
    d = max(1, int(days or 1))
    pref = (preferences or "").strip()

    atts = [
        attraction
        for attraction in (attractions or [])
        if isinstance(attraction, dict)
        and str(attraction.get("name") or "").strip()
    ]

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
            "data_source": "itinerary_builder",
            "candidates": {"attractions": []},
        }

    key = f"{departure_city}|{destination_city}|{d}|{pref}"
    picked_atts = pick_many(atts, min(len(atts), max(3, min(12, d * 2 + 2))), key, "atts")

    days_out: List[Dict[str, Any]] = []
    for i in range(1, d + 1):
        a_m = picked_atts[(i - 1) % len(picked_atts)]
        a_a = picked_atts[i % len(picked_atts)]
        route_pois = [p for p in (a_m, a_a) if isinstance(p, dict)]
        route_plan = build_local_route_plan(route_pois)
        slots = [
            {"period": "morning", "category": "attraction", "poi": _poi_slot(a_m)},
            {"period": "afternoon", "category": "attraction", "poi": _poi_slot(a_a)},
        ]
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
        "data_source": "itinerary_builder",
        "transportation": {
            "outbound": {"from_city": departure_city, "to_city": destination_city},
            "local": {"suggested_modes": ["metro", "ride_hail", "walk", "bike_share"]},
            "return": {"from_city": destination_city, "to_city": departure_city},
        },
        "plan": days_out,
        "candidates": {
            "attractions": [_poi_slot(a) for a in picked_atts[:12] if _poi_slot(a)],
        },
    }


def build_multi_city_itinerary_from_context(
    *,
    departure_city: str,
    cities: List[str],
    dates: Optional[List[str]] = None,
    preferences: Optional[str] = None,
    attractions_by_city: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """
    从按城市/日期分组的上游结果生成多城市结构化行程。
    这里保持确定性，只生成可校验 JSON；自然语言润色交给聚合 LLM。
    """
    route = [
        normalize_city_name(city)
        for city in cities
        if (city or "").strip()
    ]
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

    attractions_by_city = attractions_by_city or {}

    all_attractions = [
        a
        for items in attractions_by_city.values()
        for a in items
        if isinstance(a, dict) and str(a.get("name") or "").strip()
    ]
    if not all_attractions:
        return {
            "error": "no_attraction_candidates",
            "message": "未获取到可用景点 POI，无法生成多城市结构化行程。",
            "departure_city": departure_city,
            "cities": route,
            "dates": trip_dates,
            "preferences": (preferences or "").strip(),
            "data_source": "multi_city_itinerary_builder",
        }

    plan: List[Dict[str, Any]] = []
    for idx, date in enumerate(trip_dates):
        city = route[min(idx, len(route) - 1)]
        city_atts = [
            a for a in attractions_by_city.get(city, [])
            if isinstance(a, dict) and str(a.get("name") or "").strip()
        ] or all_attractions

        key = (
            f"{departure_city}|{'-'.join(route)}|{date}|"
            f"{(preferences or '').strip()}"
        )
        picked_atts = pick_many(city_atts, min(len(city_atts), 2), key, f"atts_{idx}")
        route_pois = [p for p in picked_atts if isinstance(p, dict)]
        route_plan = build_local_route_plan(route_pois)

        slots = []
        if picked_atts:
            slots.append({"period": "morning", "category": "attraction", "poi": _poi_slot(picked_atts[0])})
        if len(picked_atts) > 1:
            slots.append({"period": "afternoon", "category": "attraction", "poi": _poi_slot(picked_atts[1])})

        plan.append({
            "day": idx + 1,
            "date": date,
            "city": city,
            "slots": slots,
            "local_route": route_plan,
        })

    return {
        "departure_city": departure_city,
        "cities": route,
        "dates": trip_dates,
        "days": len(trip_dates),
        "preferences": (preferences or "").strip(),
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
        },
    }
