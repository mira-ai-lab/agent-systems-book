"""聚合阶段辅助：单任务直达 vs 多任务 LLM 综合"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from prompts import AGGREGATION_PROMPT
from travel_common import (
    _is_valid_attraction_poi,
    _is_valid_restaurant_poi,
    norm_text,
    normalize_city_name,
)

_ADMIN_DIVISION_SUFFIXES = ("特别行政区", "自治州", "地区", "盟", "市", "省", "县", "区")


def is_single_direct_response(results: Dict[str, Any]) -> bool:
    """仅一个子任务时，直接返回子智能体结果，避免被旅行规划模板扩写"""
    return len(results) == 1


def direct_response_from_results(results: Dict[str, Any]) -> str:
    """从单个子任务结果提取用户可读回复"""
    res = next(iter(results.values()))
    summary = (res.get("agent_summary") or "").strip()
    if summary:
        return summary
    tool_data = res.get("tool_data")
    if tool_data is not None:
        return json.dumps(tool_data, ensure_ascii=False, indent=2)
    return json.dumps(res, ensure_ascii=False, indent=2)


def _is_admin_division_name(name: str) -> bool:
    """识别“上海市/黄浦区”这类行政区划名，避免当成真实 POI。"""
    n = norm_text(name)
    for suffix in _ADMIN_DIVISION_SUFFIXES:
        if n.endswith(suffix) and len(n) > len(suffix):
            stem = n[: -len(suffix)]
            return 1 <= len(stem) <= 4
    return False


def _poi_name_invalid(name: Optional[str], *, expected_city: Optional[str] = None) -> bool:
    n = norm_text(name)
    if not n or len(n) < 2:
        return True
    if expected_city and normalize_city_name(n) == normalize_city_name(expected_city):
        return True
    if _is_admin_division_name(n):
        return True
    return False


def assess_tool_data_quality(agent: str, tool_data: Any) -> str:
    """返回 ok | error | invalid_poi"""
    if tool_data is None:
        return "error"
    if not isinstance(tool_data, dict):
        return "error"
    if tool_data.get("error"):
        return "error"

    if agent == "WeatherAgent":
        return "ok" if (tool_data.get("forecast") or tool_data.get("forecasts")) else "error"

    if agent == "RestaurantAgent":
        expected_city = tool_data.get("city") or tool_data.get("location")
        rests = tool_data.get("restaurants") or []
        valid = [
            r for r in rests
            if isinstance(r, dict) and not _restaurant_poi_invalid(r, expected_city=expected_city)
        ]
        return "ok" if valid else "invalid_poi"

    if agent == "HotelAgent":
        expected_city = tool_data.get("city")
        hotels = tool_data.get("hotels") or []
        valid = [
            h for h in hotels
            if isinstance(h, dict) and not _poi_name_invalid(h.get("name"), expected_city=expected_city)
        ]
        return "ok" if valid else "invalid_poi"

    if agent == "AttractionAgent":
        expected_city = tool_data.get("city")
        atts = tool_data.get("attractions") or tool_data.get("attraction_list") or []
        valid = [
            a for a in atts
            if (
                isinstance(a, dict)
                and not _poi_name_invalid(a.get("name"), expected_city=expected_city)
                and _is_valid_attraction_poi(a)
            )
        ]
        return "ok" if valid else "invalid_poi"

    if agent == "ItineraryAgent":
        if tool_data.get("error"):
            return "error"
        return "ok" if tool_data.get("plan") else "error"

    return "ok"


def _restaurant_poi_invalid(poi: Dict[str, Any], *, expected_city: Optional[str] = None) -> bool:
    if _poi_name_invalid(poi.get("name"), expected_city=expected_city):
        return True
    return not _is_valid_restaurant_poi(poi)


def extract_upstream_for_itinerary(prior_results: Dict[str, Any]) -> Dict[str, Any]:
    """从 T1–Tn 的 tool_data 汇总供 ItineraryAgent 使用的结构化上下文。"""
    weather_lines: List[str] = []
    attractions: List[Dict[str, Any]] = []
    hotels: List[Dict[str, Any]] = []
    restaurants: List[Dict[str, Any]] = []
    weather_by_city_date: Dict[str, Dict[str, Any]] = {}
    attractions_by_city: Dict[str, List[Dict[str, Any]]] = {}
    hotels_by_city: Dict[str, List[Dict[str, Any]]] = {}
    restaurants_by_city: Dict[str, List[Dict[str, Any]]] = {}

    def _append_city_item(
        grouped: Dict[str, List[Dict[str, Any]]],
        city: Optional[str],
        item: Dict[str, Any],
    ) -> None:
        city_key = normalize_city_name(city)
        if not city_key:
            return
        grouped.setdefault(city_key, []).append(item)

    for _tid, res in prior_results.items():
        if not isinstance(res, dict):
            continue
        agent = res.get("agent") or ""
        td = res.get("tool_data")
        if not isinstance(td, dict) or td.get("error"):
            continue

        if agent == "WeatherAgent":
            city = normalize_city_name(td.get("city", "?"))

            def _weather_line(date: str, fc: Dict[str, Any]) -> str:
                metrics = [
                    f"高温{fc.get('temp_high_c', fc.get('high', fc.get('daytemp', '?')))}°C",
                    f"低温{fc.get('temp_low_c', fc.get('low', fc.get('nighttemp', '?')))}°C",
                ]
                if fc.get("daily_chance_of_rain") is not None:
                    metrics.append(f"降水概率{fc['daily_chance_of_rain']}%")
                elif fc.get("avg_humidity") is not None:
                    metrics.append(f"湿度{fc['avg_humidity']}%")
                return (
                    f"{city} {date}: "
                    f"{fc.get('condition') or fc.get('text') or '?'} "
                    f"{' '.join(metrics)}"
                )

            if td.get("forecasts"):
                for item in td["forecasts"]:
                    if isinstance(item, dict) and item.get("forecast"):
                        date = item.get("date", "?")
                        weather_lines.append(_weather_line(date, item["forecast"]))
                        weather_by_city_date.setdefault(city, {})[date] = item["forecast"]
            elif td.get("forecast"):
                date = td.get("date", "?")
                weather_lines.append(_weather_line(date, td["forecast"]))
                weather_by_city_date.setdefault(city, {})[date] = td["forecast"]
        elif agent == "AttractionAgent":
            expected_city = normalize_city_name(td.get("city"))
            for a in td.get("attractions") or []:
                if isinstance(a, dict) and not _poi_name_invalid(a.get("name"), expected_city=expected_city):
                    item = {
                        "city": expected_city,
                        "name": a.get("name"),
                        "address": a.get("address"),
                        "district": a.get("district"),
                        "location": a.get("location"),
                        "rating": a.get("rating"),
                    }
                    attractions.append(item)
                    _append_city_item(attractions_by_city, expected_city, item)
        elif agent == "HotelAgent":
            expected_city = normalize_city_name(td.get("city"))
            for h in td.get("hotels") or []:
                if isinstance(h, dict) and not _poi_name_invalid(h.get("name"), expected_city=expected_city):
                    item = {
                        "city": expected_city,
                        "name": h.get("name"),
                        "district": h.get("district"),
                        "address": h.get("address"),
                        "avg_price_cny": h.get("avg_price_cny"),
                        "rating": h.get("rating"),
                    }
                    hotels.append(item)
                    _append_city_item(hotels_by_city, expected_city, item)
        elif agent == "RestaurantAgent":
            expected_city = normalize_city_name(td.get("city") or td.get("location"))
            for r in td.get("restaurants") or []:
                if isinstance(r, dict) and not _restaurant_poi_invalid(r, expected_city=expected_city):
                    item = {
                        "city": expected_city,
                        "name": r.get("name"),
                        "district": r.get("district"),
                        "address": r.get("address"),
                        "avg_price_cny": r.get("avg_price_cny"),
                        "rating": r.get("rating"),
                    }
                    restaurants.append(item)
                    _append_city_item(restaurants_by_city, expected_city, item)

    weather_summary = "\n".join(weather_lines) if weather_lines else ""
    return {
        "weather_summary": weather_summary,
        "attraction_list": attractions[:15],
        "hotels": hotels[:15],
        "restaurants": restaurants[:15],
        "weather_by_city_date": weather_by_city_date,
        "attractions_by_city": attractions_by_city,
        "hotels_by_city": hotels_by_city,
        "restaurants_by_city": restaurants_by_city,
    }


def inject_itinerary_params(
    task: Dict[str, Any],
    prior_results: Dict[str, Any],
) -> Dict[str, Any]:
    """为 ItineraryAgent 子任务只注入景点 tool_data 摘要。"""
    if task.get("agent") != "ItineraryAgent":
        return task
    ctx = extract_upstream_for_itinerary(prior_results)
    params = dict(task.get("params") or {})
    if ctx["attraction_list"]:
        params["attraction_list"] = ctx["attraction_list"]
    if ctx["attractions_by_city"]:
        params["attractions_by_city"] = ctx["attractions_by_city"]
    return {**task, "params": params}


def format_results_for_aggregation(
    results: Dict[str, Any],
    *,
    summary_max_chars: int = 400,
) -> str:
    """将子任务结果格式化为聚合输入：tool_data 为主，agent_summary 为辅（截断）。"""
    blocks = []
    for task_id, res in results.items():
        if not isinstance(res, dict):
            blocks.append({"task_id": task_id, "raw": res})
            continue
        agent = res.get("agent", "")
        tool_data = res.get("tool_data")
        summary = (res.get("agent_summary") or "").strip()
        if len(summary) > summary_max_chars:
            summary = summary[:summary_max_chars] + "…"
        quality = assess_tool_data_quality(agent, tool_data)
        summary_note = summary or "（无文字总结，请以 tool_data 为准）"
        if quality != "ok":
            summary_note = "（该子任务工具未返回有效数据，请忽略文字补充，仅以 tool_data 错误信息为准）"
        blocks.append({
            "task_id": task_id,
            "agent": agent,
            "data_quality": quality,
            "tool_data": tool_data,
            "agent_summary_note": summary_note,
        })
    return json.dumps(blocks, ensure_ascii=False, indent=2)


def format_date_anchor_constraint(date_anchor: Dict[str, Any] | None) -> str:
    """聚合阶段日期硬约束（优先于 pre_survey 中的示例年份）。"""
    if not date_anchor:
        return ""
    trip_dates = date_anchor.get("trip_dates") or []
    trip_range = date_anchor.get("trip_range") or (trip_dates[0] if trip_dates else "")
    dates_line = ", ".join(trip_dates) if trip_dates else "（未解析）"
    return (
        "\n\n## 日期约束（必须遵守，优于 pre_survey 中的任何示例年份）\n"
        f"- 今天是 {date_anchor.get('today')}\n"
        f"- 本请求出行日期: {trip_range}（{dates_line}）\n"
        "- 用户说的「下周/明天」等相对日期，只使用上述日期，禁止推断为 2024 或其他年份\n"
        "- 子任务 tool_data 中的日期若与上述一致，视为正确；不要捏造「系统时间基准与用户不一致」类说法\n"
        "- 无 tool_data 支撑时说明缺失，不要用其他年份的数据填补"
    )


def build_aggregation_prompt(
    *,
    user_query: str,
    execution_plan: Dict[str, Any],
    results: Dict[str, Any],
    recent_dialogue: str = "",
    date_anchor: Dict[str, Any] | None = None,
) -> str:
    """统一构建多子任务聚合 prompt（无论是否启用长期记忆）。"""
    query_for_agg = execution_plan.get("enriched_query") or user_query
    pre_survey_text = json.dumps(
        execution_plan.get("pre_survey", {}), ensure_ascii=False, indent=2
    )
    memories_text = json.dumps(
        execution_plan.get("retrieved_memories", []), ensure_ascii=False, indent=2
    )
    results_text = format_results_for_aggregation(results)
    anchor = date_anchor or execution_plan.get("date_anchor")
    prompt = AGGREGATION_PROMPT.format(
        user_query=query_for_agg,
        pre_survey=pre_survey_text,
        memories=memories_text,
        total_goal=execution_plan.get("total_goal", ""),
        results=results_text,
    )
    prompt += format_date_anchor_constraint(anchor)
    dialogue = (recent_dialogue or "").strip()
    if dialogue:
        prompt += f"\n\n## 本线程最近对话（Chapter-3 短期记忆）\n{dialogue}"
    return prompt


MEMORY_AGGREGATION_INSTRUCTION = (
    "请根据用户原始请求的范围，综合子任务执行结果生成回复。"
    "严格匹配用户问题，不要添加用户未询问的内容（例如用户只问天气，不要输出行程/酒店/美食攻略）。"
    "仅当用户明确要求旅行规划时，才提供完整的多日行程方案。"
)
