"""WeatherAgent — 天气查询。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from langchain_core.tools import tool

from travel_multi_agent.agents.base import build_agent
from travel_multi_agent.agents.prompt_fragments import (
    MULTI_ENTITY_TOOL_RULES,
    agent_time_anchor_block,
)
from travel_multi_agent.infra.travel_api import (
    amap_weather_by_city_and_date,
    require_non_empty,
    resolve_relative_date,
    wttr_weather_by_city_and_date,
)
from travel_multi_agent.infra.weather_mcp import (
    fetch_weather_forecast_via_mcp,
    fetch_weather_via_mcp,
)


def _slim_weather_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """去掉 MCP 完整 raw 载荷，避免 Agent 多轮 tool 消息撑爆 LLM 上下文。"""
    if not result or "raw" not in result:
        return result
    return {k: v for k, v in result.items() if k != "raw"}


@tool
async def get_weather_forecast(city: str, days: int = 7) -> Dict[str, Any]:
    """查询指定城市未来若干天（1–14）的逐日天气预报（优先 MCP get_forecast）。"""
    ok, err = require_non_empty(city, "city")
    if not ok:
        return {"error": err}

    n_days = max(1, min(int(days or 7), 14))
    mcp_result = await fetch_weather_forecast_via_mcp(city, n_days)
    if mcp_result and mcp_result.get("forecasts"):
        return mcp_result

    # 回退：逐日 MCP / API
    from datetime import timedelta

    forecasts = []
    today = datetime.now().date()
    for i in range(n_days):
        d = (today + timedelta(days=i)).strftime("%Y-%m-%d")
        single = await fetch_weather_via_mcp(city, d)
        if single and not single.get("error"):
            single = _slim_weather_result(single)
            fc = single.get("forecast") or {}
            forecasts.append({
                "date": d,
                "condition": fc.get("condition") or single.get("text", "未知"),
                "temp_high_c": fc.get("temp_high_c"),
                "temp_low_c": fc.get("temp_low_c"),
            })
    if forecasts:
        return {
            "city": city,
            "days": len(forecasts),
            "forecasts": forecasts,
            "data_source": "fallback/daily_get_weather",
        }
    return {"error": "无法获取多日天气预报"}


@tool
async def get_weather(city: str, date: str) -> Dict[str, Any]:
    """查询指定城市、单日的天气预报。"""
    ok, err = require_non_empty(city, "city")
    if not ok:
        return {"error": err}

    norm_date, derr = resolve_relative_date(date)
    if derr:
        return {"error": derr}

    mcp_result = await fetch_weather_via_mcp(city, norm_date)
    if mcp_result and not mcp_result.get("error"):
        return _slim_weather_result(mcp_result)

    try:
        result = await amap_weather_by_city_and_date(city, norm_date)
        if not result.get("error") and result.get("forecast"):
            return {
                "city": city,
                "date": norm_date,
                "forecast": result["forecast"],
                "data_source": "amap_weather",
            }
    except Exception:
        pass

    try:
        result = await wttr_weather_by_city_and_date(city, norm_date)
        if not result.get("error"):
            return {
                "city": city,
                "date": norm_date,
                "text": result.get("text"),
                "forecast": result.get("forecast"),
                "data_source": "wttr.in",
            }
    except Exception as e:
        return {"error": f"weather_query_failed: {str(e)}"}

    return {"error": "无法获取天气信息"}


def _system_prompt() -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    return f"""你是专业的天气查询助手。

当前日期（本地时间）：{today}

职责：
1. 使用 get_weather_forecast（多日）或 get_weather（单日）查询天气
2. 用户问「下周」「未来 N 天」时，优先调用 get_weather_forecast(city, days=N)
3. 收到任务后必须调用工具获取真实数据，再总结；多城任务对每个城市各调一次 forecast
4. 若当前消息已是明确的天气子任务，专注完成查询

注意：
- date 可传 YYYY-MM-DD，或「今天」「明天」「后天」
- 用户说「今天」时必须对应当前日期 {today}
- 查询顺序：MCP forecast → 单日 MCP → 高德 → wttr.in
{agent_time_anchor_block()}
{MULTI_ENTITY_TOOL_RULES}
"""


def create_weather_agent() -> Any:
    return build_agent([get_weather_forecast, get_weather], _system_prompt())
