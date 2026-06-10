"""WeatherAgent — 天气查询。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from langchain_core.tools import tool

from travel_multi_agent.agents.base import build_agent
from travel_multi_agent.infra.travel_api import (
    amap_weather_by_city_and_date,
    require_non_empty,
    resolve_relative_date,
    wttr_weather_by_city_and_date,
)
from travel_multi_agent.infra.weather_mcp import fetch_weather_via_mcp


def _slim_weather_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """去掉 MCP 完整 raw 载荷，避免 Agent 多轮 tool 消息撑爆 LLM 上下文。"""
    if not result or "raw" not in result:
        return result
    return {k: v for k, v in result.items() if k != "raw"}


@tool
async def get_weather(city: str, date: str) -> Dict[str, Any]:
    """查询指定城市、日期的天气预报。"""
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
1. 只能使用 get_weather 工具查询天气
2. city 和 date 参数必填
3. 返回天气信息后，用简洁友好的语言总结给用户
4. 如果用户询问多日天气，分别调用工具查询每一天
5. 若当前消息已是明确的天气子任务（含具体城市与日期），专注完成查询；不要因为原始需求里曾提到机票/行程而拒绝

注意：
- date 可传 YYYY-MM-DD，或直接传「今天」「明天」「后天」（工具会自动换算）
- 用户说「今天」时必须对应当前日期 {today}，不要臆造其他日期
- 工具查询顺序：WeatherAPI MCP → 高德 → wttr.in（无需关心底层，只调用 get_weather）
"""


def create_weather_agent() -> Any:
    return build_agent([get_weather], _system_prompt())
