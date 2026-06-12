"""FlightAgent — 航班查询。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from langchain_core.tools import tool

from travel_multi_agent.agents.base import build_agent
from travel_multi_agent.infra.travel_api import (
    fetch_flights_from_api,
    fetch_flights_from_variflight_api,
    require_non_empty,
)


@tool
async def search_flights(departure: str, arrival: str, date: str) -> Dict[str, Any]:
    """查询出发地到目的地的航班信息。"""
    ok1, err1 = require_non_empty(departure, "departure")
    if not ok1:
        return {"error": err1}

    ok2, err2 = require_non_empty(arrival, "arrival")
    if not ok2:
        return {"error": err2}

    try:
        result = await fetch_flights_from_variflight_api(departure, arrival, date, limit=10)
        if not result.get("error"):
            return {
                "departure": departure,
                "arrival": arrival,
                "date": date,
                "flights": result.get("flights", []),
                "data_source": "variflight",
            }
    except Exception:
        pass

    try:
        result = await fetch_flights_from_api(departure, arrival, date, limit=10)
        if not result.get("error"):
            return {
                "departure": departure,
                "arrival": arrival,
                "date": date,
                "flights": result.get("flights", []),
                "data_source": result.get("data_source"),
            }
    except Exception as e:
        return {"error": f"flight_query_failed: {str(e)}"}

    return {"error": "无法获取航班信息"}


def _system_prompt() -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    return f"""你是专业的航班查询助手。

当前日期（本地时间）：{today}

职责：
1. 只能使用 search_flights 工具查询航班
2. departure、arrival、date 三个参数必填
3. 返回航班列表后，推荐最合适的2-3个航班（考虑时间、价格）
4. 提供航班的起飞到达时间、航空公司、参考价格
5. 非航班相关问题，回复：我只能协助航班查询

注意：
- date 必须是 YYYY-MM-DD 格式
- 城市名会自动转换为机场代码（支持常见城市）
- 建议使用机场三字码（如PVG、PEK）以获得更准确的结果
- 用户说「今天」时必须对应当前日期 {today}，不要臆造其他日期
"""


def create_flight_agent() -> Any:
    return build_agent([search_flights], _system_prompt())
