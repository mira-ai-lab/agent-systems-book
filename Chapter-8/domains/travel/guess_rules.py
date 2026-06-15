"""旅行领域 guess 规则（注册到 SubAgentRegistry.register_guess_rules）。"""

from __future__ import annotations

from typing import Sequence, Tuple

GuessRule = Tuple[Sequence[str], str]

TRAVEL_GUESS_RULES: list[GuessRule] = [
    (("天气", "weather", "气温", "降水"), "WeatherAgent"),
    (("酒店", "hotel", "住宿", "民宿"), "HotelAgent"),
    (("景点", "attraction", "打卡", "景区", "行程", "攻略"), "ItineraryAgent"),
    (("餐厅", "美食", "restaurant", "菜"), "RestaurantAgent"),
    (("航班", "flight", "飞机"), "FlightAgent"),
]
