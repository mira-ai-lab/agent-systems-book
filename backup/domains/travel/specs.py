"""旅行领域子 Agent 元数据（无 LangChain / agent 模块依赖，可供测试与 Planner prompt 使用）。"""

from __future__ import annotations

from typing import Any, Dict, List

TRAVEL_AGENT_SPECS: Dict[str, Dict[str, Any]] = {
    "WeatherAgent": {
        "description": "查询指定城市、日期的天气预报，提供温度、天气状况和出行建议",
        "requires_tool": True,
        "skills": [
            {
                "name": "get_weather_forecast",
                "inputSchema": ["city", "days"],
                "outputSchema": ["forecasts", "city", "days"],
            },
            {
                "name": "get_weather",
                "inputSchema": ["city", "date"],
                "outputSchema": ["forecast", "temperature", "condition", "advice"],
            },
        ],
    },
    "HotelAgent": {
        "description": "根据位置、预算、偏好（近景区/安静/品牌）推荐酒店；地图关键词与主观偏好分离",
        "requires_tool": True,
        "skills": [
            {
                "name": "recommend_hotel",
                "inputSchema": ["city", "preferences", "budget_cny_per_night_max"],
                "outputSchema": ["hotels", "prices", "ratings", "locations"],
            },
        ],
    },
    "RestaurantAgent": {
        "description": "根据菜系、位置、预算推荐当地特色餐厅和美食",
        "requires_tool": True,
        "skills": [
            {
                "name": "recommend_restaurant",
                "inputSchema": ["location", "cuisine", "budget_cny_per_person"],
                "outputSchema": ["restaurants", "cuisines", "prices", "ratings"],
            },
        ],
    },
    "ItineraryAgent": {
        "description": "拉取候选 POI，基于兴趣点确定性生成逐日行程骨架（plan）；润色时可参考任务描述中的天气/酒店/美食信息",
        "requires_tool": True,
        "skills": [
            {
                "name": "fetch_candidate_pois",
                "inputSchema": ["city", "preferences", "limit"],
                "outputSchema": ["candidate_pois", "city", "count"],
            },
            {
                "name": "plan_itinerary",
                "inputSchema": ["city", "days", "candidate_pois"],
                "outputSchema": ["plan", "city", "poi_count", "transportation"],
            },
        ],
    },
    "FlightAgent": {
        "description": "查询出发地到目的地的航班信息、价格和时刻表",
        "requires_tool": True,
        "skills": [
            {
                "name": "search_flights",
                "inputSchema": ["departure", "arrival", "date"],
                "outputSchema": ["flights", "prices", "times", "airlines"],
            },
        ],
    },
}


def register_travel_agent_specs(registry: Any, creators: Dict[str, Any]) -> None:
    """将旅行领域 Agent 元数据 + creator 注册到 registry。"""
    for name, spec in TRAVEL_AGENT_SPECS.items():
        creator = creators.get(name)
        if not creator:
            continue
        registry.register(
            name,
            creator,
            description=spec["description"],
            requires_tool=spec["requires_tool"],
            skills=spec["skills"],
        )


def create_travel_registry_stub() -> Any:
    """仅注册元数据与占位 creator，供单元测试（不导入 agents 模块）。"""
    from agent_framework.domain.agent_registry import SubAgentRegistry

    registry = SubAgentRegistry()
    register_travel_agent_specs(
        registry,
        {name: (lambda: None) for name in TRAVEL_AGENT_SPECS},
    )
    from domains.travel.guess_rules import TRAVEL_GUESS_RULES
    registry.register_guess_rules(TRAVEL_GUESS_RULES)
    return registry
