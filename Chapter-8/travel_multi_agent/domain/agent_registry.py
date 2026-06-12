"""Chapter-8: 子智能体注册表（供 TaskPlanner 与 fixed_graph 路由使用）"""

from __future__ import annotations

from typing import Any


class SubAgentRegistry:
    """子智能体注册表（Chapter-5 HotelAgent 扩展为 5 个专业 Agent）"""

    def __init__(self) -> None:
        self.agents = {
            "WeatherAgent": {
                "name": "WeatherAgent",
                "description": "查询指定城市、日期的天气预报，提供温度、天气状况和出行建议",
                "requires_tool": True,
                "skills": [{
                    "name": "get_weather_forecast",
                    "inputSchema": ["city", "days"],
                    "outputSchema": ["forecasts", "city", "days"],
                }, {
                    "name": "get_weather",
                    "inputSchema": ["city", "date"],
                    "outputSchema": ["forecast", "temperature", "condition", "advice"],
                }],
            },
            "HotelAgent": {
                "name": "HotelAgent",
                "description": "根据位置、预算、偏好（近景区/安静/品牌）推荐酒店；地图关键词与主观偏好分离",
                "requires_tool": True,
                "skills": [{
                    "name": "recommend_hotel",
                    "inputSchema": ["city", "preferences", "budget_cny_per_night_max"],
                    "outputSchema": ["hotels", "prices", "ratings", "locations"],
                }],
            },
            "RestaurantAgent": {
                "name": "RestaurantAgent",
                "description": "根据菜系、位置、预算推荐当地特色餐厅和美食",
                "requires_tool": True,
                "skills": [{
                    "name": "recommend_restaurant",
                    "inputSchema": ["location", "cuisine", "budget_cny_per_person"],
                    "outputSchema": ["restaurants", "cuisines", "prices", "ratings"],
                }],
            },
            "ItineraryAgent": {
                "name": "ItineraryAgent",
                "description": "拉取候选 POI，基于兴趣点确定性生成逐日行程骨架（plan）；润色时可参考任务描述中的天气/酒店/美食信息",
                "requires_tool": True,
                "skills": [{
                    "name": "fetch_candidate_pois",
                    "inputSchema": ["city", "preferences", "limit"],
                    "outputSchema": ["candidate_pois", "city", "count"],
                }, {
                    "name": "plan_itinerary",
                    "inputSchema": ["city", "days", "candidate_pois"],
                    "outputSchema": ["plan", "city", "poi_count", "transportation"],
                }],
            },
            "FlightAgent": {
                "name": "FlightAgent",
                "description": "查询出发地到目的地的航班信息、价格和时刻表",
                "requires_tool": True,
                "skills": [{
                    "name": "search_flights",
                    "inputSchema": ["departure", "arrival", "date"],
                    "outputSchema": ["flights", "prices", "times", "airlines"],
                }],
            },
        }

    def get_all_agents_text(self) -> str:
        return "\n".join(f"- {a['name']}: {a['description']}" for a in self.agents.values())

    def get_agent_parameters_text(self) -> str:
        lines = []
        for info in self.agents.values():
            lines.append(info["name"])
            for skill in info["skills"]:
                lines.append(
                    f"\t{skill['name']}, inputSchema:{skill['inputSchema']}, "
                    f"outputSchema:{skill['outputSchema']}"
                )
        return "\n".join(lines)

    def requires_tool(self, agent_name: str) -> bool:
        """查询指定 Agent 是否必须调用工具（用于执行状态判断）。"""
        info = self.agents.get(agent_name)
        return bool(info and info.get("requires_tool", False))
