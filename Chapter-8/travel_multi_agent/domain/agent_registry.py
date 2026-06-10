"""Chapter-8: 子智能体注册表（供 TaskPlanner 与 fixed_graph 路由使用）"""

from __future__ import annotations


class SubAgentRegistry:
    """子智能体注册表（Chapter-5 HotelAgent 扩展为 6 个专业 Agent）"""

    def __init__(self) -> None:
        self.agents = {
            "WeatherAgent": {
                "name": "WeatherAgent",
                "description": "查询指定城市、日期的天气预报，提供温度、天气状况和出行建议",
                "skills": [{
                    "name": "get_weather",
                    "inputSchema": ["city", "date"],
                    "outputSchema": ["forecast", "temperature", "condition", "advice"],
                }],
            },
            "AttractionAgent": {
                "name": "AttractionAgent",
                "description": "根据城市、兴趣偏好推荐旅游景点和必去打卡地",
                "skills": [{
                    "name": "recommend_attractions",
                    "inputSchema": ["city", "preferences", "limit"],
                    "outputSchema": ["attraction_list", "ratings", "locations"],
                }],
            },
            "HotelAgent": {
                "name": "HotelAgent",
                "description": "根据位置、预算、偏好（近景区/安静/品牌）推荐酒店；地图关键词与主观偏好分离（Chapter-5）",
                "skills": [{
                    "name": "recommend_hotel",
                    "inputSchema": ["city", "preferences", "budget_cny_per_night_max"],
                    "outputSchema": ["hotels", "prices", "ratings", "locations"],
                }],
            },
            "RestaurantAgent": {
                "name": "RestaurantAgent",
                "description": "根据菜系、位置、预算推荐当地特色餐厅和美食",
                "skills": [{
                    "name": "recommend_restaurant",
                    "inputSchema": ["location", "cuisine", "budget_cny_per_person"],
                    "outputSchema": ["restaurants", "cuisines", "prices", "ratings"],
                }],
            },
            "ItineraryAgent": {
                "name": "ItineraryAgent",
                "description": "综合天气、景点、交通、住宿信息，生成详细的每日行程安排",
                "skills": [{
                    "name": "plan_itinerary",
                    "inputSchema": [
                        "departure_city", "destination_city", "days",
                        "weather_summary", "attraction_list", "preferences",
                    ],
                    "outputSchema": ["daily_plan", "transportation", "stay_suggestion"],
                }],
            },
            "FlightAgent": {
                "name": "FlightAgent",
                "description": "查询出发地到目的地的航班信息、价格和时刻表",
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
