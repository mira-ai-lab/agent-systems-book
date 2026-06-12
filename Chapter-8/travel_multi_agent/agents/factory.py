"""子智能体工厂。"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from travel_multi_agent.agents.flight import create_flight_agent
from travel_multi_agent.agents.hotel import create_hotel_agent
from travel_multi_agent.agents.itinerary import create_itinerary_agent
from travel_multi_agent.agents.restaurant import create_restaurant_agent
from travel_multi_agent.agents.weather import create_weather_agent
from travel_multi_agent.tracing import get_logger, log_info

logger = get_logger(__name__)

_AGENT_CREATORS: Dict[str, Callable[[], Any]] = {
    "WeatherAgent": create_weather_agent,
    "HotelAgent": create_hotel_agent,
    "RestaurantAgent": create_restaurant_agent,
    "FlightAgent": create_flight_agent,
    "ItineraryAgent": create_itinerary_agent,
}


class SubAgentFactory:
    """子智能体工厂，统一管理和创建所有子智能体（单例缓存）。"""

    _agents: Dict[str, Any] = {}

    @classmethod
    def get_agent(cls, agent_name: str) -> Any:
        if agent_name not in cls._agents:
            creator = _AGENT_CREATORS.get(agent_name)
            if not creator:
                raise ValueError(f"未知的子智能体: {agent_name}")
            log_info(logger, "agent.create", agent=agent_name)
            cls._agents[agent_name] = creator()
        return cls._agents[agent_name]

    @classmethod
    def get_all_agent_names(cls) -> List[str]:
        return list(_AGENT_CREATORS.keys())
