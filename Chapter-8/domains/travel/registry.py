"""旅行领域子 Agent 注册表与 DomainConfig 工厂。"""

from __future__ import annotations

from agent_framework.domain.agent_registry import SubAgentRegistry
from domains.travel.specs import register_travel_agent_specs

from domains.travel.guess_rules import TRAVEL_GUESS_RULES


def create_travel_registry() -> SubAgentRegistry:
    from domains.travel.agents.flight import create_flight_agent
    from domains.travel.agents.hotel import create_hotel_agent
    from domains.travel.agents.itinerary import create_itinerary_agent
    from domains.travel.agents.restaurant import create_restaurant_agent
    from domains.travel.agents.weather import create_weather_agent

    creators = {
        "WeatherAgent": create_weather_agent,
        "HotelAgent": create_hotel_agent,
        "RestaurantAgent": create_restaurant_agent,
        "ItineraryAgent": create_itinerary_agent,
        "FlightAgent": create_flight_agent,
    }
    registry = SubAgentRegistry()
    register_travel_agent_specs(registry, creators)
    registry.register_guess_rules(TRAVEL_GUESS_RULES)
    return registry


def travel_domain_config(enable_guess_agent: bool = False) -> "DomainConfig":
    from agent_framework.domain.domain_config import DomainConfig
    from domains.travel.plan_context import format_time_anchor_block

    return DomainConfig(
        context_builder=format_time_anchor_block,
        enable_guess_agent=enable_guess_agent,
    )
