"""Domain-agent catalog and A2A service-side agent factory.

Unlike Chapter-6, whose agents call local Chapter-5 search functions, the agents
registered here are backed by third-party MCP tools: weatherapi (weather), Baidu
Map (hotel/restaurant/POI), and VariFlight (flights). ``DomainAgentRegistry`` is
the single source of truth: it exposes a capability catalog consumed by
Chapter-6's ``task_plan`` / ``agent_routing``, and it builds the actual LangGraph
agents that each Chapter-8 A2A service publishes.

:author: zhanghong, liuhui, xingrui
:date: 2026-11-22
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

from travel_multi_agent.config import BOOK_ROOT  # noqa: F401 (registers root + Chapter-6)
from llm import create_llm
from travel_multi_agent.mcp_tools import load_tools
from travel_multi_agent.tracing import get_logger, log_info

from domain_agents import DOMAIN_AGENT_SPECS  # Chapter-6 role prompts

logger = get_logger(__name__)

AgentCreator = Callable[[], Awaitable[Any]]


class DomainAgentName(str, Enum):
    """Enumeration of the five Chapter-8 domain agents (A2A Agent Card names)."""

    WEATHER = "WeatherAgent"
    HOTEL = "HotelAgent"
    RESTAURANT = "RestaurantAgent"
    FLIGHT = "FlightAgent"
    ITINERARY = "ItineraryAgent"


@dataclass(frozen=True)
class AgentSpec:
    """Declare one domain agent's identity, creator, and skills (like Chapter-6)."""

    name: str
    description: str
    creator: AgentCreator
    skills: List[Dict[str, Any]] = field(default_factory=list)


def build_agent(tools: List[Any], system_prompt: str) -> Any:
    """Create a LangChain ReAct agent with the shared LLM and the given tools."""
    return create_agent(
        create_llm(),
        tools=list(tools),
        system_prompt=system_prompt,
        checkpointer=MemorySaver(),
    )


def _make_creator(agent_name: str, server_name: str) -> AgentCreator:
    """Build an async creator that binds one MCP server's tools to one agent."""

    async def creator() -> Any:
        tools = await load_tools(server_name)
        return build_agent(tools, DOMAIN_AGENT_SPECS[agent_name].system_prompt)

    return creator


def _make_no_tool_creator(agent_name: str) -> AgentCreator:
    """Build an async creator for a synthesis agent that binds no tools."""

    async def creator() -> Any:
        return build_agent([], DOMAIN_AGENT_SPECS[agent_name].system_prompt)

    return creator


TRAVEL_AGENT_SPECS: Dict[DomainAgentName, AgentSpec] = {
    DomainAgentName.WEATHER: AgentSpec(
        name=DomainAgentName.WEATHER.value,
        description="查询指定城市、日期的天气预报，提供温度、天气状况和出行建议",
        creator=_make_creator(DomainAgentName.WEATHER.value, "weather"),
        skills=[
            {
                "name": "weather_query",
                "inputSchema": ["city", "date"],
                "outputSchema": ["forecast"],
            }
        ],
    ),
    DomainAgentName.HOTEL: AgentSpec(
        name=DomainAgentName.HOTEL.value,
        description="根据位置、预算、偏好（近景区/安静/品牌）推荐酒店；地图关键词与主观偏好分离",
        creator=_make_creator(DomainAgentName.HOTEL.value, "baidu"),
        skills=[
            {
                "name": "hotel_search",
                "inputSchema": ["city", "preferences", "budget"],
                "outputSchema": ["hotels"],
            }
        ],
    ),
    DomainAgentName.RESTAURANT: AgentSpec(
        name=DomainAgentName.RESTAURANT.value,
        description="根据菜系、位置、预算推荐当地特色餐厅和美食",
        creator=_make_creator(DomainAgentName.RESTAURANT.value, "baidu"),
        skills=[
            {
                "name": "restaurant_search",
                "inputSchema": ["location", "cuisine", "budget"],
                "outputSchema": ["restaurants"],
            }
        ],
    ),
    DomainAgentName.ITINERARY: AgentSpec(
        name=DomainAgentName.ITINERARY.value,
        description="综合旅行出发时间和返回时间、天气、景点、酒店、餐厅和航班结果，生成最终旅行日程与用户答复。",
        creator=_make_no_tool_creator(DomainAgentName.ITINERARY.value),
        skills=[
            {
                "name": "itinerary_synthesis",
                "inputSchema": [
                    "cities",
                    "dates",
                    "preferences",
                    "departure_city",
                    "destination_city",
                    "days",
                ],
                "outputSchema": ["final_response"],
            }
        ],
    ),
    DomainAgentName.FLIGHT: AgentSpec(
        name=DomainAgentName.FLIGHT.value,
        description="查询出发地到目的地的航班信息、价格和时刻表",
        creator=_make_creator(DomainAgentName.FLIGHT.value, "variflight"),
        skills=[
            {
                "name": "flight_search",
                "inputSchema": ["departure", "arrival", "date"],
                "outputSchema": ["flights"],
            }
        ],
    ),
}


class DomainAgentRegistry:
    """Capability catalog plus lazy A2A-service agent creation (Ch6-style)."""

    def __init__(self) -> None:
        self.agents: Dict[str, Dict[str, Any]] = {}
        self._creators: Dict[str, AgentCreator] = {}
        self._instances: Dict[str, Any] = {}

    def register(
        self,
        name: str,
        creator: AgentCreator,
        *,
        description: str = "",
        skills: Optional[List[Dict[str, Any]]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register one domain agent with metadata and a lazy async creator."""
        info = dict(meta or {})
        info.setdefault("name", name)
        info.setdefault("description", description)
        info.setdefault("skills", skills or [])
        self.agents[name] = info
        self._creators[name] = creator

    async def get_agent(self, agent_name: str) -> Any:
        """Create (awaiting its async creator) and cache one agent lazily."""
        if agent_name not in self._instances:
            creator = self._creators.get(agent_name)
            if not creator:
                raise ValueError(f"Unknown domain agent: {agent_name}")
            log_info(logger, "agent.create", agent=agent_name)
            self._instances[agent_name] = await creator()
        return self._instances[agent_name]

    def get_agent_names(self) -> List[str]:
        """Return all registered agent names."""
        return list(self.agents.keys())

    def get_all_agents_text(self) -> str:
        """Render agent names and descriptions for routing prompts."""
        return "\n".join(
            f"- {a['name']}: {a['description']}" for a in self.agents.values()
        )

    def get_capability_catalog_text(self) -> str:
        """Render the capability catalog without agent names (same as Chapter-6)."""
        return "\n".join(f"- {a['description']}" for a in self.agents.values())

    def get_agent_parameters_text(self) -> str:
        """Render skill input/output schemas for dependency analysis."""
        lines = []
        for info in self.agents.values():
            lines.append(f"{info['name']}: {info['description']}")
            for skill in info.get("skills", []):
                lines.append(
                    f"\t{skill['name']}, inputSchema:{skill['inputSchema']}, "
                    f"outputSchema:{skill['outputSchema']}"
                )
        return "\n".join(lines)

    def get_agent_input_fields(self, agent_name: str) -> set:
        """Return all input-field names declared by an agent's skills."""
        info = self.agents.get(agent_name) or {}
        return {
            str(field_name)
            for skill in info.get("skills", [])
            for field_name in skill.get("inputSchema", [])
        }


def create_travel_registry() -> DomainAgentRegistry:
    """Build the travel registry from ``TRAVEL_AGENT_SPECS``."""
    registry = DomainAgentRegistry()
    for spec in TRAVEL_AGENT_SPECS.values():
        registry.register(
            spec.name,
            spec.creator,
            description=spec.description,
            skills=spec.skills,
        )
    return registry
