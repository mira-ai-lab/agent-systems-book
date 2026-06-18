"""domain 层：SubAgentRegistry、DomainConfig、旅行 specs。"""

import pytest

from agent_framework.domain.agent_registry import SubAgentRegistry
from agent_framework.domain.domain_config import DomainConfig, empty_context_builder
from domains.travel.guess_rules import TRAVEL_GUESS_RULES
from domains.travel.specs import TRAVEL_AGENT_SPECS, create_travel_registry_stub


def test_sub_agent_registry_register_and_resolve():
    registry = SubAgentRegistry()
    registry.register("DemoAgent", lambda: object(), description="demo", requires_tool=True)
    assert registry.has_agent("DemoAgent")
    assert registry.resolve_agent("DemoAgent") == "DemoAgent"
    assert registry.resolve_agent("Missing") is None
    assert registry.requires_tool("DemoAgent")


def test_sub_agent_registry_lazy_instance():
    calls = []

    def creator():
        calls.append(1)
        return {"name": "demo"}

    registry = SubAgentRegistry()
    registry.register("DemoAgent", creator)
    a = registry.get_agent("DemoAgent")
    b = registry.get_agent("DemoAgent")
    assert a is b
    assert calls == [1]


def test_sub_agent_registry_unknown_raises():
    registry = SubAgentRegistry()
    with pytest.raises(ValueError, match="未知的子智能体"):
        registry.get_agent("Nope")


def test_sub_agent_registry_guess_rules():
    registry = SubAgentRegistry()
    registry.register("WeatherAgent", lambda: None, description="w")
    registry.register_guess_rules([(("rain", "雨"), "WeatherAgent")])
    assert registry.guess_agent("明天有雨吗") == "WeatherAgent"
    assert registry.guess_agent("hello") is None


def test_domain_config_context_and_guess():
    config = DomainConfig(
        context_builder=lambda: "【时间锚点】2026-06-11",
        enable_guess_agent=True,
    )
    assert "2026-06-11" in config.build_context_block()
    registry = create_travel_registry_stub()
    assert config.guess_agent("查航班", registry) == "FlightAgent"


def test_domain_config_custom_guess_fn():
    config = DomainConfig(guess_fn=lambda desc, reg: "HotelAgent" if "住" in desc else None)
    registry = create_travel_registry_stub()
    assert config.guess_agent("想住一晚", registry) == "HotelAgent"


def test_empty_context_builder():
    assert empty_context_builder() == ""


def test_travel_agent_specs_cover_five_agents():
    assert set(TRAVEL_AGENT_SPECS) == {
        "WeatherAgent",
        "HotelAgent",
        "RestaurantAgent",
        "ItineraryAgent",
        "FlightAgent",
    }


def test_travel_guess_rules_non_empty():
    assert len(TRAVEL_GUESS_RULES) >= 5
