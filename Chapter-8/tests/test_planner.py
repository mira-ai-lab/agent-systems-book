"""Unit tests for the registry catalog and pipeline (no LLM / network calls).

:author: zhanghong, liuhui, xingrui
:date: 2026-11-22"""

from travel_multi_agent.agents.registry import DomainAgentName, create_travel_registry


def test_travel_registry_lists_five_agents():
    registry = create_travel_registry()
    assert len(registry.agents) == 5
    assert set(registry.get_agent_names()) == {member.value for member in DomainAgentName}
    assert "ItineraryAgent" in registry.agents
    text = registry.get_all_agents_text()
    assert "ItineraryAgent" in text
    params = registry.get_agent_parameters_text()
    assert "itinerary_synthesis" in params


def test_agent_input_fields():
    registry = create_travel_registry()
    assert registry.get_agent_input_fields("WeatherAgent") == {"city", "date"}
    assert registry.get_agent_input_fields("FlightAgent") == {"departure", "arrival", "date"}


def test_pipeline_config():
    from travel_multi_agent.orchestration.pipeline import PipelineConfig

    full = PipelineConfig()
    assert full.enable_pre_survey and full.enable_memory
    minimal = PipelineConfig(enable_pre_survey=False, enable_memory=False)
    assert not minimal.enable_pre_survey
    assert not minimal.needs_save_memory