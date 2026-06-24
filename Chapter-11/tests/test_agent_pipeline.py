"""Agent-B2 pipeline 与多 Agent 优化测试（mock，无需 live LLM）。"""

from __future__ import annotations

import pytest

from agent_framework.optimization.agent_pipeline import parse_agent_slots
from agent_framework.optimization.agents.fixtures import load_single_agent_cases
from agent_framework.optimization.agents.runtime import (
    AGENT_REQUIRED_PLACEHOLDERS,
    TRAVEL_OPTIMIZABLE_AGENTS,
    extract_agent_system_prompt,
    get_agent_prompt_template,
)
from agent_framework.optimization.optimizers.textgrad_agent.loss import agent_graph_constraints


def test_parse_agent_slots_all():
    agents = parse_agent_slots("all")
    assert agents == list(TRAVEL_OPTIMIZABLE_AGENTS)


def test_parse_agent_slots_comma_list():
    agents = parse_agent_slots("FlightAgent, WeatherAgent")
    assert agents == ["FlightAgent", "WeatherAgent"]


def test_parse_agent_slots_invalid():
    with pytest.raises(ValueError, match="不支持"):
        parse_agent_slots("UnknownAgent")


def test_all_agents_fixtures_train_dev_split():
    """每个 Agent 在 train/dev 各 1 条 case。"""
    fixtures = load_single_agent_cases()
    for agent_name in TRAVEL_OPTIMIZABLE_AGENTS:
        train = fixtures.cases_for_split("train", agent_name=agent_name)
        dev = fixtures.cases_for_split("dev", agent_name=agent_name)
        assert len(train) == 1, agent_name
        assert len(dev) == 1, agent_name
        assert train[0].agent_name == agent_name
        assert dev[0].agent_name == agent_name


def test_all_agent_templates_have_required_placeholders():
    """locales 基线模板满足占位符校验规则。"""
    for agent_name in TRAVEL_OPTIMIZABLE_AGENTS:
        template = get_agent_prompt_template(agent_name, locale="zh")
        cleaned = extract_agent_system_prompt(template, agent_name=agent_name)
        for token in AGENT_REQUIRED_PLACEHOLDERS[agent_name]:
            assert token in cleaned


def test_agent_graph_constraints_cover_all_agents():
    for agent_name in TRAVEL_OPTIMIZABLE_AGENTS:
        constraints = agent_graph_constraints(agent_name)
        assert constraints
        assert any("{today}" in c or "{time_anchor" in c or "Return only" in c for c in constraints)
