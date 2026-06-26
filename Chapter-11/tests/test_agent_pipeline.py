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


def test_render_agent_prompt_template_ignores_json_braces():
    """optimizer 产出含 JSON 示例时不应触发 str.format KeyError。"""
    from agent_framework.optimization.agents.runtime import render_agent_prompt_template

    template = (
        "你是航班助手。当前日期：{today}\n"
        '输出示例: {"final_response": "...", "invoked_tools": ["search_flights"]}'
    )
    rendered = render_agent_prompt_template(template, locale="zh")
    assert "final_response" in rendered
    assert "{today}" not in rendered

def test_repair_agent_system_prompt_placeholders_flight():
    from agent_framework.optimization.agents.runtime import (
        extract_agent_system_prompt,
        repair_agent_system_prompt_placeholders,
    )

    broken = "你是航班助手。收到查询后必须调用 search_flights。"
    fixed = repair_agent_system_prompt_placeholders(broken, agent_name="FlightAgent")
    assert "{today}" in fixed
    assert extract_agent_system_prompt(fixed, agent_name="FlightAgent")
    assert extract_agent_system_prompt(broken, agent_name="FlightAgent", repair_missing=True)


def test_resolve_optimization_start_template_locales_vs_optimized():
    from agent_framework.optimization.agents.runtime import resolve_optimization_start_template

    locales = get_agent_prompt_template("FlightAgent", locale="zh")
    assert "只能使用 search_flights" in locales
    assert resolve_optimization_start_template(
        "FlightAgent", locale="zh", start_from_locales=True
    ) == locales
    broken = resolve_optimization_start_template("FlightAgent", locale="zh", start_from_locales=False)
    assert broken != locales
    assert "{today}" in broken


def test_all_agents_fixtures_train_dev_split():
    """每个 Agent 在 train/dev 各有 case；FlightAgent 因扩展 benchmark 为 3+2。"""
    fixtures = load_single_agent_cases()
    expected_counts = {
        "FlightAgent": (3, 2),
    }
    for agent_name in TRAVEL_OPTIMIZABLE_AGENTS:
        train = fixtures.cases_for_split("train", agent_name=agent_name)
        dev = fixtures.cases_for_split("dev", agent_name=agent_name)
        exp_train, exp_dev = expected_counts.get(agent_name, (1, 1))
        assert len(train) == exp_train, agent_name
        assert len(dev) == exp_dev, agent_name
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
