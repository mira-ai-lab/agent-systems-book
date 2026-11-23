"""domain/parsing 与聚合辅助函数单元测试（无 LLM）。"""

import json

import pytest

from agent_framework.domain.parsing import (
    guess_agent,
    order_from_dependency_json,
    parse_decomposition_response,
    parse_json_from_llm,
    parse_pre_survey,
)
from agent_framework.infra.memory.aggregation_helpers import (
    direct_response_from_results,
    is_single_direct_response,
)
from domains.travel.specs import create_travel_registry_stub


def test_parse_decomposition_response_zh():
    raw = """
# 目标
帮用户查上海明天天气并推荐酒店

# 任务拆解
- 查询上海明天天气预报
- 推荐上海陆家嘴附近酒店
"""
    out = parse_decomposition_response(raw, lang="zh")
    assert "上海" in out["totalGoal"]
    assert len(out["subSteps"]) == 2
    assert "天气预报" in out["subSteps"][0]


def test_parse_decomposition_response_null_fallback():
    raw = """
# 目标
仅闲聊

# 任务拆解
- NULL
"""
    out = parse_decomposition_response(raw, lang="zh")
    assert out["subSteps"] == ["NULL"]


def test_parse_json_from_llm_plain():
    assert parse_json_from_llm('{"a": 1}') == {"a": 1}


def test_parse_json_from_llm_codeblock():
    text = '说明如下：\n```json\n{"tasks": ["T1"]}\n```'
    assert parse_json_from_llm(text) == {"tasks": ["T1"]}


def test_parse_json_from_llm_raises_on_invalid():
    with pytest.raises((ValueError, json.JSONDecodeError)):
        parse_json_from_llm("not json at all")


def test_order_from_dependency_json_mapping():
    order = order_from_dependency_json({"1": "T2", "2": "T1"}, num_tasks=2)
    assert order == ["T2", "T1"]


def test_order_from_dependency_json_fallback():
    order = order_from_dependency_json({}, num_tasks=3)
    assert order == ["T1", "T2", "T3"]


def test_guess_agent_delegates_to_registry():
    registry = create_travel_registry_stub()
    assert guess_agent("查北京天气", registry) == "WeatherAgent"
    assert guess_agent("unknown topic", registry) is None


def test_is_single_direct_response():
    assert is_single_direct_response({"T1": {"agent_summary": "晴"}})
    assert not is_single_direct_response({"T1": {}, "T2": {}})


def test_direct_response_from_results_prefers_summary():
    results = {"T1": {"agent_summary": "明天晴", "tool_data": {"temp": 25}}}
    assert direct_response_from_results(results) == "明天晴"


def test_direct_response_from_results_tool_data():
    results = {"T1": {"tool_data": {"city": "北京"}}}
    text = direct_response_from_results(results)
    assert "北京" in text


def test_parse_pre_survey_empty_sections():
    result = parse_pre_survey("无结构化内容")
    assert result["given_facts"] == []
    assert result["raw_text"] == "无结构化内容"
