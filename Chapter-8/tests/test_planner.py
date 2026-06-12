"""TaskPlanner 与规划上下文单元测试（不调用 LLM）。"""
from datetime import date

from travel_multi_agent.domain.agent_registry import SubAgentRegistry
from travel_multi_agent.domain.parsing import guess_agent, parse_pre_survey
from travel_multi_agent.domain.plan_context import build_time_anchor, format_time_anchor_block


def test_parse_pre_survey_sections():
    raw = """
        1. 已给出或已验证的事实
        - 用户想去上海
        2. 需要查阅的事实
        - 上海明天天气
        3. 需要推导的事实
        - 行程天数
        4. 有根据的猜测
        - 用户偏好文化景点
    """
    result = parse_pre_survey(raw)
    assert "上海" in " ".join(result["given_facts"])
    assert result["facts_to_lookup"]
    assert result["facts_to_derive"]
    assert result["educated_guesses"]


def test_agent_registry_lists_five_agents():
    registry = SubAgentRegistry()
    assert len(registry.agents) == 5
    assert "WeatherAgent" in registry.agents
    assert "ItineraryAgent" in registry.agents
    assert "AttractionAgent" not in registry.agents
    assert registry.requires_tool("ItineraryAgent")
    text = registry.get_all_agents_text()
    assert "ItineraryAgent" in text
    params = registry.get_agent_parameters_text()
    assert "fetch_candidate_pois" in params
    assert "plan_itinerary" in params


def test_guess_agent():
    assert guess_agent("查询北京明天天气") == "WeatherAgent"
    assert guess_agent("推荐附近酒店") == "HotelAgent"
    assert guess_agent("上海有哪些打卡景点") == "ItineraryAgent"


def test_build_time_anchor_next_week():
    anchor = build_time_anchor(ref=date(2026, 6, 11))
    assert anchor["today"] == "2026-06-11"
    assert anchor["today_weekday"] == "周四"
    assert anchor["next_week_start"] == "2026-06-15"
    assert anchor["next_week_end"] == "2026-06-21"


def test_format_time_anchor_block_includes_dates():
    anchor = build_time_anchor(ref=date(2026, 6, 11))
    block = format_time_anchor_block(anchor)
    assert "2026-06-11" in block
    assert "2026-06-15" in block
    assert "禁止臆造" in block
