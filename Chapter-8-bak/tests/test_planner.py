"""TaskPlanner 与预调查解析的单元测试（不调用 LLM）。"""
import pytest

from travel_multi_agent.domain.agent_registry import SubAgentRegistry
from travel_multi_agent.domain.parsing import guess_agent, parse_pre_survey

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


def test_agent_registry_lists_six_agents():
    registry = SubAgentRegistry()
    assert len(registry.agents) == 6
    assert "WeatherAgent" in registry.agents
    text = registry.get_all_agents_text()
    assert "HotelAgent" in text


def test_guess_agent():
    assert guess_agent("查询北京明天天气") == "WeatherAgent"
    assert guess_agent("推荐附近酒店") == "HotelAgent"

# 手动执行
test_guess_agent()
