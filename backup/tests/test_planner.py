"""TaskPlanner 与规划上下文单元测试（不调用 LLM）。"""
from datetime import date

from domains.travel.plan_context import build_time_anchor, format_time_anchor_block
from domains.travel.prompt_bundle import TravelPrompts
from domains.travel.specs import create_travel_registry_stub


def test_parse_pre_survey_sections():
    from agent_framework.domain.parsing import parse_pre_survey

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


def test_travel_registry_lists_five_agents():
    registry = create_travel_registry_stub()
    assert len(registry.agents) == 5
    assert "WeatherAgent" in registry.agents
    assert "ItineraryAgent" in registry.agents
    assert "AttractionAgent" not in registry.agents
    assert registry.requires_tool("ItineraryAgent")
    assert registry.guess_agent("查询北京明天天气") == "WeatherAgent"
    text = registry.get_all_agents_text()
    assert "ItineraryAgent" in text
    params = registry.get_agent_parameters_text()
    assert "fetch_candidate_pois" in params
    assert "plan_itinerary" in params


def test_registry_register_guess_rules():
    registry = create_travel_registry_stub()
    assert registry.guess_agent("推荐附近酒店") == "HotelAgent"
    assert registry.guess_agent("随便聊聊") is None


def test_travel_prompts_build():
    prompts = TravelPrompts.build()
    assert "旅行" in prompts.central_agent_system
    assert "旅行助手" in prompts.aggregation
    assert prompts.multi_task_title


def test_pipeline_config():
    from agent_framework.domain.pipeline import PipelineConfig

    full = PipelineConfig()
    assert full.enable_pre_survey and full.enable_memory
    minimal = PipelineConfig(enable_pre_survey=False, enable_memory=False)
    assert not minimal.enable_pre_survey
    assert not minimal.needs_save_memory


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
