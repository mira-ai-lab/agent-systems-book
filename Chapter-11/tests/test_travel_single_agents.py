"""Travel 五个子智能体的单测（工具层 + 可选 live 集成）。

测试分层
--------
1. **Fixture / Registry**：JSON 用例能否加载，agent/tool 是否在 registry 中注册。
2. **Tool + Mock API**（默认 CI 跑这条）：直接 ``tool.ainvoke(tool_args)``，patch 外部 HTTP/MCP，
   不经过 LLM、不经过 Router/TaskPlanner。
3. **Integration**（``@pytest.mark.integration``，需 API Key）：
   - ``test_flight_tool_live_*``：仅真实航班 API，仍不经过 LLM。
   - ``test_single_agent_live_invoke``：完整 ReAct Agent（LLM + 真实工具）。

Fixtures
--------
``tests/fixtures/travel_single_agent_cases.json`` — 每个 Agent 2 条，共 10 条 case。

运行（在 Chapter-11 目录下）::

    python -m pytest tests/test_travel_single_agents.py -q
    python -m pytest tests/test_travel_single_agents.py -k flight -q
    python -m pytest tests/test_travel_single_agents.py -k "flight-beijing-sanya" -s -v   # debug 单条
    python -m pytest tests/test_travel_single_agents.py -m integration -q   # 需 Key

环境变量
--------
- LLM live：``DASHSCOPE_API_KEY`` 或 ``OPENAI_API_KEY``
- 航班 live：``VARIFLIGHT_API_KEY`` / ``X_VARIFLIGHT_KEY`` / ``AVIATIONSTACK_KEY``
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Callable, Dict, List
from unittest.mock import AsyncMock, patch

import pytest

from agent_framework.config import load_project_dotenv
from domains.travel.specs import TRAVEL_AGENT_SPECS, create_travel_registry_stub
from tests.travel_agents.cases import SingleAgentCase, load_single_agent_cases

load_project_dotenv()

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

# integration 用例在无 Key 时 skip，不影响默认 mock 测试
_HAS_LLM_KEY = bool(os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY"))
_HAS_FLIGHT_API_KEY = bool(
    os.getenv("VARIFLIGHT_API_KEY") or os.getenv("X_VARIFLIGHT_KEY") or os.getenv("AVIATIONSTACK_KEY")
)


# ---------------------------------------------------------------------------
# 断言 / 解析辅助
# ---------------------------------------------------------------------------


def _tool_result_ok(result: Any) -> bool:
    """Travel tool 约定：成功返回 dict 且无 error 字段。"""
    return isinstance(result, dict) and not result.get("error")


def _extract_ai_text(state: Dict[str, Any]) -> str:
    """从 LangGraph Agent 终态 messages 里拼接 AI 文本（live invoke 用）。"""
    messages = state.get("messages") or []
    chunks: List[str] = []
    for msg in messages:
        if getattr(msg, "type", None) == "ai" and getattr(msg, "content", None):
            chunks.append(str(msg.content))
    return "\n".join(chunks)


# ---------------------------------------------------------------------------
# Mock 外部 API 返回值（形状需与各 agent tool 期望一致）
# ---------------------------------------------------------------------------


def _mock_flight_payload(case: SingleAgentCase) -> Dict[str, Any]:
    args = case.tool_args
    return {
        "departure": args["departure"],
        "arrival": args["arrival"],
        "date": args["date"],
        "flights": [
            {
                "flight_no": "CA1357",
                "departure": args["departure"],
                "arrival": args["arrival"],
                "dep_time": "08:30",
                "arr_time": "12:45",
                "price_cny": 1280,
            }
        ],
        "data_source": "mock/variflight",
    }


def _mock_weather_payload(case: SingleAgentCase) -> Dict[str, Any]:
    if case.tool == "get_weather_forecast":
        city = case.tool_args["city"]
        days = int(case.tool_args.get("days") or 7)
        return {
            "city": city,
            "days": days,
            "forecasts": [
                {"date": f"2026-06-{20 + i:02d}", "condition": "晴", "temp_high_c": 30, "temp_low_c": 22}
                for i in range(days)
            ],
            "data_source": "mock/mcp",
        }
    return {
        "city": case.tool_args["city"],
        "date": "2026-06-23",
        "forecast": {"condition": "晴", "temp_high_c": 28, "temp_low_c": 20},
        "data_source": "mock/amap",
    }


def _mock_hotel_payload(case: SingleAgentCase) -> Dict[str, Any]:
    city = case.tool_args["city"]
    return {
        "city": city,
        "search_query": case.tool_args.get("preferences") or "酒店",
        "hotels": [
            {"name": f"{city}示例酒店", "district": "示例区", "avg_price_cny": 420, "rating": 4.6}
        ],
        "data_source": "mock/amap",
    }


def _mock_restaurant_payload(case: SingleAgentCase) -> Dict[str, Any]:
    location = case.tool_args["location"]
    return {
        "location": location,
        "restaurants": [
            {"name": f"{location}老字号", "cuisine": case.tool_args.get("cuisine") or "本地菜", "rating": 4.5}
        ],
        "data_source": "mock/amap",
    }


def _mock_itinerary_payload(case: SingleAgentCase) -> Dict[str, Any]:
    if case.tool == "fetch_candidate_pois":
        city = case.tool_args["city"]
        return {
            "city": city,
            "attractions": [
                {"name": "西湖", "type": "景点"},
                {"name": "灵隐寺", "type": "景点"},
            ],
            "data_source": "mock/amap",
        }
    city = case.tool_args["city"]
    days = int(case.tool_args.get("days") or 3)
    return {
        "city": city,
        "days": days,
        "plan": [{"day": i + 1, "items": [f"{city}景点{i + 1}"]} for i in range(days)],
        "poi_count": len(case.tool_args.get("candidate_pois") or []),
        "transportation": "地铁+步行",
        "data_source": "mock/planner",
    }


_MOCK_BUILDERS: Dict[str, Callable[[SingleAgentCase], Dict[str, Any]]] = {
    "FlightAgent": _mock_flight_payload,
    "WeatherAgent": _mock_weather_payload,
    "HotelAgent": _mock_hotel_payload,
    "RestaurantAgent": _mock_restaurant_payload,
    "ItineraryAgent": _mock_itinerary_payload,
}


# ---------------------------------------------------------------------------
# Agent / Tool 解析（按 fixture 里的 agent_name + tool 名定位实现）
# ---------------------------------------------------------------------------


def _resolve_tool(agent_name: str, tool_name: str):
    """返回 LangChain StructuredTool，供 ``tool.ainvoke(tool_args)`` 直接调用。"""
    if agent_name == "FlightAgent":
        from domains.travel.agents.flight import search_flights

        return search_flights
    if agent_name == "WeatherAgent":
        from domains.travel.agents import weather as weather_mod

        return getattr(weather_mod, tool_name)
    if agent_name == "HotelAgent":
        from domains.travel.agents.hotel import recommend_hotel

        return recommend_hotel
    if agent_name == "RestaurantAgent":
        from domains.travel.agents.restaurant import recommend_restaurant

        return recommend_restaurant
    if agent_name == "ItineraryAgent":
        from domains.travel.agents import itinerary as itinerary_mod

        return getattr(itinerary_mod, tool_name)
    raise KeyError(f"unknown agent: {agent_name}")


def _create_agent(agent_name: str):
    """创建完整 ReAct Agent 图（仅 live invoke 测试使用）。"""
    creators = {
        "FlightAgent": "domains.travel.agents.flight.create_flight_agent",
        "WeatherAgent": "domains.travel.agents.weather.create_weather_agent",
        "HotelAgent": "domains.travel.agents.hotel.create_hotel_agent",
        "RestaurantAgent": "domains.travel.agents.restaurant.create_restaurant_agent",
        "ItineraryAgent": "domains.travel.agents.itinerary.create_itinerary_agent",
    }
    import importlib

    module_path, func_name = creators[agent_name].rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, func_name)()


def _patch_targets(agent_name: str, tool_name: str) -> List[str]:
    """每个 tool 在 agent 模块内实际调用的 fetch 函数，patch 此处即可 mock 外部 API。

    注意：路径必须是 ``domains.travel.agents.*`` 模块内被引用的符号，
    否则 patch 不生效，测试会误走 live 网络。
    """
    if agent_name == "FlightAgent":
        return [
            "domains.travel.agents.flight.fetch_flights_from_variflight_api",
            "domains.travel.agents.flight.fetch_flights_from_api",
        ]
    if agent_name == "WeatherAgent" and tool_name == "get_weather":
        return [
            "domains.travel.agents.weather.fetch_weather_via_mcp",
            "domains.travel.agents.weather.amap_weather_by_city_and_date",
        ]
    if agent_name == "WeatherAgent":
        return ["domains.travel.agents.weather.fetch_weather_forecast_via_mcp"]
    if agent_name == "HotelAgent":
        return ["domains.travel.agents.hotel.fetch_hotels_from_api"]
    if agent_name == "RestaurantAgent":
        return ["domains.travel.agents.restaurant.fetch_restaurants_from_api"]
    if agent_name == "ItineraryAgent" and tool_name == "fetch_candidate_pois":
        return ["domains.travel.agents.itinerary.fetch_attractions_from_api"]
    if agent_name == "ItineraryAgent":
        return ["domains.travel.agents.itinerary.build_itinerary_from_candidates"]
    return []


@pytest.fixture(scope="module")
def single_agent_fixtures():
    """全文件共享 fixture，避免每条 case 重复读 JSON。"""
    return load_single_agent_cases()


# ---------------------------------------------------------------------------
# Layer 1：Fixture / Registry 冒烟
# ---------------------------------------------------------------------------


def test_load_single_agent_cases(single_agent_fixtures):
    """JSON 结构正确，且 Flight 样例可读。"""
    assert single_agent_fixtures.locale == "zh"
    assert len(single_agent_fixtures.cases) == 10
    flight_cases = single_agent_fixtures.cases_for_agent("FlightAgent")
    assert flight_cases[0].case_id == "flight-beijing-sanya-jun25"
    assert "北京飞三亚" in flight_cases[0].user_query or "北京" in flight_cases[0].user_query


def test_registry_covers_all_single_agent_cases(single_agent_fixtures):
    """fixture 中的 agent/tool 必须在 TRAVEL_AGENT_SPECS 与 registry 里存在。"""
    registry = create_travel_registry_stub()
    for case in single_agent_fixtures.cases:
        assert case.agent_name in TRAVEL_AGENT_SPECS
        assert case.agent_name in registry.get_agent_names()
        skill_names = {skill["name"] for skill in TRAVEL_AGENT_SPECS[case.agent_name]["skills"]}
        assert case.tool in skill_names


# ---------------------------------------------------------------------------
# Layer 2：Tool + Mock API（默认 CI / 本地回归主路径）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case_id",
    [
        "flight-beijing-sanya-jun25",
        "flight-shanghai-chengdu-jun20",
        "weather-beijing-tomorrow",
        "weather-xian-forecast-7d",
        "hotel-sanya-haitang",
        "hotel-xian-center",
        "restaurant-xian-local",
        "restaurant-hangzhou-west-lake",
        "itinerary-hangzhou-3d",
        "itinerary-shenzhen-hangzhou-5d",
    ],
)
def test_single_agent_tool_with_mock_api(case_id: str, single_agent_fixtures):
    """核心单测：patch 外部 API → 直接 invoke tool → 断言返回结构。

    新增 case 时：1) 改 JSON  2) 把 case_id 加入上方 parametrize 列表。
    """
    case = next(item for item in single_agent_fixtures.cases if item.case_id == case_id)
    tool = _resolve_tool(case.agent_name, case.tool)
    mock_payload = _MOCK_BUILDERS[case.agent_name](case)
    targets = _patch_targets(case.agent_name, case.tool)

    async def _run():
        patchers = [patch(target, new=AsyncMock(return_value=mock_payload)) for target in targets]
        # plan_itinerary 走同步 builder，需单独 patch
        if case.agent_name == "ItineraryAgent" and case.tool == "plan_itinerary":
            patchers = [
                patch(
                    "domains.travel.agents.itinerary.build_itinerary_from_candidates",
                    return_value=mock_payload,
                )
            ]
        for patcher in patchers:
            patcher.start()
        try:
            return await tool.ainvoke(case.tool_args)
        finally:
            for patcher in patchers:
                patcher.stop()

    result = asyncio.run(_run())
    assert _tool_result_ok(result), result
    if case.agent_name == "FlightAgent":
        assert result.get("departure") == case.tool_args["departure"]
        assert result.get("arrival") == case.tool_args["arrival"]
        assert result.get("flights")


@pytest.mark.parametrize("case_id", ["flight-beijing-sanya-jun25"])
def test_flight_agent_user_query_matches_case(case_id: str, single_agent_fixtures):
    """锁定 fixture 文案与 tool_args，防止 JSON 被误改导致 live/mock 用例不一致。"""
    case = next(item for item in single_agent_fixtures.cases if item.case_id == case_id)
    assert case.user_query == "查 6 月 25 日北京飞三亚的机票"
    assert case.tool_args == {
        "departure": "北京",
        "arrival": "三亚",
        "date": "2026-06-25",
    }


# ---------------------------------------------------------------------------
# Layer 3：Integration（需 Key；CI 默认 skip）
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_FLIGHT_API_KEY, reason="需要 VARIFLIGHT_API_KEY 或 AVIATIONSTACK_KEY")
def test_flight_tool_live_beijing_sanya(single_agent_fixtures):
    """真实航班 API：北京 → 三亚。只测 search_flights，不经过 LLM。"""
    case = next(item for item in single_agent_fixtures.cases if item.case_id == "flight-beijing-sanya-jun25")
    tool = _resolve_tool(case.agent_name, case.tool)

    async def _run():
        return await tool.ainvoke(case.tool_args)

    result = asyncio.run(_run())
    assert _tool_result_ok(result), result
    assert result.get("flights"), "航班列表为空"


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_LLM_KEY, reason="需要 DASHSCOPE_API_KEY 或 OPENAI_API_KEY")
@pytest.mark.parametrize(
    "case_id",
    [
        "flight-beijing-sanya-jun25",
        "weather-beijing-tomorrow",
        "hotel-sanya-haitang",
    ],
)
def test_single_agent_live_invoke(case_id: str, single_agent_fixtures):
    """Live：user_query → ReAct Agent → LLM 选 tool → 真实外部 API → 自然语言回复。

    断言 response_keywords 出现在最终 AI 文本中（非 trace preview）。
    """
    from agent_framework.config import create_llm, load_project_dotenv

    load_project_dotenv()
    configure = __import__(
        "domains.travel.agents.base",
        fromlist=["configure_agent_llm"],
    ).configure_agent_llm
    configure(create_llm(temperature=0))

    case = next(item for item in single_agent_fixtures.cases if item.case_id == case_id)
    agent = _create_agent(case.agent_name)

    async def _run():
        return await agent.ainvoke(
            {"messages": [("user", case.user_query)]},
            {"configurable": {"thread_id": f"single-agent-{case.case_id}"}},
        )

    state = asyncio.run(_run())
    final_text = _extract_ai_text(state)
    assert final_text.strip()
    for keyword in case.response_keywords:
        assert keyword in final_text, f"回复缺少关键词 {keyword!r}: {final_text[:200]}"
