"""子 Agent 运行时：构建 Agent、渲染 prompt、同步 invoke。

Agent-B1 从 FlightAgent 起步；Agent-B2 扩展为 5 个子 Agent 并列优化。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI

from agent_framework.domain.locale_loader import agent_fragment, agent_system_prompt
from agent_framework.infra.agent_runtime import build_agent
from domains.travel.plan_context import build_time_anchor, format_time_anchor_block

from .scorer import extract_ai_text, extract_invoked_tool_names

# Agent-B2：可优化的 5 个旅行子 Agent（与 specs / fixtures 一致）
TRAVEL_OPTIMIZABLE_AGENTS: Tuple[str, ...] = (
    "FlightAgent",
    "WeatherAgent",
    "HotelAgent",
    "RestaurantAgent",
    "ItineraryAgent",
)

# 各 Agent 模板必须保留的占位符（与 locales/zh.json 一致，optimizer 不得删掉）
AGENT_REQUIRED_PLACEHOLDERS: Dict[str, Tuple[str, ...]] = {
    "FlightAgent": ("{today}",),
    "WeatherAgent": ("{today}", "{time_anchor_rules}", "{multi_entity_rules}"),
    "HotelAgent": ("{time_anchor_rules}", "{multi_entity_rules}"),
    "RestaurantAgent": ("{time_anchor_rules}", "{multi_entity_rules}"),
    "ItineraryAgent": ("{today}", "{time_anchor_rules}", "{multi_entity_rules}"),
}

# 向后兼容 B1 常量名
FLIGHT_AGENT_REQUIRED_PLACEHOLDERS = AGENT_REQUIRED_PLACEHOLDERS["FlightAgent"]
COMMON_AGENT_PLACEHOLDERS = ("{today}", "{time_anchor_rules}", "{multi_entity_rules}", "{time_anchor}")


def get_agent_prompt_template(agent_name: str, *, locale: str = "zh") -> str:
    """读取 locales 中的原始 system_prompt 模板（未 format）。"""
    return agent_system_prompt("travel", agent_name, locale)


def render_agent_prompt_template(template: str, *, locale: str = "zh") -> str:
    """将模板渲染为可传给 build_agent 的最终 system_prompt。"""
    from datetime import datetime

    today = datetime.now().strftime("%Y-%m-%d")
    try:
        time_rules = agent_fragment("travel", "time_anchor_rules", locale)
    except KeyError:
        time_rules = ""
    try:
        multi_rules = agent_fragment("travel", "multi_entity_rules", locale)
    except KeyError:
        multi_rules = ""
    time_anchor = format_time_anchor_block(build_time_anchor())
    return template.format(
        today=today,
        time_anchor=time_anchor,
        time_anchor_rules=time_rules.format(time_anchor=time_anchor) if "{time_anchor}" in time_rules else time_rules,
        multi_entity_rules=multi_rules,
    )


def extract_agent_system_prompt(raw_text: str, *, agent_name: str = "FlightAgent") -> str:
    """清洗 optimizer 输出，并校验该 Agent 的关键占位符。"""
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("optimizer 返回空 agent system_prompt")

    fenced = re.search(r"```(?:markdown|text|prompt)?\s*([\s\S]*?)```", text)
    if fenced:
        text = fenced.group(1).strip()

    required = AGENT_REQUIRED_PLACEHOLDERS.get(agent_name, ())
    missing = [token for token in required if token not in text]
    if missing:
        raise ValueError(f"{agent_name} prompt 缺少占位符: {missing}")
    return text


def _agent_tools(agent_name: str) -> list:
    """按 Agent 名懒加载工具列表（与生产 create_*_agent 一致）。"""
    if agent_name == "FlightAgent":
        from domains.travel.agents.flight import search_flights

        return [search_flights]
    if agent_name == "WeatherAgent":
        from domains.travel.agents.weather import get_weather, get_weather_forecast

        return [get_weather_forecast, get_weather]
    if agent_name == "HotelAgent":
        from domains.travel.agents.hotel import recommend_hotel

        return [recommend_hotel]
    if agent_name == "RestaurantAgent":
        from domains.travel.agents.restaurant import recommend_restaurant

        return [recommend_restaurant]
    if agent_name == "ItineraryAgent":
        from domains.travel.agents.itinerary import fetch_candidate_pois, plan_itinerary

        return [fetch_candidate_pois, plan_itinerary]
    raise ValueError(f"不支持的 agent_name={agent_name!r}，可选: {TRAVEL_OPTIMIZABLE_AGENTS}")


def build_travel_agent(
    agent_name: str,
    system_prompt_template: str,
    llm: ChatOpenAI,
    *,
    locale: str = "zh",
) -> Any:
    """用指定 prompt 模板构建任意旅行子 Agent（与生产 create_*_agent 同工具）。"""
    rendered = render_agent_prompt_template(system_prompt_template, locale=locale)
    return build_agent(_agent_tools(agent_name), rendered, llm=llm)


def build_flight_agent(
    system_prompt_template: str,
    llm: ChatOpenAI,
    *,
    locale: str = "zh",
) -> Any:
    """B1 兼容：构建 FlightAgent。"""
    return build_travel_agent("FlightAgent", system_prompt_template, llm, locale=locale)


def default_agent_prompt_template(agent_name: str, *, locale: str = "zh") -> str:
    """默认 Agent 模板（优先 optimized override，否则 locales）。"""
    if agent_name not in TRAVEL_OPTIMIZABLE_AGENTS:
        raise ValueError(f"不支持的 agent_name={agent_name!r}")
    from agent_framework.optimization.agent_prompt_store import load_optimized_agent_prompt_template

    override = load_optimized_agent_prompt_template(agent_name, locale=locale)
    if override:
        return override
    return get_agent_prompt_template(agent_name, locale=locale)


def default_flight_prompt_template(*, locale: str = "zh") -> str:
    """B1 兼容：FlightAgent 默认模板。"""
    return default_agent_prompt_template("FlightAgent", locale=locale)


def default_flight_system_prompt(*, locale: str = "zh") -> str:
    """运行时最终 system_prompt（含 optimized override）。"""
    from domains.travel.agents.prompt_loader import travel_agent_prompt

    return travel_agent_prompt("FlightAgent", locale=locale)


class AgentSyncBridge:
    """将子 Agent ``ainvoke`` 包装为 sync，供 textgrad StringBasedFunction 使用。"""

    def __init__(self, *, llm: ChatOpenAI, locale: str = "zh", agent_name: str = "FlightAgent"):
        self._llm = llm
        self._locale = locale
        if agent_name not in TRAVEL_OPTIMIZABLE_AGENTS:
            raise ValueError(f"不支持的 agent_name={agent_name!r}")
        self._agent_name = agent_name

    @property
    def agent_name(self) -> str:
        return self._agent_name

    @staticmethod
    def _run_async(coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coro).result()

    def invoke(
        self,
        *,
        system_prompt_template: str,
        user_query: str,
        thread_id: str,
    ) -> Dict[str, Any]:
        """同步执行单 Agent，返回状态 dict（含 messages）。"""
        agent = build_travel_agent(
            self._agent_name,
            system_prompt_template,
            self._llm,
            locale=self._locale,
        )
        config = {"configurable": {"thread_id": thread_id}}

        async def _call():
            return await agent.ainvoke({"messages": [("user", user_query)]}, config)

        state = self._run_async(_call())
        return state if isinstance(state, dict) else dict(state)

    @staticmethod
    def format_agent_output(state: Dict[str, Any]) -> str:
        """序列化 Agent 输出供 MultiFieldEvaluation 使用。"""
        payload = {
            "final_response": extract_ai_text(state),
            "invoked_tools": extract_invoked_tool_names(state),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)
