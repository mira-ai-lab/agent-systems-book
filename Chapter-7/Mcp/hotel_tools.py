"""LangChain 酒店工具与 Agent（依赖 hotel_core 的 POI 查询）。"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from hotel_core import recommend_hotel_impl

HOTEL_AGENT_SYSTEM_PROMPT = """你是酒店推荐助手，只能通过工具 recommend_hotel 查酒店。
规则：
1. 从用户话里提取 city（必填）；预算、区域、品牌等写入工具参数（供记录），工具会返回候选列表。
2. 根据工具返回的 hotels 列表，结合用户预算与偏好，由你挑选并只向用户推荐一家最合适的酒店。
3. 非酒店相关问题，回复：我只能协助酒店推荐。"""


def create_llm() -> ChatOpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("请设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY")

    ssl_verify = os.getenv("OPENAI_SSL_VERIFY", "false").lower() not in ("0", "false", "no")
    return ChatOpenAI(
        model=os.getenv("DASHSCOPE_CHAT_MODEL", "qwen-plus"),
        temperature=0,
        api_key=api_key,
        base_url=os.getenv(
            "DASHSCOPE_CHAT_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ).rstrip("/"),
        http_client=httpx.Client(verify=ssl_verify),
    )


@tool
async def recommend_hotel(
    city: str,
    preferences: Optional[str] = None,
    budget_cny_per_night_max: Optional[int] = None,
) -> Dict[str, Any]:
    """查询酒店候选列表（不做规则筛选），由大模型根据返回结果为用户选一家。preferences 可写区名、品牌或地标。"""
    return await recommend_hotel_impl(city, preferences, budget_cny_per_night_max)


def create_hotel_agent() -> Any:
    return create_agent(
        create_llm(),
        tools=[recommend_hotel],
        system_prompt=HOTEL_AGENT_SYSTEM_PROMPT,
        checkpointer=MemorySaver(),
    )


async def run_hotel_agent(user_query: str, *, thread_id: str = "hotel_mcp") -> str:
    agent = create_hotel_agent()
    result = await agent.ainvoke(
        {"messages": [("user", user_query)]},
        {"configurable": {"thread_id": thread_id}},
    )
    messages = result.get("messages") or []
    if not messages:
        return ""
    last = messages[-1]
    content = getattr(last, "content", None)
    if content is not None:
        return str(content)
    return str(last)
