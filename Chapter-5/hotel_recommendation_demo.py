import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from typing import Any, Dict, Optional
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver


# 外部接口：百度 / 高德 Place API（见 Chapter-5/travel_common.py）
from travel_common import create_llm, ensure_project_dotenv_loaded, fetch_hotels_from_api


def _normalize_hotel_row(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": item.get("name"),
        "district": item.get("district"),
        "address": item.get("address"),
        "rating": item.get("rating"),
        "avg_price": item.get("avg_price_cny") or item.get("avg_price"),
        "tel": item.get("tel"),
    }


async def fetch_hotels(city: str, keyword: str = "酒店") -> Dict[str, Any]:
    """调用地图 API 查询酒店；需配置 BAIDU_MAP_AK 或 AMAP_KEY。"""
    ensure_project_dotenv_loaded()
    res = await fetch_hotels_from_api(city, limit=10, keyword=keyword)
    if res.get("error"):
        return {
            "hotels": [],
            "source": "api_error",
            "error": res.get("error"),
            "message": res.get("message") or res.get("error"),
        }
    hotels = [_normalize_hotel_row(h) for h in (res.get("hotels") or []) if isinstance(h, dict)]
    return {
        "hotels": hotels,
        "source": res.get("data_source", "api"),
        "search_query": res.get("search_query"),
    }

# 定义标准工具（@tool 装饰器）
@tool
async def recommend_hotel(
        city: str,
        preferences: Optional[str] = None,
        budget_max: Optional[int] = None
) -> Dict[str, Any]:
    """
    查询指定城市的酒店候选列表
    city: 目标城市（必填）
    preferences: 酒店偏好，如近景区、安静、品牌等
    budget_max: 单晚最高预算，单位：元
    """
    search_key = f"{preferences} 酒店" if preferences else "酒店"
    result = await fetch_hotels(city, search_key)
    payload: Dict[str, Any] = {
        "city": city,
        "preferences": preferences,
        "budget_max": budget_max,
        "hotel_list": result["hotels"],
        "data_source": result["source"],
    }
    if result.get("search_query"):
        payload["search_query"] = result["search_query"]
    if result.get("error"):
        payload["error"] = result["error"]
        payload["message"] = result.get("message")
    return payload

# 组装带记忆的工具 Agent
llm = create_llm()
memory = MemorySaver()
agent = create_agent(
    llm,
    tools=[recommend_hotel],
    system_prompt="""你是酒店推荐助手，只能通过工具 recommend_hotel 查酒店。
规则：
1. 从用户话里提取 city（必填）；预算、区域、品牌等写入工具参数（供记录），工具会返回候选列表。
2. 根据工具返回的 hotel_list 列表，结合用户预算与偏好，由你挑选并只向用户推荐一家最合适的酒店。
3. 非酒店相关问题，回复：我只能协助酒店推荐。""",
    checkpointer=memory
)


# 流式执行与工具调用事件监听（stream_mode=messages 兼容新版 create_agent）
def _content_to_str(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content) if content else ""


async def run_hotel_agent(user_query: str, thread_id: str = "demo_01"):
    inputs = {"messages": [("user", user_query)]}
    config = {"configurable": {"thread_id": thread_id}}
    seen_tool_starts: set[str] = set()
    ai_text_buffer = ""

    async for msg, _meta in agent.astream(inputs, config, stream_mode="messages"):
        if isinstance(msg, (AIMessage, AIMessageChunk)):
            for tc in msg.tool_calls or []:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
                if name and name not in seen_tool_starts:
                    seen_tool_starts.add(name)
                    yield f"\n[调用工具：{name}]\n"
            if msg.tool_calls:
                continue
            text = _content_to_str(msg.content)
            if not text:
                continue
            if text.startswith(ai_text_buffer):
                delta = text[len(ai_text_buffer) :]
                ai_text_buffer = text
            else:
                delta = text
                ai_text_buffer += text
            if delta:
                yield delta
        elif isinstance(msg, ToolMessage):
            yield "\n[工具返回结果]\n"

USER_QUERY = "帮我推荐杭州西湖附近的酒店，预算每晚不超过500元"


async def main() -> None:
    print(f"用户: {USER_QUERY}\n")
    print("模型回答: ", end="", flush=True)
    async for chunk in run_hotel_agent(USER_QUERY):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())