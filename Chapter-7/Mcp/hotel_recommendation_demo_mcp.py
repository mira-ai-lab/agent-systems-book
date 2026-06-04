"""
酒店推荐 Agent 示例 - 工具经 MCP 协议从 hotel_mcp_server 加载

读者只需抓住一条链路：
  用户提问 → Agent（大模型）→ MCP recommend_hotel_tool → 百度地图查酒店 → 模型组织回答

前置（需先启动 MCP 服务）：
  python hotel_mcp_server.py          # 默认 http://127.0.0.1:8765/sse

依赖（书根目录 .env）：
  DASHSCOPE_API_KEY=...        # 大模型
  BAIDU_MAP_AK=...             # 百度地图 Place API（在 MCP Server 侧使用）
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, List

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
import httpx
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from hotel_mcp_client import fetch_hotels_via_mcp, sse_url

USER_QUERY = "我要去大同玩三天给我推荐酒店，需要近景区"

# 工具名与 MCP Server 中 @Mcp.tool 注册名一致
SYSTEM_INSTRUCTION = """你是酒店推荐助手，只能通过 MCP 工具 recommend_hotel_tool 查酒店。
规则：
1. 从用户话里提取 city（必填）；预算、区域、品牌等写入工具参数，工具会返回候选列表。
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


def _mcp_client() -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            "hotel": {
                "transport": "sse",
                "url": sse_url(),
                "timeout": 10.0,
                "sse_read_timeout": 300.0,
            }
        }
    )


async def load_mcp_tools() -> List[BaseTool]:
    """从 hotel_mcp_server 拉取 MCP 工具，转为 LangChain Tool。"""
    client = _mcp_client()
    tools = await client.get_tools()
    if not tools:
        raise RuntimeError(
            f"未从 MCP 获取到任何工具，请先启动: python hotel_mcp_server.py  (SSE: {sse_url()})"
        )
    return tools


async def create_hotel_agent_with_mcp_tools() -> Any:
    tools = await load_mcp_tools()
    return create_agent(
        create_llm(),
        tools=tools,
        system_prompt=SYSTEM_INSTRUCTION,
        checkpointer=MemorySaver(),
    )


def _short_json(data: Any) -> str:
    if hasattr(data, "content"):
        data = data.content
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return data[:800]
    text = json.dumps(data, ensure_ascii=False, indent=2) if isinstance(data, (dict, list)) else str(data)
    return text if len(text) <= 800 else text[:800] + "…"


async def demo_direct_mcp_tool() -> None:
    """对比：不经过 Agent，程序员直接调 MCP 工具。"""
    print("=" * 60)
    print("【对比】直接调用 MCP 工具 recommend_hotel_tool")
    print("=" * 60)
    result = await fetch_hotels_via_mcp("大同")
    if not result:
        print("调用失败，请确认 hotel_mcp_server.py 已启动")
        print()
        return
    print(f"SSE: {sse_url()}")
    print(f"搜索词: {result.get('search_query')}")
    print(f"共 {len(result.get('hotels') or [])} 家候选（交给模型挑选，工具不做过滤）")
    for i, h in enumerate((result.get("hotels") or []), 1):
        print(f"  {i}. {h.get('name')} | {h.get('district') or h.get('address')}")
    print()


async def demo_agent_calls_mcp_tool() -> None:
    """正文：Agent 绑定 MCP 工具，由模型自动决定调用 recommend_hotel_tool。"""
    print("=" * 60)
    print("【正文】Agent + MCP 工具自动调用")
    print("=" * 60)
    print(f"用户: {USER_QUERY}\n")

    agent = await create_hotel_agent_with_mcp_tools()
    inputs = {"messages": [("user", USER_QUERY)]}
    config = {"configurable": {"thread_id": "book_demo_mcp"}}

    print("模型回答: ", end="", flush=True)
    streamed = False
    async for event in agent.astream_events(inputs, config, version="v2"):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"].content
            if chunk:
                if isinstance(chunk, list):
                    chunk = "".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in chunk
                    )
                print(chunk, end="", flush=True)
                streamed = True

        elif kind == "on_tool_start":
            print(f"\n\n>>> MCP 工具调用: {event.get('name')}")
            print(f">>> 参数: {_short_json(event['data'].get('input'))}")

        elif kind == "on_tool_end":
            print(f">>> 工具返回: {_short_json(event['data'].get('output'))}\n")
            print("模型继续: ", end="", flush=True)

        elif kind == "on_chain_end" and event.get("name") == "LangGraph":
            if not streamed:
                out = event.get("data", {}).get("output") or {}
                messages = out.get("messages") if isinstance(out, dict) else None
                if messages:
                    last = messages[-1]
                    content = getattr(last, "content", None) or ""
                    if content:
                        print(content, end="", flush=True)
            break

    print("\n")


async def main() -> None:
    print(f"MCP SSE 地址: {sse_url()}\n")
    await demo_direct_mcp_tool()
    await demo_agent_calls_mcp_tool()


if __name__ == "__main__":
    asyncio.run(main())

# MCP SSE 地址: http://127.0.0.1:8765/sse
#
# ============================================================
# 【对比】直接调用 MCP 工具 recommend_hotel_tool
# ============================================================
# SSE: http://127.0.0.1:8765/sse
# 搜索词: 酒店
# 共 10 家候选（交给模型挑选，工具不做过滤）
#   1. 天镇久天宾馆 | 天镇县
#   2. 佳园宾馆 | 云冈区
#   3. 大同海波诚信驿站民宿 | 平城区
#   4. 子鼠丑牛客栈(大同古城店) | 平城区
#   5. 家馨宾馆 | 广灵县
#   6. 大同朴宿微澜民宿 | 平城区
#   7. 大同碧海情缘公寓 | 平城区
#   8. 微风轻语民宿 | 平城区
#   9. 闲然之家民宿 | 平城区
#   10. 玥庭兰舍民宿 | 浑源县
#
# ============================================================
# 【正文】Agent + MCP 工具自动调用
# ============================================================
# 用户: 我要去大同玩三天给我推荐酒店，需要近景区
#
# 模型回答:
#
# >>> MCP 工具调用: recommend_hotel_tool
# >>> 参数: {
#   "city": "大同",
#   "preferences": "近景区"
# }
# >>> 工具返回: [
#   {
#     "type": "text",
#     "text": "{\n  \"city\": \"大同\",\n  \"search_query\": \"近景区 酒店\",\n  \"preferences\": \"近景区\",\n  \"budget_cny_per_night_max\": null,\n  \"hotels\": [\n    {\n      \"name\": \"浑源旨岭宜景酒店(北岳恒山景区真武庙店)\",\n      \"district\": \"浑源县\",\n      \"address\": \"恒山景区真武庙旁边\",\n      \"tel\": \"15525220428\",\n      \"location\": \"113.740723,39.668781\",\n      \"rating\": 4.8,\n      \"avg_price_cny\": null,\n      \"type\": \"hotel\"\n    },\n    {\n      \"name\": \"遇见·恒山\",\n      \"district\": \"浑源县\",\n      \"address\": \"大同市恒山北路国际绿洲·和园七号楼一单元801\",\n      \"tel\": \"13935273488\",\n      \"location\": \"113.70018,39.71848\",\n      \"rating\": 4.8,\n      \"avg_price_cny\": null,\n      \"type\": \"hotel\"\n    },\n    {\n      \"name\": \"瑞福民宿(大同古城景区店)\",\n      \…
#
# 模型继续: 根据您“去大同玩三天、需近景区”的需求，我为您精选推荐：
#
# **瑞福民宿（大同古城景区店）**
# 📍地址：山西省大同市平城区古城街道云路街20号（就在大同古城内，步行可达华严寺、善化寺、代王府、城墙等核心景点）
# ⭐评分：4.7分（高口碑）
# 📞电话：15303524888
# ✅优势：真正“近景区”——地处古城核心区，免去交通耗时，方便您深度体验大同千年古都风貌；民宿风格温馨，位置安静又便利。
#
# 适合您三天沉浸式游览古城及周边精华。需要帮您查询价格或预订方式，可随时告诉我！