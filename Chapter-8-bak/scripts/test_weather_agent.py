"""手动测试 WeatherAgent 与 get_weather 工具。"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from travel_multi_agent.config import load_project_dotenv

def _safe_print(text: str) -> None:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    safe = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(safe)


def _mask_key(name: str) -> str:
    value = os.getenv(name, "")
    if not value:
        return "未设置"
    return f"已设置 ({len(value)} chars)"


async def test_get_weather_tool() -> bool:
    from travel_multi_agent.agents.weather import get_weather

    print("\n=== 1. get_weather 工具（上海，明天）===")
    result = await get_weather.ainvoke({"city": "上海", "date": "明天"})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if isinstance(result, dict) and result.get("error"):
        return False
    return bool(result)


async def test_weather_agent() -> bool:
    from travel_multi_agent.agents.weather import create_weather_agent

    print("\n=== 2. WeatherAgent 完整调用 ===")
    agent = create_weather_agent()
    state = await agent.ainvoke(
        {"messages": [("user", "查询上海明天的天气")]},
        {"configurable": {"thread_id": "weather_test_001"}},
    )

    messages = state.get("messages", [])
    print(f"消息数: {len(messages)}")
    for msg in messages:
        msg_type = getattr(msg, "type", type(msg).__name__)
        name = getattr(msg, "name", "")
        content = getattr(msg, "content", "")
        if isinstance(content, str) and len(content) > 300:
            preview = content[:300] + "..."
        else:
            preview = content
        tool_calls = getattr(msg, "tool_calls", None)
        suffix = ""
        if name:
            suffix += f"/{name}"
        if tool_calls:
            suffix += f" tool_calls={len(tool_calls)}"
        _safe_print(f"  [{msg_type}{suffix}] {preview}")

    ai_messages = [
        m
        for m in messages
        if getattr(m, "type", None) == "ai" and getattr(m, "content", None)
    ]
    if not ai_messages:
        print("\n=== 无 AI 文本回复 ===")
        return False

    _safe_print("\n=== 最终 AI 回复 ===")
    _safe_print(ai_messages[-1].content)
    return True


async def main() -> int:
    load_project_dotenv()
    print("=== 环境变量 ===")
    for key in (
        "DASHSCOPE_API_KEY",
        "OPENAI_API_KEY",
        "WEATHERAPI_KEY",
        "AMAP_KEY",
        "WEATHER_USE_MCP",
    ):
        print(f"{key}: {_mask_key(key)}")

    tool_ok = await test_get_weather_tool()
    print(f"\n工具层结果: {'PASS' if tool_ok else 'FAIL'}")

    if not os.getenv("DASHSCOPE_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("跳过 Agent 测试：未配置 DASHSCOPE_API_KEY / OPENAI_API_KEY")
        return 0 if tool_ok else 1

    agent_ok = await test_weather_agent()
    print(f"\nAgent 层结果: {'PASS' if agent_ok else 'FAIL'}")
    return 0 if tool_ok and agent_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
