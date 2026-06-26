#!/usr/bin/env python3
"""手动测试 FlightAgent / search_flights 工具（真实 API + 可选 LLM）。

用法::

    python scripts/flight_agent_test.py
    python scripts/flight_agent_test.py --query "查 6 月 25 日北京飞三亚的机票"
    python scripts/flight_agent_test.py --tool-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent_framework.config import create_llm, load_project_dotenv
from tests.travel_agents.cases import load_single_agent_cases


def _mask_key(name: str) -> str:
    value = os.getenv(name, "")
    if not value:
        return "未设置"
    return f"已设置 ({len(value)} chars)"


def _has_error(result) -> bool:
    return isinstance(result, dict) and bool(result.get("error"))


async def test_search_flights_tool(
    *,
    departure: str,
    arrival: str,
    date: str,
) -> bool:
    from domains.travel.agents.flight import search_flights

    print(f"\n=== 1. search_flights 工具（{departure} → {arrival}，{date}）===")
    result = await search_flights.ainvoke(
        {"departure": departure, "arrival": arrival, "date": date}
    )
    print(json.dumps(result, ensure_ascii=False, indent=2)[:2000])
    if _has_error(result):
        print("\n工具层: FAIL")
        if result.get("hint"):
            print(f"  hint: {result['hint']}")
        return False
    flights = result.get("flights") or []
    print(f"\n工具层: PASS（{len(flights)} 条航班，source={result.get('data_source')}）")
    return bool(flights)


async def test_flight_agent(user_query: str, thread_id: str) -> bool:
    from domains.travel.agents.flight import create_flight_agent

    print(f"\n=== 2. FlightAgent 完整调用 ===")
    print(f"user_query: {user_query}")
    configure = __import__(
        "domains.travel.agents.base",
        fromlist=["configure_agent_llm"],
    ).configure_agent_llm
    configure(create_llm(temperature=0))

    agent = create_flight_agent()
    state = await agent.ainvoke(
        {"messages": [("user", user_query)]},
        {"configurable": {"thread_id": thread_id}},
    )

    messages = state.get("messages") or []
    print(f"消息数: {len(messages)}")
    for msg in messages:
        msg_type = getattr(msg, "type", type(msg).__name__)
        name = getattr(msg, "name", "")
        content = getattr(msg, "content", "")
        tool_calls = getattr(msg, "tool_calls", None)
        suffix = f"/{name}" if name else ""
        if tool_calls:
            suffix += f" tool_calls={len(tool_calls)}"
        preview = str(content)[:300] + ("..." if len(str(content)) > 300 else "")
        print(f"  [{msg_type}{suffix}] {preview}")

    ai_messages = [
        m for m in messages if getattr(m, "type", None) == "ai" and getattr(m, "content", None)
    ]
    if not ai_messages:
        print("\nAgent 层: FAIL（无 AI 文本回复）")
        return False

    final = str(ai_messages[-1].content)
    print("\n=== 最终 AI 回复 ===")
    print(final[:1500])
    ok = bool(final.strip())
    print(f"\nAgent 层: {'PASS' if ok else 'FAIL'}")
    return ok


async def main() -> int:
    parser = argparse.ArgumentParser(description="FlightAgent / search_flights 测试")
    parser.add_argument(
        "--case-id",
        default="flight-beijing-sanya-jun25",
        help="tests/fixtures/travel_single_agent_cases.json 中的 case_id",
    )
    parser.add_argument("--query", help="覆盖 fixture 中的 user_query")
    parser.add_argument("--tool-only", action="store_true", help="只测工具，不调用 LLM Agent")
    args = parser.parse_args()

    load_project_dotenv()
    case = next(
        item for item in load_single_agent_cases().cases if item.case_id == args.case_id
    )
    user_query = args.query or case.user_query
    tool_args = case.tool_args

    print("=== 环境变量 ===")
    for key in (
        "DASHSCOPE_API_KEY",
        "OPENAI_API_KEY",
        "VARIFLIGHT_API_KEY",
        "X_VARIFLIGHT_KEY",
        "AVIATIONSTACK_KEY",
    ):
        print(f"{key}: {_mask_key(key)}")

    tool_ok = await test_search_flights_tool(
        departure=str(tool_args["departure"]),
        arrival=str(tool_args["arrival"]),
        date=str(tool_args["date"]),
    )

    if args.tool_only:
        return 0 if tool_ok else 1

    if not os.getenv("DASHSCOPE_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\n跳过 Agent 测试：未配置 DASHSCOPE_API_KEY / OPENAI_API_KEY")
        return 0 if tool_ok else 1

    agent_ok = await test_flight_agent(user_query, thread_id=f"flight_test_{case.case_id}")
    return 0 if tool_ok and agent_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
