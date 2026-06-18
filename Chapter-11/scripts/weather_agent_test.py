"""手动测试 WeatherAgent 与 get_weather / get_weather_forecast 工具。"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

# 允许在 scripts/ 下直接运行：python weather_agent_test.py
_CHAPTER8_ROOT = Path(__file__).resolve().parent.parent
if str(_CHAPTER8_ROOT) not in sys.path:
    sys.path.insert(0, str(_CHAPTER8_ROOT))

from agent_framework.config import load_project_dotenv
from domains.travel.plan_context import build_time_anchor

def _safe_print(text: str) -> None:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    safe = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(safe)


def _has_error(result: Any) -> bool:
    return isinstance(result, dict) and bool(result.get("error"))


def _forecast_dates(result: Dict[str, Any]) -> List[str]:
    items = result.get("forecasts") or []
    return [str(x.get("date", "")) for x in items if isinstance(x, dict)]


async def _run_tool(label: str, coro) -> Tuple[str, bool, Dict[str, Any]]:
    result = await coro
    ok = not _has_error(result)
    print(f"\n--- {label} → {'PASS' if ok else 'FAIL'} ---")
    print(json.dumps(result, ensure_ascii=False, indent=2)[:1200])
    if isinstance(result, dict) and result.get("forecasts"):
        dates = _forecast_dates(result)
        print(f"  预报日期范围: {dates[0]} ~ {dates[-1]}（共 {len(dates)} 天）")
    return label, ok, result if isinstance(result, dict) else {"raw": result}


async def _get_weather_forecast_direct(city: str, days: int = 7) -> Dict[str, Any]:
    """与 weather.get_weather_forecast 相同逻辑，不 import LangChain。"""
    from datetime import datetime, timedelta

    from domains.travel.infra.travel_api import require_non_empty
    from domains.travel.infra.weather_mcp import (
        fetch_weather_forecast_via_mcp,
        fetch_weather_via_mcp,
    )

    ok, err = require_non_empty(city, "city")
    if not ok:
        return {"error": err}

    n_days = max(1, min(int(days or 7), 14))
    mcp_result = await fetch_weather_forecast_via_mcp(city, n_days)
    if mcp_result and mcp_result.get("forecasts"):
        return mcp_result

    forecasts = []
    today = datetime.now().date()
    for i in range(n_days):
        d = (today + timedelta(days=i)).strftime("%Y-%m-%d")
        single = await _get_weather_direct(city, d)
        if single and not single.get("error"):
            fc = single.get("forecast") or {}
            forecasts.append({
                "date": d,
                "condition": fc.get("condition") or single.get("text", "未知"),
                "temp_high_c": fc.get("temp_high_c"),
                "temp_low_c": fc.get("temp_low_c"),
            })
    if forecasts:
        return {
            "city": city,
            "days": len(forecasts),
            "forecasts": forecasts,
            "data_source": "fallback/daily_chain",
        }
    return {"error": "无法获取多日天气预报"}


async def _get_weather_direct(city: str, date_str: str) -> Dict[str, Any]:
    """与 weather.get_weather 相同逻辑，不 import LangChain。"""
    from domains.travel.infra.travel_api import (
        amap_weather_by_city_and_date,
        require_non_empty,
        resolve_relative_date,
        wttr_weather_by_city_and_date,
    )
    from domains.travel.infra.weather_mcp import fetch_weather_via_mcp

    ok, err = require_non_empty(city, "city")
    if not ok:
        return {"error": err}

    norm_date, derr = resolve_relative_date(date_str)
    if derr:
        return {"error": derr}

    mcp_result = await fetch_weather_via_mcp(city, norm_date)
    if mcp_result and not mcp_result.get("error"):
        return {k: v for k, v in mcp_result.items() if k != "raw"}

    try:
        result = await amap_weather_by_city_and_date(city, norm_date)
        if not result.get("error") and result.get("forecast"):
            return {
                "city": city,
                "date": norm_date,
                "forecast": result["forecast"],
                "data_source": "amap_weather",
            }
    except Exception:
        pass

    try:
        result = await wttr_weather_by_city_and_date(city, norm_date)
        if not result.get("error"):
            return {
                "city": city,
                "date": norm_date,
                "text": result.get("text"),
                "forecast": result.get("forecast"),
                "data_source": "wttr.in",
            }
    except Exception as e:
        return {"error": f"weather_query_failed: {str(e)}"}

    return {"error": "无法获取天气信息"}


async def test_trace_failure_replay(*, ref: date | None = None) -> bool:
    """复现 spans_20260615_155826.jsonl 中 T1/WeatherAgent 的 tool 调用链。"""
    anchor = build_time_anchor(ref=ref)
    ref_date = ref or date.today()
    next_start = date.fromisoformat(anchor["next_week_start"])
    next_end = date.fromisoformat(anchor["next_week_end"])

    print("\n" + "=" * 72)
    print("=== 3. Trace 失败案例复现（T1 / WeatherAgent）===")
    print(f"参考日期: {ref_date.isoformat()}（trace 原文件为 2026-06-15）")
    print(f"下周区间: {anchor['next_week_label']}")
    print("任务: 上海 / 苏州 / 杭州 下周每日天气")
    print("=" * 72)

    results: List[Tuple[str, bool, Dict[str, Any]]] = []

    # 与 trace 一致：先 forecast days=7（三城）
    for city in ("上海", "苏州", "杭州市"):
        results.append(
            await _run_tool(
                f"get_weather_forecast({city!r}, days=7)",
                _get_weather_forecast_direct(city, 7),
            )
        )

    # trace 中上海 days=14 失败
    results.append(
        await _run_tool(
            "get_weather_forecast('上海', days=14)",
            _get_weather_forecast_direct("上海", 14),
        )
    )

    # 苏州 / 杭州 days=14 在 trace 中成功
    for city in ("苏州", "杭州市"):
        results.append(
            await _run_tool(
                f"get_weather_forecast({city!r}, days=14)",
                _get_weather_forecast_direct(city, 14),
            )
        )

    # trace 中 Agent 对上海下周 7 天逐日 get_weather，全部失败
    shanghai_next_week: List[str] = []
    d = next_start
    while d <= next_end:
        shanghai_next_week.append(d.isoformat())
        d += timedelta(days=1)

    print(f"\n--- 上海下周逐日 get_weather（{len(shanghai_next_week)} 次，trace 中均 FAIL）---")
    daily_failures = 0
    for day in shanghai_next_week:
        label, ok, payload = await _run_tool(
            f"get_weather('上海', date={day})",
            _get_weather_direct("上海", day),
        )
        results.append((label, ok, payload))
        if not ok:
            daily_failures += 1

    # 汇总：下周是否被 forecast 覆盖
    print("\n" + "=" * 72)
    print("=== 汇总 ===")
    for label, ok, _ in results:
        print(f"  {'PASS' if ok else 'FAIL':4}  {label}")

    shanghai_14 = next(
        (p for lbl, _, p in results if "days=14" in lbl and "上海" in lbl),
        {},
    )
    suzhou_14 = next(
        (p for lbl, _, p in results if "days=14" in lbl and "苏州" in lbl),
        {},
    )
    hangzhou_14 = next(
        (p for lbl, _, p in results if "days=14" in lbl and "杭州市" in lbl),
        {},
    )

    def _covers_next_week(payload: Dict[str, Any]) -> bool:
        dates = set(_forecast_dates(payload))
        needed = {d.isoformat() for d in (
            next_start + timedelta(days=i) for i in range((next_end - next_start).days + 1)
        )}
        return needed.issubset(dates)

    print(f"\n  上海 forecast(14) 覆盖下周: {_covers_next_week(shanghai_14)}")
    print(f"  苏州 forecast(14) 覆盖下周: {_covers_next_week(suzhou_14)}")
    print(f"  杭州 forecast(14) 覆盖下周: {_covers_next_week(hangzhou_14)}")
    print(f"  上海逐日 get_weather 失败次数: {daily_failures}/{len(shanghai_next_week)}")

    any_ok = any(ok for _, ok, _ in results)
    print(
        "\n  编排层判定（与 nodes._evaluate_subtask_status 一致）:"
        f" 任一 tool 成功 → {'completed' if any_ok else 'failed'}"
    )

    # 复现成功：至少说明与 trace 同类调用可观测；不要求全部 PASS
    return True


def _mask_key(name: str) -> str:
    value = os.getenv(name, "")
    if not value:
        return "未设置"
    return f"已设置 ({len(value)} chars)"


async def test_get_weather_tool() -> bool:
    from domains.travel.agents.weather import get_weather

    print("\n=== 1. get_weather 工具（上海，明天）===")
    result = await get_weather.ainvoke({"city": "上海", "date": "明天"})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if isinstance(result, dict) and result.get("error"):
        return False
    return bool(result)


async def test_weather_agent() -> bool:
    from domains.travel.agents.weather import create_weather_agent

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
    parser = argparse.ArgumentParser(description="WeatherAgent / 天气工具测试")
    parser.add_argument(
        "--trace-case",
        action="store_true",
        help="复现 spans_20260615 trace 中 T1 天气失败调用链（仅 tool 层，无需 LLM）",
    )
    parser.add_argument(
        "--ref-date",
        default="2026-06-15",
        help="trace 复现参考日期（默认 2026-06-15，与 trace 文件一致）",
    )
    parser.add_argument(
        "--skip-basic",
        action="store_true",
        help="跳过基础 get_weather / Agent 测试",
    )
    args = parser.parse_args()

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

    if args.trace_case:
        ref = date.fromisoformat(args.ref_date)
        await test_trace_failure_replay(ref=ref)
        return 0

    tool_ok = True
    agent_ok = True
    if not args.skip_basic:
        tool_ok = await test_get_weather_tool()
        print(f"\n工具层结果: {'PASS' if tool_ok else 'FAIL'}")

        if not os.getenv("DASHSCOPE_API_KEY") and not os.getenv("OPENAI_API_KEY"):
            print("跳过 Agent 测试：未配置 DASHSCOPE_API_KEY / OPENAI_API_KEY")
        else:
            agent_ok = await test_weather_agent()
            print(f"\nAgent 层结果: {'PASS' if agent_ok else 'FAIL'}")

    ref = date.fromisoformat(args.ref_date)
    await test_trace_failure_replay(ref=ref)
    if args.skip_basic:
        return 0
    if not os.getenv("DASHSCOPE_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        return 0 if tool_ok else 1
    return 0 if tool_ok and agent_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
