"""规划上下文：供 TaskPlanner prompt 注入的系统时间与日期锚点。"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, Optional


def build_time_anchor(*, ref: Optional[date] = None) -> Dict[str, str]:
    """生成规划用的日期锚点（今天、下周区间）。"""
    today = ref or date.today()
    weekday = today.weekday()  # 0=Mon
    days_until_next_mon = (7 - weekday) % 7
    if days_until_next_mon == 0:
        days_until_next_mon = 7
    next_mon = today + timedelta(days=days_until_next_mon)
    next_sun = next_mon + timedelta(days=6)

    return {
        "today": today.isoformat(),
        "today_weekday": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][weekday],
        "next_week_start": next_mon.isoformat(),
        "next_week_end": next_sun.isoformat(),
        "next_week_label": f"{next_mon.isoformat()}（周一）～{next_sun.isoformat()}（周日）",
    }


def format_time_anchor_block(anchor: Optional[Dict[str, str]] = None) -> str:
    """供 LLM prompt 注入的时间锚点文本。"""
    a = anchor or build_time_anchor()
    return (
        "【系统时间锚点（必须使用，禁止臆造其他年份/日期）】\n"
        f"- 今天：{a['today']}（{a['today_weekday']}）\n"
        f"- 用户说「下周」指：{a['next_week_label']}\n"
        "- params 中的 date / start_date 必须落在今天或之后，且在未来 14 天预报范围内\n"
        "- 禁止输出 2024、2025 等早于今天的历史日期作为「下周」"
    )


def build_agent_routing_format_kwargs(
    *,
    agent_team: str,
    subtasks_json: str,
    ref: Optional[date] = None,
) -> Dict[str, str]:
    """组装 agent_routing 模板 format 所需占位符（运行时注入 today / time_anchor）。"""
    anchor = build_time_anchor(ref=ref)
    return {
        "agent_team": agent_team,
        "subtasks_json": subtasks_json,
        "today": anchor["today"],
        "time_anchor": format_time_anchor_block(anchor),
    }
