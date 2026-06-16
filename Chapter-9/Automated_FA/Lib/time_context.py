"""为失败归因评估 prompt 注入参考日期（今天）。"""

from __future__ import annotations

import os
from datetime import date


def get_reference_date(reference_date: str | None = None) -> str:
    """返回 YYYY-MM-DD 参考日期。

    优先级：显式参数 > 环境变量 FA_REFERENCE_DATE > 运行时的今天。
    """
    if reference_date:
        return reference_date
    env_date = os.getenv("FA_REFERENCE_DATE", "").strip()
    if env_date:
        return env_date
    return date.today().isoformat()


def format_time_context_for_eval(reference_date: str | None = None) -> str:
    """生成写入评估 prompt 的日期上下文块。"""
    today = get_reference_date(reference_date)
    return (
        "【评估专用·当前日期】\n"
        f"- 今天：{today}\n"
        "- 判断「下周」「未来2周」等相对时间时，必须以今天为基准推算；"
        "禁止假设当前为 2024/2025 等与上文不符的年份。\n"
        "- 若对话中的绝对日期落在今天起的合理窗口内，通常不算 Planner 阶段错误；"
        "重点检查各 Agent 的工具调用与返回是否满足用户意图。"
    )
