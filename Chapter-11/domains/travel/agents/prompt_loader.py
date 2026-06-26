"""Travel 子 Agent prompt 加载（locales JSON + 动态片段 + optimized override）。"""

from __future__ import annotations

from agent_framework.domain.locale_loader import agent_system_prompt
from agent_framework.i18n.agent_locale_context import get_agent_locale
from agent_framework.optimization.agent_prompt_store import load_optimized_agent_prompt_template
from domains.travel.agents.prompt_fragments import render_travel_agent_prompt_template


def travel_agent_prompt(agent_name: str, *, locale: str | None = None) -> str:
    loc = locale or get_agent_locale()
    # 优先使用 optimization 写回的 Agent prompt 模板
    template = load_optimized_agent_prompt_template(agent_name, locale=loc)
    if not template:
        template = agent_system_prompt("travel", agent_name, loc)
    return render_travel_agent_prompt_template(template, locale=loc)
