"""Travel 子 Agent prompt 加载（locales JSON + 动态片段 + optimized override）。"""

from __future__ import annotations

from datetime import datetime

from agent_framework.domain.locale_loader import agent_fragment, agent_system_prompt
from agent_framework.i18n.agent_locale_context import get_agent_locale
from agent_framework.optimization.agent_prompt_store import load_optimized_agent_prompt_template
from domains.travel.agents.prompt_fragments import agent_time_anchor_block
from domains.travel.plan_context import build_time_anchor, format_time_anchor_block


def travel_agent_prompt(agent_name: str, *, locale: str | None = None) -> str:
    loc = locale or get_agent_locale()
    # 优先使用 optimization 写回的 Agent prompt 模板
    template = load_optimized_agent_prompt_template(agent_name, locale=loc)
    if not template:
        template = agent_system_prompt("travel", agent_name, loc)
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        time_rules = agent_fragment("travel", "time_anchor_rules", loc)
    except KeyError:
        time_rules = ""
    try:
        multi_rules = agent_fragment("travel", "multi_entity_rules", loc)
    except KeyError:
        multi_rules = ""
    time_anchor = format_time_anchor_block(build_time_anchor())
    return template.format(
        today=today,
        time_anchor=time_anchor,
        time_anchor_rules=time_rules.format(time_anchor=time_anchor) if "{time_anchor}" in time_rules else time_rules,
        multi_entity_rules=multi_rules,
    )
