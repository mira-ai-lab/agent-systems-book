"""子智能体 system prompt 公共片段（多实体拆解、时间锚点）。"""

from __future__ import annotations

from domains.travel.plan_context import build_time_anchor, format_time_anchor_block

# 通用：工具为单实体设计时，Agent 自行拆解任务
MULTI_ENTITY_TOOL_RULES = """
【多实体任务（必须遵守）】
1. 工具参数 city / location 每次只能填**一个**地点；禁止把多个城市拼成一个字符串传入。
2. 若任务描述涉及多个地点（如「上海、苏州、杭州」或「三地」），须**依次对每个地点各调用一次工具**，全部查完后再汇总回复。
3. 禁止只查询第一个地点就结束；禁止在未调用工具前声称「将继续查询」并结束。
4. 从任务描述自行识别地点列表；若描述模糊，按已明确列出的城市/区域逐个处理。
"""

AGENT_TIME_ANCHOR_RULES = """
【日期与时间】
- 用户说「下周」「未来 N 天」时，必须使用下方系统时间锚点，禁止臆造 2024 等历史年份。
{time_anchor}
"""


def agent_time_anchor_block() -> str:
    return AGENT_TIME_ANCHOR_RULES.format(
        time_anchor=format_time_anchor_block(build_time_anchor()),
    )
