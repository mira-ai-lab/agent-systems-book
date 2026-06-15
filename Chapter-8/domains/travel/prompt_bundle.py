"""旅行领域 prompt 包（薄封装：prompt 原文在 domains.travel.prompts）。"""

from __future__ import annotations

from agent_framework.domain.domain_prompts import DomainPrompts
from domains.travel.prompts import (
    AGENT_ROUTING_PROMPT,
    AGGREGATION_PROMPT,
    CENTRAL_AGENT_SYSTEM_PROMPT,
    DEPENDENCY_SYSTEM_PROMPT_ZH,
    DEPENDENCY_USER_PROMPT_ZH,
    FACTS_PROMPT,
    PROMPT_TP_ZH,
)


class TravelPrompts(DomainPrompts):
    @staticmethod
    def build() -> TravelPrompts:
        return TravelPrompts(
            central_agent_system=CENTRAL_AGENT_SYSTEM_PROMPT,
            aggregation=AGGREGATION_PROMPT,
            facts_prompt=FACTS_PROMPT,
            decomposition_prompt=PROMPT_TP_ZH,
            dependency_system=DEPENDENCY_SYSTEM_PROMPT_ZH,
            dependency_user=DEPENDENCY_USER_PROMPT_ZH,
            agent_routing=AGENT_ROUTING_PROMPT,
            multi_task_title="📋 最终旅行规划",
            single_task_title="📋 最终回复",
            aggregation_skip_hint="单任务查询，直接使用子智能体回复（跳过旅行规划聚合）",
            memory_aggregation_instruction=(
                "请根据用户原始请求的范围，综合子任务执行结果生成回复。"
                "严格匹配用户问题，不要添加用户未询问的内容（例如用户只问天气，不要输出行程/酒店/美食攻略）。"
                "仅当用户明确要求旅行规划时，才提供完整的多日行程方案。"
            ),
        )
