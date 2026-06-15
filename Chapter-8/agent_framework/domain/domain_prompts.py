"""领域 prompt 集合（基类）；旅行实现见 domains.travel.TravelPrompts。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DomainPrompts:
    central_agent_system: str
    aggregation: str
    facts_prompt: str
    decomposition_prompt: str
    dependency_system: str
    dependency_user: str
    agent_routing: str
    multi_task_title: str = "📋 最终规划"
    single_task_title: str = "📋 最终回复"
    aggregation_skip_hint: str = "单任务查询，直接使用子智能体回复（跳过聚合 LLM）"
    memory_aggregation_instruction: str = (
        "请根据用户原始请求的范围，综合子任务执行结果生成回复。"
        "严格匹配用户问题，不要添加用户未询问的内容。"
    )
