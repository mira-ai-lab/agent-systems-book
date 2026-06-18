"""领域 prompt 集合（基类）；旅行实现见 domains.travel.TravelPrompts。"""

from __future__ import annotations

from dataclasses import dataclass, fields

from agent_framework.prompts.platform_defaults import get_platform_domain_prompts


@dataclass(frozen=True)
class DomainPrompts:
    central_agent_system: str
    aggregation: str
    facts_prompt: str
    decomposition_prompt: str
    dependency_system: str
    dependency_user: str
    agent_routing: str
    supervisor_system: str = ""
    multi_task_title: str = "📋 最终规划"
    single_task_title: str = "📋 最终回复"
    aggregation_skip_hint: str = "单任务查询，直接使用子智能体回复（跳过聚合 LLM）"
    memory_aggregation_instruction: str = (
        "请根据用户原始请求的范围，综合子任务执行结果生成回复。"
        "严格匹配用户问题，不要添加用户未询问的内容。"
    )

    def with_platform_defaults(self, locale: str = "zh") -> "DomainPrompts":
        """空字段回退到平台默认话术（`agent_framework/prompts/locales/`）。"""
        defaults = get_platform_domain_prompts(locale)
        merged: dict[str, str] = {}
        for field in fields(self):
            current = str(getattr(self, field.name) or "")
            fallback = defaults.get(field.name, "")
            merged[field.name] = current if current.strip() else fallback
        return DomainPrompts(**merged)
