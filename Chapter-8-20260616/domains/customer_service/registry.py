"""客服领域子 Agent 注册表与 DomainConfig。"""

from __future__ import annotations

from agent_framework.domain.agent_registry import SubAgentRegistry
from agent_framework.domain.domain_config import DomainConfig
from domains.customer_service.specs import CUSTOMER_SERVICE_AGENT_SPECS


def create_customer_service_registry() -> SubAgentRegistry:
    from domains.customer_service.agents.faq import create_faq_agent, create_ticket_agent

    registry = SubAgentRegistry()
    creators = {
        "FAQAgent": create_faq_agent,
        "TicketAgent": create_ticket_agent,
    }
    for name, spec in CUSTOMER_SERVICE_AGENT_SPECS.items():
        creator = creators.get(name)
        if not creator:
            continue
        registry.register(
            name,
            creator,
            description=spec["description"],
            requires_tool=spec.get("requires_tool", False),
            skills=spec.get("skills"),
        )
    return registry


def customer_service_domain_config(*, enable_guess_agent: bool = False) -> DomainConfig:
    def _guess(description: str, reg: SubAgentRegistry) -> str | None:
        text = description.lower()
        if any(k in text for k in ("投诉", "工单", "人工", "ticket")):
            return "TicketAgent"
        if any(k in text for k in ("退货", "物流", "会员", "支付", "faq", "政策")):
            return "FAQAgent"
        return None

    return DomainConfig(
        guess_fn=_guess if enable_guess_agent else None,
        routing_fallback="FAQAgent",
        enable_guess_agent=enable_guess_agent,
    )
