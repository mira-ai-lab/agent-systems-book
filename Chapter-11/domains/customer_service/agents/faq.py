"""客服子 Agent。"""

from __future__ import annotations

from langchain_core.tools import tool

from agent_framework.domain.locale_loader import agent_system_prompt
from agent_framework.i18n.agent_locale_context import get_agent_locale
from agent_framework.infra.agent_runtime import build_agent

_FAQ_KB = {
    "退换货": "签收后 7 天内可无理由退货；拆封不影响二次销售的商品可退。",
    "物流": "标准快递 2–5 个工作日；加急订单 24 小时内出库。",
    "会员": "银卡满 500 元、金卡满 2000 元；会员享专属客服与生日礼券。",
    "支付": "支持微信、支付宝、银联；大额订单可走对公转账。",
}


@tool
def lookup_faq(topic: str) -> str:
    """按主题检索 FAQ 知识库。"""
    key = (topic or "").strip()
    for k, v in _FAQ_KB.items():
        if k in key or key in k:
            return v
    return "未找到匹配条目，建议转 TicketAgent 创建工单。"


def create_faq_agent():
    locale = get_agent_locale()
    return build_agent(
        tools=[lookup_faq],
        system_prompt=agent_system_prompt("customer_service", "FAQAgent", locale),
    )


@tool
def create_ticket(issue_type: str, description: str) -> str:
    """创建客服工单并返回工单号。"""
    import uuid

    ticket_id = f"CS-{uuid.uuid4().hex[:8].upper()}"
    return (
        f"工单已创建：{ticket_id}，类型={issue_type}，"
        f"摘要={description[:80]}，预计 2 小时内首次回复。"
    )


def create_ticket_agent():
    locale = get_agent_locale()
    return build_agent(
        tools=[create_ticket],
        system_prompt=agent_system_prompt("customer_service", "TicketAgent", locale),
    )
