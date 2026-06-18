"""客服领域子 Agent 元数据（测试与 Planner prompt 使用）。"""

from __future__ import annotations

from typing import Any, Dict

CUSTOMER_SERVICE_AGENT_SPECS: Dict[str, Dict[str, Any]] = {
    "FAQAgent": {
        "description": "回答常见问题：退换货政策、物流时效、会员权益、支付方式",
        "requires_tool": True,
        "skills": [
            {
                "name": "知识支持-退换货政策",
                "description": "退换货与物流政策知识库",
                "tags": ["政策-退货", "政策-换货", "物流-时效"],
                "keywords": ["退货", "换货", "物流"],
                "inputSchema": ["topic"],
                "outputSchema": ["answer", "policy_ref"],
            },
            {
                "name": "lookup_faq",
                "inputSchema": ["topic"],
                "outputSchema": ["answer", "policy_ref"],
            },
        ],
    },
    "TicketAgent": {
        "description": "创建工单、查询工单进度、升级投诉与人工转接",
        "requires_tool": True,
        "skills": [
            {
                "name": "create_ticket",
                "inputSchema": ["issue_type", "description"],
                "outputSchema": ["ticket_id", "status"],
            },
        ],
    },
}
