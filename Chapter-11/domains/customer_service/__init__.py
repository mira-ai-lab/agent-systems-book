"""客服领域插件包。"""

from domains.customer_service.prompt_bundle import CustomerServicePrompts
from domains.customer_service.registry import (
    create_customer_service_registry,
    customer_service_domain_config,
)

__all__ = [
    "CustomerServicePrompts",
    "create_customer_service_registry",
    "customer_service_domain_config",
]
