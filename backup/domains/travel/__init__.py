"""旅行领域插件：agents / prompts / specs / infra 唯一实现源。"""

from domains.travel.prompt_bundle import TravelPrompts
from domains.travel.registry import create_travel_registry, travel_domain_config

__all__ = ["TravelPrompts", "create_travel_registry", "travel_domain_config"]
