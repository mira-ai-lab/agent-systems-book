"""Deprecated shim — 英文 domain 话术请使用 ``domains/travel/locales/en.json``。"""

from __future__ import annotations

import warnings

warnings.warn(
    "domains.travel.prompts_en is deprecated; use domains/travel/locales/en.json",
    DeprecationWarning,
    stacklevel=2,
)

from agent_framework.domain.locale_loader import load_domain_locale_payload

_DATA = load_domain_locale_payload("travel", "en")

CENTRAL_AGENT_SYSTEM_PROMPT = _DATA["central_agent_system"]
AGGREGATION_PROMPT = _DATA["aggregation"]
FACTS_PROMPT = _DATA["facts_prompt"]
PROMPT_TP_EN = _DATA["decomposition_prompt"]
DEPENDENCY_SYSTEM_PROMPT_EN = _DATA["dependency_system"]
DEPENDENCY_USER_PROMPT_EN = _DATA["dependency_user"]
AGENT_ROUTING_PROMPT = _DATA["agent_routing"]
SUPERVISOR_SYSTEM_PROMPT = _DATA["supervisor_system"]
