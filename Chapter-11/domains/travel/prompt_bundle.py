"""旅行领域 prompt 包（locales/{zh,en}.json）。"""

from __future__ import annotations

from dataclasses import fields

from agent_framework.domain.domain_prompts import DomainPrompts
from agent_framework.domain.locale_loader import domain_prompts_from_locale


class TravelPrompts(DomainPrompts):
    @staticmethod
    def build(locale: str = "zh") -> TravelPrompts:
        base = domain_prompts_from_locale("travel", locale)
        return TravelPrompts(**{f.name: getattr(base, f.name) for f in fields(base)})
