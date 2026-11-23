"""Demo 领域 prompt 包。"""

from __future__ import annotations

from dataclasses import fields

from agent_framework.domain.domain_prompts import DomainPrompts
from agent_framework.domain.locale_loader import domain_prompts_from_locale


class DemoPrompts(DomainPrompts):
    @staticmethod
    def build(locale: str = "zh") -> DemoPrompts:
        base = domain_prompts_from_locale("demo", locale)
        return DemoPrompts(**{f.name: getattr(base, f.name) for f in fields(base)})
