"""旅行领域 prompt 包（locales/{zh,en}.json）。"""

from __future__ import annotations

from dataclasses import fields

from agent_framework.domain.domain_prompts import DomainPrompts
from agent_framework.domain.locale_loader import domain_prompts_from_locale
from agent_framework.optimization.prompt_store import apply_prompt_overrides


class TravelPrompts(DomainPrompts):
    @staticmethod
    def build(locale: str = "zh", *, use_optimized: bool = True) -> TravelPrompts:
        base = domain_prompts_from_locale("travel", locale)
        prompts = TravelPrompts(**{field.name: getattr(base, field.name) for field in fields(base)})
        if use_optimized:
            prompts = apply_prompt_overrides(prompts, locale=locale)
            return TravelPrompts(**{field.name: getattr(prompts, field.name) for field in fields(prompts)})
        return prompts
