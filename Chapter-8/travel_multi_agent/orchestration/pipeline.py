"""Pipeline toggles for the LangGraph orchestration.

:author: zhanghong, liuhui, xingrui
:date: 2026-11-22"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    enable_pre_survey: bool = True
    enable_memory: bool = True

    @property
    def needs_save_memory(self) -> bool:
        return self.enable_memory
