"""LangGraph 流水线节点开关。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    enable_pre_survey: bool = True
    enable_memory: bool = True

    @property
    def needs_save_memory(self) -> bool:
        return self.enable_memory
