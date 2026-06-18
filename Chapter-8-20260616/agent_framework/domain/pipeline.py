"""LangGraph 流水线节点开关。"""

from __future__ import annotations

from dataclasses import dataclass

PRE_SURVEY_MODE_ROUTER_PREFILL = "router_prefill"
PRE_SURVEY_MODE_FULL_CH2 = "full_ch2"
PRE_SURVEY_MODE_OFF = "off"


def normalize_pre_survey_mode(value: str) -> str:
    mode = (value or PRE_SURVEY_MODE_ROUTER_PREFILL).strip().lower()
    if mode in (
        PRE_SURVEY_MODE_ROUTER_PREFILL,
        PRE_SURVEY_MODE_FULL_CH2,
        PRE_SURVEY_MODE_OFF,
    ):
        return mode
    return PRE_SURVEY_MODE_ROUTER_PREFILL


@dataclass(frozen=True)
class PipelineConfig:
    enable_pre_survey: bool = True
    pre_survey_mode: str = PRE_SURVEY_MODE_ROUTER_PREFILL
    enable_memory: bool = True
    enable_step_summary: bool = False
    step_summary_min_chars: int = 200
    enable_stage_summary: bool = False
    stage_summary_min_steps: int = 2
    enable_thread_stage_context: bool = True
    allow_task_planner_decomposition: bool = True

    @property
    def needs_save_memory(self) -> bool:
        return self.enable_memory

    @property
    def runs_pre_survey_node(self) -> bool:
        """是否在 Fixed Graph 中运行 pre_survey 节点。"""
        return self.enable_pre_survey and normalize_pre_survey_mode(
            self.pre_survey_mode
        ) != PRE_SURVEY_MODE_OFF

    @property
    def resolved_pre_survey_mode(self) -> str:
        return normalize_pre_survey_mode(self.pre_survey_mode)
