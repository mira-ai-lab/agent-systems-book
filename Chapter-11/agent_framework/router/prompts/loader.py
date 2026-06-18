"""平台 Router prompt locale 加载。"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


_LOCALES_DIR = Path(__file__).resolve().parent / "locales"


@lru_cache(maxsize=8)
def load_locale(locale: str) -> dict[str, Any]:
    loc = (locale or "zh").strip() or "zh"
    path = _LOCALES_DIR / f"{loc}.json"
    if not path.is_file():
        if loc != "zh":
            return load_locale("zh")
        raise FileNotFoundError(f"Router locale 不存在: {locale} ({path})")
    return json.loads(path.read_text(encoding="utf-8"))


def get_classification_prompts(locale: str) -> dict[str, str]:
    data = load_locale(locale)
    block = data.get("selection", {}).get("classification", {})
    return {
        "prompt_base": str(block.get("prompt_base", "")),
        "agent_template": str(block.get("agent_template", "Name:{name}\nDescription:{description}\n")),
        "note": str(block.get("note", "")),
    }


def get_history_gate_prompts(locale: str) -> dict[str, str]:
    data = load_locale(locale)
    block = data.get("flow_builder", {})
    return {
        "history_prompt": str(block.get("history_prompt", "")),
    }


def get_interaction_rewrite_prompts(locale: str) -> dict[str, str]:
    data = load_locale(locale)
    block = data.get("response_handler", {}).get("handle_interaction", {}).get("prompt_rewrite", {})
    return {
        "system": str(block.get("system", "")),
        "user_template": str(block.get("user_template", "")),
    }


def get_instruction_build_prompts(locale: str) -> dict[str, str]:
    data = load_locale(locale)
    block = data.get("response_handler", {}).get("build_instruction", {}).get("context_rebuild", {})
    return {
        "system": str(block.get("system", "")),
        "user_template": str(block.get("user_template", "")),
    }


def get_step_summary_prompts(locale: str) -> dict[str, str]:
    data = load_locale(locale)
    block = data.get("task", {}).get("summary", {})
    return {
        "round_info": str(block.get("round_info", "")),
        "prompt": str(block.get("prompt", "")),
    }


def get_stage_summary_prompts(locale: str) -> dict[str, str]:
    data = load_locale(locale)
    block = data.get("task", {}).get("stage", {})
    return {
        "no_previous_summary": str(block.get("no_previous_summary", "")),
        "all_steps_completed": str(block.get("all_steps_completed", "")),
        "step_level_context_desc": str(block.get("step_level_context_desc", "")),
        "step_level_context_detail": str(block.get("step_level_context_detail", "")),
        "prompt": str(block.get("prompt", "")),
    }


def get_extraction_prompts(locale: str) -> dict[str, str]:
    data = load_locale(locale)
    block = data.get("selection", {}).get("extraction", {})
    return {
        "single": str(block.get("single", "")),
        "multi": str(block.get("multi", "")),
    }


def get_selection_locale_strings(locale: str) -> dict[str, str]:
    data = load_locale(locale)
    block = data.get("selection", {})
    return {
        "knowledge_keyword": str(block.get("knowledge_keyword", "知识支持")),
        "knowledge_points": str(block.get("knowledge_points", "包含的知识点有")),
        "separator": str(block.get("separator", "、")),
        "skill_none": str(block.get("skill_none", "无")),
    }


def get_domain_classification_prompts(locale: str) -> dict[str, str]:
    data = load_locale(locale)
    block = data.get("platform", {}).get("domain_classification", {})
    return {
        "prompt_base": str(block.get("prompt_base", "")),
        "domain_template": str(
            block.get("domain_template", "领域:{name}\n名称:{display_name}\n能力:{agents}\n")
        ),
        "note": str(block.get("note", "")),
    }


def get_task_decomposition_prompts(locale: str) -> dict[str, str]:
    data = load_locale(locale)
    block = data.get("flow_builder", {})
    return {
        "prompt": str(block.get("task_decomposition_prompt", "")),
        "keyword_goal": str(block.get("keyword_goal", "整体目标：")),
        "keyword_subtasks": str(block.get("keyword_subtasks", "子任务：")),
    }
