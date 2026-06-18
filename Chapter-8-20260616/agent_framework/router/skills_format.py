"""Agent skills 格式化（含 knowledge_keyword 知识点展开）。"""

from __future__ import annotations

from typing import Any, List

from agent_framework.router.prompts.loader import get_selection_locale_strings


def format_skill_tags(tags: List[Any], *, locale: str = "zh") -> str:
    strings = get_selection_locale_strings(locale)
    separator = strings.get("separator", "、")
    points_label = strings.get("knowledge_points", "")
    grouped: dict[str, list[str]] = {}
    for tag in tags:
        text = str(tag or "").strip()
        if not text:
            continue
        parts = text.split("-", 1)
        if len(parts) == 2:
            grouped.setdefault(parts[0], []).append(parts[1])
    if not grouped:
        return ""
    lines: List[str] = []
    keys = list(grouped.keys())
    for idx, key in enumerate(keys):
        values = separator.join(grouped[key])
        suffix = ";" if idx == len(keys) - 1 else "; \n  -"
        lines.append(f"{key}: {values}{suffix}")
    tag_str = "".join(lines)
    return f"\n{points_label} \n  -{tag_str}" if tag_str else ""


def format_agent_skills(info: dict[str, Any], *, locale: str = "zh") -> str:
    skills = info.get("skills") or []
    if not skills:
        return get_selection_locale_strings(locale).get("skill_none", "无")

    strings = get_selection_locale_strings(locale)
    knowledge_keyword = strings.get("knowledge_keyword", "知识支持")
    parts: List[str] = []
    for idx, skill in enumerate(skills, start=1):
        if not isinstance(skill, dict):
            parts.append(f"{idx}.{skill}")
            continue
        name = str(skill.get("name") or f"skill_{idx}")
        description = str(skill.get("description") or "")
        if knowledge_keyword in name:
            description += format_skill_tags(skill.get("tags") or [], locale=locale)
        keywords = skill.get("keywords") or []
        if keywords:
            description += f" 关键词:{','.join(str(k) for k in keywords)}"
        parts.append(f"{idx}.{name}:{description or name}")
    return "\n".join(parts)
