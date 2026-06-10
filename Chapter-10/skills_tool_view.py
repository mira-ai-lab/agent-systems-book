"""
skills_tool_view.py — skill_view(name, tier) Hermes 渐进式加载工具

对应 Hermes 生产环境 skill_view：
  - LLM 扫描 System Prompt 中的 <available_skills>
  - 相关则调用 skill_view(name, tier=2) 加载完整 SKILL.md
  - 返回文本进入 ToolMessage，不修改 System Prompt（保护 Prompt Cache）

Tier 说明：
  0 — 分类索引（devops: skill_a, skill_b）
  1 — 元数据 JSON
  2 — 完整正文（Trigger / Steps / Pitfalls / Verification）← 执行前默认
  3 — Tier 2 + skills/{name}/support/ 支撑文件
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, List

from langchain_core.tools import tool

from skill_commands import build_skill_tool_result

if TYPE_CHECKING:
    from skills_tool import SkillLoader


def make_skill_view_tool(loader: "SkillLoader"):
    """
    工厂函数：绑定 SkillLoader 实例，生成 LangChain @tool。

    用法（Hermes_evolution_langgraph.py）：
        skill_view = make_skill_view_tool(runtime.loader)
        tools = [skill_view, scan_security, ...]
    """

    @tool
    def skill_view(name: str = "", tier: int = 2) -> str:
        """
        Load a skill by name before executing a task (Hermes skill_view).

        Args:
            name: Skill name from <available_skills> (e.g. python_code_review).
            tier: 0=category index, 1=metadata, 2=full skill body (default), 3=full + support files.

        Call tier=2 (or 3) when a skill matches or is partially relevant — mandatory per system prompt.
        """
        tier = int(tier)

        # --- Tier 0：返回全部分类 ---
        if tier <= 0:
            cats = loader.tier0_categories()
            if not cats:
                return "No skills indexed yet."
            lines = ["Skill categories (Tier 0):"]
            for cat, names in sorted(cats.items()):
                lines.append(f"  {cat}: {', '.join(names)}")
            return "\n".join(lines)

        skill_name = (name or "").strip()

        # --- 未指定 name：列出可用技能或 Tier-1 全量元数据 ---
        if not skill_name:
            available = loader.list_skill_names()
            if tier == 1:
                meta = loader.tier1_metadata()
                return json.dumps(meta, ensure_ascii=False, indent=2)
            return (
                "skill_view requires `name`. Available skills: "
                + (", ".join(available) if available else "(none)")
            )

        # --- 校验技能是否存在 ---
        if skill_name not in loader.list_skill_names():
            return (
                f"Skill '{skill_name}' not found. "
                f"Available: {', '.join(loader.list_skill_names()) or '(none)'}"
            )

        # --- Tier 1：单个技能元数据 ---
        if tier == 1:
            meta = loader.tier1_metadata(skill_name)
            return json.dumps(meta[0] if meta else {}, ensure_ascii=False, indent=2)

        # --- Tier 2/3：完整正文 ---
        payload = loader.tier2_full(skill_name)
        if not payload:
            return f"Failed to load skill '{skill_name}'."

        body = build_skill_tool_result(payload, tier=tier)

        # Tier 3：附加 support/ 目录下的配置文件、示例等
        if tier >= 3:
            support = loader.tier3_support_files(skill_name)
            if support:
                body += "\n\n## Support files (Tier 3)\n"
                for fname, content in support.items():
                    body += f"\n### {fname}\n{content}\n"

        return body

    return skill_view


def list_executor_tool_names(base_tools: List) -> List[str]:
    """调试辅助：列出工具名。"""
    return [t.name for t in base_tools]
