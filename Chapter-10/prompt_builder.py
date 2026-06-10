"""
prompt_builder.py — 扫描 SKILL.md，构建 <available_skills> 并注入 System Prompt

调用链：
  PromptBuilder.build_system_prompt()
    → SkillsIndexCache.get_index_block()   # L1/L2/L3 缓存
    → build_cacheable_system_prompt()      # skill_commands.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from skill_commands import build_cacheable_system_prompt
from skills_index_cache import SkillsIndexCache

if TYPE_CHECKING:
    from skills_tool import SkillLoader

# ---------------------------------------------------------------------------
# SKILLS_GUIDANCE：告诉 LLM 何时 create / patch 技能（写入 System Prompt）
# ---------------------------------------------------------------------------

SKILLS_GUIDANCE = (
    "After completing a complex task (5+ tool calls), fixing a tricky error, "
    "or discovering a non-trivial workflow, save the approach as a "
    "skill with skill_manage so you can reuse it next time.\n"
    "When using a skill and finding it outdated, incomplete, or wrong, "
    "patch it immediately with skill_manage(action='patch') — don't wait to be asked. "
    "Skills that aren't maintained become liabilities."
)

SKILLS_GUIDANCE_ZH = (
    "完成复杂任务（≥5 次 tool calls）、修复棘手错误或沉淀非平凡工作流后，"
    "请调用 skill_manage(action='create') 保存为技能以便下次复用。\n"
    "使用技能时若发现过时、不完整或错误，立即 skill_manage(action='patch') 修补，"
    "勿等待用户提醒；未维护的技能会成为负债。"
)


class PromptBuilder:
    """
    技能索引 + System Prompt 构建器。

    索引三级缓存详见 skills_index_cache.py：
      L1 进程 LRU → L2 .skills_prompt_snapshot.json → L3 frontmatter 冷扫描
    """

    def __init__(self, loader: "SkillLoader", storage_dir: str):
        self.loader = loader
        self._index_cache = SkillsIndexCache(storage_dir, loader)

    def invalidate_cache(self) -> None:
        """技能 create / patch 后调用：清空 L1 + 删除 L2 快照。"""
        self._index_cache.invalidate_cache()

    def build_available_skills_index(self, *, tier: int = 1) -> str:
        """
        构建 Hermes 风格 <available_skills> 分类索引。

        tier=0：仅技能名；tier=1（默认）：名 + 一行描述
        """
        return self._index_cache.get_index_block(tier=tier)

    def build_available_skills_xml(self, *, tier: int = 1) -> str:
        """兼容旧方法名（早期版本用 XML 格式索引）。"""
        return self.build_available_skills_index(tier=tier)

    def build_system_prompt(self, memory_snapshot: str, *, base: Optional[str] = None) -> str:
        """组装 executor / router 使用的完整 System Prompt。"""
        base = base or "你是 Hermes 自我进化 Agent，擅长从任务中学习并复用技能。"
        return build_cacheable_system_prompt(
            base=base,
            memory_snapshot=memory_snapshot,
            available_skills_index=self.build_available_skills_index(),
            skills_guidance=f"{SKILLS_GUIDANCE}\n\n{SKILLS_GUIDANCE_ZH}",
        )
