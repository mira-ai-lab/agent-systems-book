"""
skill_commands.py — System Prompt 文案与技能正文格式化

职责：
  1. 定义 Hermes「Skills (mandatory)」强制规则（中英文）
  2. 组装可缓存的 System Prompt（索引 + 记忆 + 生命周期指引）
  3. build_skill_invocation_message / build_skill_tool_result — 技能注入（[SYSTEM: ...] 前缀）

设计要点：
  - System Prompt 只放轻量索引 <available_skills>（可缓存）
  - 技能全文通过 skill_view 工具返回 ToolMessage，不修改 System Prompt
"""

from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import HumanMessage

# ---------------------------------------------------------------------------
# Hermes System Prompt 强制规则（与 agent/prompt_builder.py 对齐）
# ---------------------------------------------------------------------------

SKILLS_MANDATORY = """## Skills (mandatory)

Before replying, scan the skills below. If a skill matches or is even
partially relevant to your task, you MUST load it with skill_view(name)
and follow its instructions before proceeding.

Only proceed without loading a skill if genuinely none are relevant."""

SKILLS_MANDATORY_ZH = """## 技能（强制）

回复之前，扫描下方技能索引。若某技能与任务匹配或 even partially relevant（部分相关），
你必须通过 skill_view(name) 加载完整技能并严格遵循其步骤，然后再继续。

仅当确实没有任何相关技能时，才允许不加载直接处理。"""

# executor 运行时提示：告诉 LLM 用 skill_view 工具，而非改 System Prompt
RUNTIME_SKILL_VIEW_NOTE = (
    "Runtime: call the skill_view tool (tier=2) to load full skill instructions "
    "before using domain tools. This preserves the cached system prompt above."
)


RUNTIME_SKILL_VIEW_NOTE_ZH = (
    "运行时规则：在调用领域专用工具前，需先执行二级 skill_view 工具以加载完整技能指令。"
    "该操作会保留上文已缓存的系统提示词。"
)

# ---------------------------------------------------------------------------
# 技能正文格式化（供 skill_view 工具 / User Message 注入）
#
# Hermes 约定：用 [SYSTEM: ...] 前缀标记「权威指令块」，消息 role 仍为 user/tool，
# 不把技能全文塞进 System Prompt（保护 Prompt Cache）。
# ---------------------------------------------------------------------------


def build_skill_activation_note(
    skill_name: str,
    *,
    source: str = "user",
    tier: int = 2,
) -> str:
    """
    Hermes 激活说明（内嵌于 HumanMessage / ToolMessage 正文，非 SystemMessage role）。

    source:
      user       — 用户 /skill 或 build_skill_invocation_message 显式调用
      skill_view — executor ReAct 调 skill_view 工具后写入 ToolMessage
    """
    if source == "skill_view":
        return (
            f'[SYSTEM: skill_view loaded the "{skill_name}" skill (tier={tier}). '
            "You MUST follow its instructions below before using other tools.]"
        )
    return (
        f'[SYSTEM: The user has invoked the "{skill_name}" skill, indicating they '
        "want you to follow its instructions. The full skill content is loaded below.]"
    )


def format_skill_content(skill_payload: Dict[str, Any], *, tier: int = 2) -> str:
    """
    将技能 dict 格式化为 Markdown 字符串。

    tier <= 1：仅元数据（name / description / tags）
    tier >= 2：完整 Trigger conditions / Steps / Pitfalls / Verification
    """
    name = skill_payload.get("name", "unknown")

    if tier <= 1:
        tags = (skill_payload.get("metadata") or {}).get("hermes", {}).get("tags", [])
        return (
            f"Skill: {name}\n"
            f"Description: {skill_payload.get('description', '')}\n"
            f"Version: {skill_payload.get('version', '1.0.0')}\n"
            f"Tags: {', '.join(tags)}\n"
        )

    title = skill_payload.get("title") or name.replace("_", " ").title()
    triggers = skill_payload.get("trigger_conditions") or []
    steps = skill_payload.get("steps") or skill_payload.get("procedure") or []
    pitfalls = skill_payload.get("pitfalls") or []
    verification = skill_payload.get("verification") or []
    return (
        f"# {title}\n"
        f"Description: {skill_payload.get('description', '')}\n\n"
        f"## Trigger conditions\n"
        + "\n".join(f"- {t}" for t in triggers)
        + "\n\n## Steps\n"
        + "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
        + "\n\n## Pitfalls\n"
        + "\n".join(f"- {p}" for p in pitfalls)
        + "\n\n## Verification\n"
        + "\n".join(f"- {v}" for v in verification)
        + "\n"
    )


def build_skill_invocation_message(
    skill_payload: Dict[str, Any],
    *,
    tier: int = 2,
    user_instruction: str = "",
    source: str = "user",
) -> HumanMessage:
    """
    Hermes build_skill_invocation_message 等价实现：用户显式调用技能时注入 HumanMessage。

    主路径仍是 executor 调 skill_view → ToolMessage（见 build_skill_tool_result）；
    此函数供 /skill 命令、测试或非 ReAct 场景。
    """
    name = skill_payload.get("name", "unknown")
    activation = build_skill_activation_note(name, source=source, tier=tier)
    body = format_skill_content(skill_payload, tier=tier)
    if user_instruction.strip():
        content = f"{activation}\n\n{user_instruction.strip()}\n\n---\n{body}"
    else:
        content = f"{activation}\n\n---\n{body}"
    return HumanMessage(
        content=content,
        additional_kwargs={"skill_invocation": True, "skill_name": name, "tier": tier},
    )


def build_skill_tool_result(
    skill_payload: Dict[str, Any],
    *,
    tier: int = 2,
) -> str:
    """skill_view 工具返回值：ToolMessage 正文（同样用 [SYSTEM: ...] 前缀）。"""
    name = skill_payload.get("name", "unknown")
    activation = build_skill_activation_note(name, source="skill_view", tier=tier)
    return f"{activation}\n\n{format_skill_content(skill_payload, tier=max(tier, 2))}"


def format_skill_user_message(skill_payload: Dict[str, Any], *, tier: int = 2) -> HumanMessage:
    """兼容旧名 → build_skill_invocation_message。"""
    return build_skill_invocation_message(skill_payload, tier=tier, source="user")


# ---------------------------------------------------------------------------
# 可缓存 System Prompt 组装
# ---------------------------------------------------------------------------


def build_cacheable_system_prompt(
    *,
    base: str,
    memory_snapshot: str,
    available_skills_index: str,
    skills_guidance: str = "",
    include_runtime_note: bool = True,
) -> str:
    """
    拼接完整 System Prompt，结构固定以利于 Prompt Cache：

        base 角色设定
        → L1 hot memory
        → Skills (mandatory) + <available_skills>
        → SKILLS_GUIDANCE（create / patch 触发条件）
    """
    memory_block = memory_snapshot.strip() or "（暂无热记忆）"
    guidance_block = skills_guidance.strip()
    guidance_section = (
        f"\n## Skill lifecycle guidance (SKILLS_GUIDANCE)\n{guidance_block}\n"
        if guidance_block
        else ""
    )
    runtime_note = f"\n{RUNTIME_SKILL_VIEW_NOTE}\n" if include_runtime_note else ""

    return f"""{base}

## L1 hot memory
{memory_block}

{SKILLS_MANDATORY}

{available_skills_index}

{SKILLS_MANDATORY_ZH}
{runtime_note}{guidance_section}"""
