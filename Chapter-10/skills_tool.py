"""
skills_tool.py — agentskills.io / Hermes 标准 SKILL.md 读写 + Tier 0–3 渐进加载

文件格式：
  ---
  技能名称 / 功能描述 / 版本号 / 适配平台 / Hermes框架元数据
  name / description / version / platforms / metadata.hermes
  ---
  # Title   技能标题
  ## Trigger conditions   触发条件
  ## Steps         执行步骤 （请使用编号列表编写）
  ## Pitfalls     避坑要点
  ## Verification   结果验证标准

sidecar：{name}.stats.json 存 use_count / avg_score / version（运行时统计）

Tier 加载（配合 skill_view）：
  Tier 0 — tier0_categories()     分类 → 技能名列表
  Tier 1 — tier1_metadata()       元数据 JSON
  Tier 2 — tier2_full()           完整 Skill dict
  Tier 3 — tier3_support_files()  support/ 支撑文件
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Pydantic 模型（对应 SKILL.md frontmatter + 正文）
# ---------------------------------------------------------------------------


class HermesConfigItem(BaseModel):
    """metadata.hermes.config 单项（如 deploy 技能需要的 vercel.team 配置项）。"""

    key: str          # 配置键名
    description: str = ""  # 配置项说明
    default: str = ""      # 默认值
    prompt: str = ""       # 向用户询问时的提示语


class HermesMetadataBlock(BaseModel):
    """metadata.hermes 块：分类标签、关联技能、工具集依赖等 Hermes 扩展元数据。"""

    tags: List[str] = Field(default_factory=list)                  # 分类标签，tags[0] 用作 Tier 0 索引分类
    related_skills: List[str] = Field(default_factory=list)        # 相关技能名列表
    fallback_for_toolsets: List[str] = Field(default_factory=list) # 某工具集不可用时的兜底技能
    requires_toolsets: List[str] = Field(default_factory=list)     # 执行此技能需要的工具集（如 terminal）
    config: List[HermesConfigItem] = Field(default_factory=list)   # 技能运行所需的可配置项


class SkillMetadata(BaseModel):
    """SKILL.md frontmatter 中 metadata 字段的容器。"""

    hermes: HermesMetadataBlock = Field(default_factory=HermesMetadataBlock)


class Skill(BaseModel):
    """
    内存中的技能对象（整个技能系统的核心数据结构）。

     frontmatter 字段：技能名称 / 功能描述 / 版本号 / 适配平台 / 框架元数据
                     name / description / version / platforms / metadata
    ---
    正文字段：
    # Title           技能标题
    ## Trigger conditions     触发条件
    ## Steps                 执行步骤（编号列表）
    ## Pitfalls              避坑要点
    ## Verification          结果验证标准

    ---
    运行时字段：使用次数 / 平均评分 / 创建时间 / 更新时间（源自 stats.json）
             use_count / avg_score / created_at / updated_at
    """

    name: str
    description: str
    version: str = "1.0.0"
    platforms: List[str] = Field(default_factory=lambda: ["macos", "linux", "windows"])
    metadata: SkillMetadata = Field(default_factory=SkillMetadata)

    title: str = ""
    trigger_conditions: List[str] = Field(default_factory=list)
    steps: List[str] = Field(default_factory=list)
    pitfalls: List[str] = Field(default_factory=list)
    verification: List[str] = Field(default_factory=list)

    use_count: int = 0
    avg_score: float = 0.0
    created_at: str = ""
    updated_at: str = ""

    @field_validator("version", mode="before")
    @classmethod
    def _coerce_version(cls, v: Any) -> str:
        """
        Pydantic 字段校验器：在正式校验前规范化 version 字段。

        兼容旧版数据：整数 1 → 字符串 "1.0.0"；None → "1.0.0"。
        """
        if v is None:
            return "1.0.0"
        if isinstance(v, int):
            return f"{v}.0.0"
        return str(v)

    @property
    def category(self) -> str:
        """
        只读属性：技能的 Tier 0 分类名。

        取 metadata.hermes.tags 的第一项；无 tag 时返回 "general"。
        用于 tier0_categories() 和 <available_skills> 索引分组。
        """
        tags = self.metadata.hermes.tags
        return tags[0] if tags else "general"

    @classmethod
    def from_extractor_dict(cls, data: Dict[str, Any]) -> "Skill":
        """
        工厂方法：将 LLM skill_extractor 输出的 JSON dict 转为 Skill 对象。

        调用方：skill_extractor_node → skill_manage(create)。

        兼容 legacy 字段名：
          procedure  → steps
          category   → metadata.hermes.tags
          task_type  → 追加到 tags
        若缺少 title，则从 name 自动生成（snake_case → Title Case）。
        """
        payload = dict(data)
        if "procedure" in payload and "steps" not in payload:
            payload["steps"] = payload.pop("procedure")
        if "category" in payload and "metadata" not in payload:
            tag = payload.pop("category")
            payload["metadata"] = {"hermes": {"tags": [tag] if tag else []}}
        if "task_type" in payload:
            task_type = payload.pop("task_type")
            meta = payload.setdefault("metadata", {"hermes": {}})
            hermes = meta.setdefault("hermes", {})
            tags = hermes.setdefault("tags", [])
            if task_type and task_type not in tags:
                tags.append(task_type)
        if not payload.get("title"):
            name = payload.get("name", "skill")
            payload["title"] = name.replace("_", " ").title()
        return cls(**payload)


# ---------------------------------------------------------------------------
# SkillLoader：磁盘 ↔ 内存，Tier 0–3 API
# ---------------------------------------------------------------------------


class SkillLoader:
    """
    技能库加载器（技能系统的「数据库层」）。

    职责：
      - 扫描 skills_dir 下 *.md（主格式）和 legacy *.json
      - 解析为 Skill 对象缓存到 _cache
      - 提供 Tier 0–3 渐进加载 API（供 skill_view / 索引缓存使用）
      - 提供 SKILL.md 渲染与版本号工具方法
    """

    def __init__(self, skills_dir: str):
        """
        初始化加载器并立即 reload()。

        Args:
            skills_dir: 技能目录路径，如 my_agent_memory/skills/
        """
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Skill] = {}  # 内存缓存：skill_name → Skill
        self.reload()

    def reload(self) -> None:
        """
        全量重载技能库到内存 _cache。

        加载顺序：
          1. *.md — agentskills.io 标准 SKILL.md（优先）
          2. *.json — legacy 格式（仅当同名 .md 不存在时加载）
        跳过 *.stats.json（那是统计 sidecar，不是完整技能定义）。

        调用时机：初始化、skill create/patch 后、rebuild_index_node。
        """
        self._cache.clear()
        for path in self.skills_dir.glob("*.md"):
            skill = self._parse_skill_md(path)
            if skill:
                self._cache[skill.name] = skill
        for path in self.skills_dir.glob("*.json"):
            if path.name.endswith(".stats.json"):
                continue
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if not data.get("name"):
                continue
            skill = Skill.from_extractor_dict(data)
            if skill.name not in self._cache:
                self._cache[skill.name] = skill

    def list_skill_names(self) -> List[str]:
        """返回当前已加载的全部技能名（字母序）。"""
        return sorted(self._cache.keys())

    def tier0_categories(self) -> Dict[str, List[str]]:
        """
        Tier 0：按分类返回技能名列表（最轻量，仅索引用）。

        Returns:
            {"devops": ["deploy_nextjs", ...], "general": [...], ...}

        调用方：skill_view(name="", tier=0)。
        """
        cats: Dict[str, List[str]] = {}
        for s in self._cache.values():
            cats.setdefault(s.category, []).append(s.name)
        return cats

    def tier1_metadata(self, name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Tier 1：返回元数据 JSON（不含 steps 等正文）。

        Args:
            name: 指定技能名则只返回该条；None 则返回全部。

        Returns:
            [{"name", "description", "version", "platforms", "tags"}, ...]

        调用方：skill_view(tier=1)、skills_index_cache L3 冷扫描。
        """
        skills = [self._cache[name]] if name else list(self._cache.values())
        return [
            {
                "name": s.name,
                "description": s.description,
                "version": s.version,
                "platforms": s.platforms,
                "tags": s.metadata.hermes.tags,
            }
            for s in skills
        ]

    def tier2_full(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Tier 2：返回完整 Skill dict（含 trigger/steps/pitfalls/verification）。

        Args:
            name: 技能名

        Returns:
            Skill.model_dump() 字典；技能不存在时返回 None。

        调用方：skill_view(name, tier=2) → format_skill_content → ToolMessage。
        """
        s = self._cache.get(name)
        return s.model_dump() if s else None

    def tier3_support_files(self, name: str) -> Dict[str, str]:
        """
        Tier 3：读取 skills/{name}/support/ 下的支撑文件。

        Args:
            name: 技能名

        Returns:
            {文件名: 文件内容}；目录不存在时返回空 dict。

        典型内容：配置模板、示例脚本、参考文档等。
        调用方：skill_view(name, tier=3) 在 Tier 2 正文后追加。
        """
        support_dir = self.skills_dir / name / "support"
        if not support_dir.is_dir():
            return {}
        return {
            fp.name: fp.read_text(encoding="utf-8")
            for fp in support_dir.iterdir()
            if fp.is_file()
        }

    def get(self, name: str) -> Optional[Skill]:
        """
        从内存缓存获取 Skill 对象。

        Args:
            name: 技能名

        Returns:
            Skill 实例；不存在时返回 None。

        调用方：skill_patcher_node、evaluator update_stats、SkillManager。
        """
        return self._cache.get(name)

    def skill_md_path(self, name: str) -> Path:
        """
        返回技能 SKILL.md 的磁盘路径。

        Args:
            name: 技能名

        Returns:
            skills_dir/{name}.md

        调用方：SkillManager 读写 / patch / rollback。
        """
        return self.skills_dir / f"{name}.md"

    # ------------------------------------------------------------------
    # SKILL.md 解析（yaml.safe_load frontmatter + 正文章节）
    # ------------------------------------------------------------------

    def _parse_skill_md(self, path: Path) -> Optional[Skill]:
        """
        解析单个 SKILL.md 文件为 Skill 对象。

        流程：
          1. 用 --- 分隔符切分 YAML frontmatter 与 Markdown 正文
          2. yaml.safe_load 解析 frontmatter
          3. _parse_body_sections 解析正文章节
          4. 读取同名 {name}.stats.json 合并运行时统计

        Args:
            path: SKILL.md 文件路径

        Returns:
            Skill 对象；格式不合法（无 frontmatter）时返回 None。
        """
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---"):
            return None
        parts = text.split("---", 2)
        if len(parts) < 3:
            return None

        meta = yaml.safe_load(parts[1]) or {}
        body = parts[2]
        name = meta.get("name", path.stem)

        # 合并 stats sidecar 中的运行时统计（不写入 SKILL.md frontmatter）
        stats_path = self.skills_dir / f"{name}.stats.json"
        stats: Dict[str, Any] = {}
        if stats_path.exists():
            stats = json.loads(stats_path.read_text(encoding="utf-8"))

        sections = self._parse_body_sections(body)
        title = sections.get("title") or name.replace("_", " ").title()

        return Skill(
            name=name,
            description=str(meta.get("description", "")),
            version=meta.get("version", stats.get("version", "1.0.0")),
            platforms=list(meta.get("platforms") or ["macos", "linux", "windows"]),
            metadata=SkillMetadata(**(meta.get("metadata") or {})),
            title=title,
            trigger_conditions=sections.get("trigger_conditions", []),
            steps=sections.get("steps", []),
            pitfalls=sections.get("pitfalls", []),
            verification=sections.get("verification", []),
            use_count=int(stats.get("use_count", 0)),
            avg_score=float(stats.get("avg_score", 0.0)),
            created_at=stats.get("created_at", ""),
            updated_at=stats.get("updated_at", ""),
        )

    @staticmethod
    def _parse_body_sections(body: str) -> Dict[str, Any]:
        """
        解析 SKILL.md Markdown 正文，提取各章节内容。

        识别规则：
          # Title           → title 字段
          ## Trigger ...    → trigger_conditions（- bullet）
          ## Steps/Procedure → steps（1. 编号 或 - bullet）
          ## Pitfalls        → pitfalls（- bullet）
          ## Verification    → verification（- bullet）
        支持中英文标题关键字（触发/步骤/陷阱/验证）。

        Args:
            body: frontmatter 之后的 Markdown 正文

        Returns:
            {"title", "trigger_conditions", "steps", "pitfalls", "verification"}
        """
        sections: Dict[str, List[str]] = {
            "trigger_conditions": [],
            "steps": [],
            "pitfalls": [],
            "verification": [],
        }
        title = ""
        current: Optional[str] = None  # 当前正在收集的章节名

        for line in body.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            # 一级标题 → 技能标题
            if stripped.startswith("# ") and not stripped.startswith("## "):
                title = stripped[2:].strip()
                continue

            # 二级标题 → 切换当前章节
            low = stripped.lower()
            if low.startswith("## trigger") or "触发" in stripped:
                current = "trigger_conditions"
                continue
            if low.startswith("## steps") or low.startswith("## procedure") or stripped == "## 步骤":
                current = "steps"
                continue
            if low.startswith("## pitfalls") or "陷阱" in stripped:
                current = "pitfalls"
                continue
            if low.startswith("## verification") or "验证" in stripped:
                current = "verification"
                continue

            # 收集章节内容行
            if current and stripped.startswith("- "):
                sections[current].append(stripped[2:])
            elif current == "steps":
                m = re.match(r"^\d+\.\s+(.*)", stripped)
                if m:
                    sections["steps"].append(m.group(1))

        return {"title": title, **sections}

    # ------------------------------------------------------------------
    # SKILL.md 渲染（skill_manage create 写盘）
    # ------------------------------------------------------------------

    @staticmethod
    def render_skill_md(skill: Skill) -> str:
        """
        将 Skill 对象序列化为 agentskills.io 标准 SKILL.md 文本。

        输出结构：YAML frontmatter + Markdown 四段正文。
        空字段写入占位符（待补充），避免生成不完整文件。

        Args:
            skill: 内存中的 Skill 对象

        Returns:
            可直接写入磁盘的完整 SKILL.md 字符串。

        调用方：SkillManager._create / _atomic_save。
        """
        frontmatter = {
            "name": skill.name,
            "description": skill.description,
            "version": skill.version,
            "platforms": skill.platforms,
            "metadata": skill.metadata.model_dump(),
        }
        front_yaml = yaml.dump(
            frontmatter,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        ).strip()

        title = skill.title or skill.name.replace("_", " ").title()
        triggers = "\n".join(f"- {t}" for t in skill.trigger_conditions) or "- （待补充触发条件）"
        steps = "\n".join(f"{i}. {s}" for i, s in enumerate(skill.steps, 1)) or "1. （待补充步骤）"
        pitfalls = "\n".join(f"- {p}" for p in skill.pitfalls) or "- （待补充）"
        verification = "\n".join(f"- {v}" for v in skill.verification) or "- （待补充）"

        return f"""---
{front_yaml}
---

# {title}

## Trigger conditions
{triggers}

## Steps
{steps}

## Pitfalls
{pitfalls}

## Verification
{verification}
"""

    @staticmethod
    def bump_patch_version(version: str) -> str:
        """
        semver 补丁版本号 +1（只递增第三位）。

        示例：1.0.0 → 1.0.1；1.0 → 1.0.1（不足三位补 0）

        Args:
            version: 当前版本字符串

        Returns:
            递增后的版本字符串

        调用方：SkillManager patch / rollback 成功后更新 stats.json。
        """
        parts = str(version).split(".")
        while len(parts) < 3:
            parts.append("0")
        try:
            parts[2] = str(int(parts[2]) + 1)
        except ValueError:
            parts[2] = "1"
        return ".".join(parts[:3])
