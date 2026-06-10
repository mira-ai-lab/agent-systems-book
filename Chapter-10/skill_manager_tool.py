"""
skill_manager_tool.py — skill_manage(action=create|patch|rollback)

Hermes 技能生命周期写盘入口：
  create   — LLM 抽取 JSON → 校验 → 安全扫描 → 写 SKILL.md
  patch    — Fuzzy Match 定位章节 → 原子替换 → 失败 rollback
  rollback — 从 skills/{name}/versions/ 恢复上一版

统计信息（use_count / avg_score）写入 *.stats.json sidecar，不污染 frontmatter。
"""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from security_scan import scan_skill_dict, scan_skill_text
from skills_tool import Skill, SkillLoader


class SkillManager:
    """技能 CRUD + 版本备份 + 原子写入。"""

    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.skills_dir = self.storage_dir / "skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.loader = SkillLoader(str(self.skills_dir))

    def manage(
        self,
        action: str,
        *,
        skill_data: Optional[Dict[str, Any]] = None,
        skill_name: Optional[str] = None,
        patch: Optional[Dict[str, str]] = None,
        version_tag: Optional[str] = None,
    ) -> Dict[str, Any]:
        """统一入口：create | patch | rollback。"""
        action = action.lower().strip()
        if action == "create":
            return self._create(skill_data or {})
        if action == "patch":
            return self._patch(skill_name or "", patch or {})
        if action == "rollback":
            return self._rollback(skill_name or "", version_tag)
        return {"ok": False, "error": f"未知 action: {action}"}

    # ------------------------------------------------------------------
    # create
    # ------------------------------------------------------------------

    def _create(self, skill_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建新技能：校验 → 安全扫描 → 写 SKILL.md + stats。"""
        try:
            skill = Skill.from_extractor_dict(skill_data)
        except ValidationError as exc:
            return {"ok": False, "error": str(exc)}

        scan = scan_skill_dict(skill.model_dump())
        if not scan.passed:
            return {"ok": False, "error": "安全扫描未通过", "issues": scan.issues}

        if not self._validate_skill(skill):
            return {"ok": False, "error": "技能验证失败：description / steps 不完整"}

        now = datetime.now().isoformat()
        skill.created_at = now
        skill.updated_at = now
        skill.version = "1.0.0"
        self._atomic_save(skill, backup=False)
        self.loader.reload()
        return {"ok": True, "action": "create", "skill": skill.name, "version": skill.version}

    # ------------------------------------------------------------------
    # patch + rollback
    # ------------------------------------------------------------------

    def _patch(self, skill_name: str, patch: Dict[str, str]) -> Dict[str, Any]:
        """
        改进已有技能。

        patch 格式：
          section: trigger_conditions | steps | pitfalls | verification
          old_fragment: 要替换的原句（Fuzzy Match 定位）
          new_fragment: 新内容
        """
        skill = self.loader.get(skill_name)
        if not skill:
            return {"ok": False, "error": f"技能不存在: {skill_name}"}

        md_path = self.loader.skill_md_path(skill_name)
        original = md_path.read_text(encoding="utf-8") if md_path.exists() else SkillLoader.render_skill_md(skill)
        self._backup_version(skill_name, original)  # patch 前备份

        section = patch.get("section", "steps")
        if section == "procedure":  # 兼容旧字段名
            section = "steps"
        old_fragment = patch.get("old_fragment", "")
        new_fragment = patch.get("new_fragment", "")
        if not new_fragment:
            return {"ok": False, "error": "patch 缺少 new_fragment"}

        updated_text, matched = self._fuzzy_apply(original, old_fragment, new_fragment, section)
        if not matched:
            self._rollback(skill_name, None)
            return {"ok": False, "error": "Fuzzy Match 未定位到可替换片段"}

        scan = scan_skill_text(updated_text)
        if not scan.passed:
            self._rollback(skill_name, None)
            return {"ok": False, "error": "patch 安全扫描未通过", "issues": scan.issues}

        self._atomic_write_text(md_path, updated_text)
        self.loader.reload()
        patched = self.loader.get(skill_name)
        if patched:
            patched.version = SkillLoader.bump_patch_version(patched.version)  # 1.0.0 → 1.0.1
            patched.updated_at = datetime.now().isoformat()
            self._save_stats(patched)
        return {
            "ok": True,
            "action": "patch",
            "skill": skill_name,
            "version": patched.version if patched else None,
        }

    def _rollback(self, skill_name: str, version_tag: Optional[str]) -> Dict[str, Any]:
        """回滚到 versions/ 下某备份；version_tag 为空则取最新备份。"""
        versions_dir = self.skills_dir / skill_name / "versions"
        if not versions_dir.is_dir():
            return {"ok": False, "error": "无版本历史"}

        if version_tag:
            src = versions_dir / version_tag
        else:
            backups = sorted(versions_dir.glob("*.md"), reverse=True)
            if not backups:
                return {"ok": False, "error": "无可用备份"}
            src = backups[0]

        if not src.exists():
            return {"ok": False, "error": f"版本不存在: {src.name}"}

        dst = self.loader.skill_md_path(skill_name)
        shutil.copy2(src, dst)
        self.loader.reload()
        skill = self.loader.get(skill_name)
        if skill:
            skill.version = SkillLoader.bump_patch_version(skill.version)
            skill.updated_at = datetime.now().isoformat()
            self._save_stats(skill)
        return {"ok": True, "action": "rollback", "skill": skill_name, "restored": src.name}

    # ------------------------------------------------------------------
    # 统计与持久化
    # ------------------------------------------------------------------

    def update_stats(self, name: str, score: float) -> None:
        """evaluator 调用：更新 use_count 与滑动平均 avg_score（70% 历史 + 30% 新分）。"""
        skill = self.loader.get(name)
        if not skill:
            return
        skill.use_count += 1
        skill.avg_score = 0.7 * skill.avg_score + 0.3 * score
        skill.updated_at = datetime.now().isoformat()
        self._save_stats(skill)

    def _validate_skill(self, skill: Skill) -> bool:
        return bool(skill.name and skill.description and len(skill.steps) >= 2)

    def _atomic_save(self, skill: Skill, *, backup: bool) -> None:
        """写 SKILL.md + stats；backup=True 时先存 versions/。"""
        md_path = self.loader.skill_md_path(skill.name)
        content = SkillLoader.render_skill_md(skill)
        if backup and md_path.exists():
            self._backup_version(skill.name, md_path.read_text(encoding="utf-8"))
        self._atomic_write_text(md_path, content)
        self._save_stats(skill)

    def _save_stats(self, skill: Skill) -> None:
        """运行时统计写入 sidecar，不进入 SKILL.md frontmatter。"""
        stats_path = self.skills_dir / f"{skill.name}.stats.json"
        stats = {
            "use_count": skill.use_count,
            "avg_score": skill.avg_score,
            "version": skill.version,
            "created_at": skill.created_at,
            "updated_at": skill.updated_at,
        }
        self._atomic_write_text(stats_path, json.dumps(stats, indent=2, ensure_ascii=False))

    def _backup_version(self, skill_name: str, content: str) -> None:
        """备份到 skills/{name}/versions/{timestamp}.md。"""
        versions_dir = self.skills_dir / skill_name / "versions"
        versions_dir.mkdir(parents=True, exist_ok=True)
        tag = datetime.now().strftime("%Y%m%d_%H%M%S") + ".md"
        (versions_dir / tag).write_text(content, encoding="utf-8")

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        """先写 .tmp 再 os.replace，避免写一半崩溃导致文件损坏。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)

    # ------------------------------------------------------------------
    # Fuzzy Match patch 引擎
    # ------------------------------------------------------------------

    @staticmethod
    def _fuzzy_apply(
        text: str,
        old_fragment: str,
        new_fragment: str,
        section: str,
    ) -> Tuple[str, bool]:
        """
        在 SKILL.md 指定章节内定位并替换片段。

        策略：
          1. 精确 substring 匹配
          2. SequenceMatcher 模糊匹配 bullet / 编号行（阈值 0.55）
          3. old_fragment 为空时在章节末尾追加
        """
        section_markers = {
            "trigger_conditions": ("## Trigger conditions", "## Steps"),
            "steps": ("## Steps", "## Pitfalls"),
            "pitfalls": ("## Pitfalls", "## Verification"),
            "verification": ("## Verification", None),
        }
        start_marker, end_marker = section_markers.get(section, ("## Steps", "## Pitfalls"))
        start_idx = text.find(start_marker)
        if start_idx < 0:
            return text, False
        end_idx = text.find(end_marker, start_idx + 1) if end_marker else len(text)
        section_text = text[start_idx:end_idx]

        # 精确匹配
        if old_fragment and old_fragment in section_text:
            new_section = section_text.replace(old_fragment, new_fragment, 1)
            return text[:start_idx] + new_section + text[end_idx:], True

        # 收集章节内 bullet / 编号行作为模糊匹配候选
        candidates = []
        for ln in section_text.splitlines():
            s = ln.strip()
            if s.startswith("- "):
                candidates.append(ln)
            elif re.match(r"^\d+\.\s+", s):
                candidates.append(ln)

        best_ratio, best_line = 0.0, ""
        for ln in candidates:
            plain = ln.strip()
            if plain.startswith("- "):
                plain = plain[2:]
            else:
                m = re.match(r"^\d+\.\s+(.*)", plain)
                plain = m.group(1) if m else plain
            ratio = SequenceMatcher(None, old_fragment, plain).ratio() if old_fragment else 0.0
            if ratio > best_ratio:
                best_ratio, best_line = ratio, ln

        if best_ratio >= 0.55 and best_line:
            if section == "steps" and not re.match(r"^\d+\.", new_fragment.strip()):
                idx = candidates.index(best_line) + 1 if best_line in candidates else len(candidates) + 1
                new_line = f"{idx}. {new_fragment.lstrip('0123456789. ')}"
            elif best_line.strip().startswith("- "):
                new_line = f"- {new_fragment.lstrip('- ')}"
            else:
                new_line = new_fragment
            new_section = section_text.replace(best_line, new_line, 1)
            return text[:start_idx] + new_section + text[end_idx:], True

        # 追加新行
        if not old_fragment:
            if section == "steps":
                n = len(candidates) + 1
                new_section = section_text.rstrip() + f"\n{n}. {new_fragment}\n"
            else:
                new_section = section_text.rstrip() + f"\n- {new_fragment}\n"
            return text[:start_idx] + new_section + text[end_idx:], True

        return text, False


def skill_manage(action: str, **kwargs) -> Dict[str, Any]:
    """LangChain @tool 占位；实际由 HermesRuntime.skill_manager.manage() 调用。"""
    raise RuntimeError("请通过 HermesRuntime.skill_manager.manage() 调用")
