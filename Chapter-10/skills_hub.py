"""
skills_hub.py — 社区技能 Hub（agentskills.io 的本地演示替代）

真实 Hermes 可将成熟技能发布到 agentskills.io；
本书 demo 在 storage_dir/hub_export/ 模拟发布与安装。

目录结构：
  hub_export/
    manifest.json              — 已发布技能清单
    {skill_name}/
      SKILL.md
      support/                 — 可选支撑文件
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from skills_tool import SkillLoader


class SkillsHub:
    """
    演示级 Hub API：
      publish(skill_name) — 导出 SKILL.md + 更新 manifest
      install(skill_name) — 从 hub_export 复制到本地 skills/
      list_published()    — 列出已发布技能
    """

    HUB_URL = "https://agentskills.io"  # 真实平台 URL；demo 仅写 manifest 占位

    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.export_dir = self.storage_dir / "hub_export"
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.export_dir / "manifest.json"
        self.loader = SkillLoader(str(self.storage_dir / "skills"))

    def publish(self, skill_name: str, *, min_score: float = 7.0) -> Dict[str, Any]:
        """
        发布技能到本地 Hub。

        门槛：avg_score >= min_score（且 use_count > 0 时检查，避免未验证技能发布）
        """
        skill = self.loader.get(skill_name)
        if not skill:
            return {"ok": False, "error": f"技能不存在: {skill_name}"}
        if skill.avg_score < min_score and skill.use_count > 0:
            return {"ok": False, "error": f"平均得分 {skill.avg_score:.1f} 低于发布阈值 {min_score}"}

        dest = self.export_dir / skill_name
        dest.mkdir(parents=True, exist_ok=True)

        # 复制 SKILL.md
        src_md = self.loader.skill_md_path(skill_name)
        if src_md.exists():
            shutil.copy2(src_md, dest / "SKILL.md")

        # 复制 support/ 支撑文件（若有）
        support_src = self.storage_dir / "skills" / skill_name / "support"
        if support_src.is_dir():
            shutil.copytree(support_src, dest / "support", dirs_exist_ok=True)

        # 更新 manifest.json
        manifest = self._load_manifest()
        manifest[skill_name] = {
            "published_at": datetime.now().isoformat(),
            "description": skill.description,
            "version": skill.version,
            "hub_url": f"{self.HUB_URL}/skills/{skill_name}",
        }
        self._save_manifest(manifest)
        return {"ok": True, "skill": skill_name, "path": str(dest), "hub_url": manifest[skill_name]["hub_url"]}

    def install(self, skill_name: str) -> Dict[str, Any]:
        """从 hub_export 安装技能到本地 skills/ 目录。"""
        src = self.export_dir / skill_name / "SKILL.md"
        if not src.exists():
            return {"ok": False, "error": f"Hub 中无技能: {skill_name}"}

        skills_dir = self.storage_dir / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, skills_dir / f"{skill_name}.md")

        support_src = self.export_dir / skill_name / "support"
        if support_src.is_dir():
            shutil.copytree(support_src, skills_dir / skill_name / "support", dirs_exist_ok=True)

        self.loader.reload()
        return {"ok": True, "skill": skill_name, "installed_to": str(skills_dir)}

    def list_published(self) -> List[Dict[str, Any]]:
        """返回 manifest 中所有已发布技能。"""
        manifest = self._load_manifest()
        return [{"name": k, **v} for k, v in manifest.items()]

    def _load_manifest(self) -> Dict[str, Any]:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {}

    def _save_manifest(self, data: Dict[str, Any]) -> None:
        self.manifest_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
