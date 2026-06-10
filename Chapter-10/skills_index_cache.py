"""
skills_index_cache.py — Skills 索引三层缓存（对齐 Hermes 架构图）

请求 <available_skills> 索引时的路径：

  L1 进程内 LRU（OrderedDict）     ~0.001ms   同进程重复请求
       ↓ miss
  L2 .skills_prompt_snapshot.json  ~1ms       进程重启后仍可用
       ↓ hash 不匹配 / 不存在
  L3 扫描 skills/*.md frontmatter  50–500ms+ 冷启动 / 技能变更后

content_hash：skills/ 下 .md/.json 的路径+mtime+size 指纹；
              任一文件变化则 L2 失效，触发 L3 重建。
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from skills_tool import SkillLoader

SNAPSHOT_FILENAME = ".skills_prompt_snapshot.json"
SNAPSHOT_VERSION = 1
DEFAULT_LRU_SIZE = 16


class SkillsIndexCache:
    """Skills 索引 L1 LRU + L2 磁盘快照 + L3 冷扫描。"""

    def __init__(
        self,
        storage_dir: str,
        loader: "SkillLoader",
        *,
        lru_size: int = DEFAULT_LRU_SIZE,
    ):
        self.storage_dir = Path(storage_dir)
        self.loader = loader
        self.skills_dir = loader.skills_dir
        self.snapshot_path = self.storage_dir / SNAPSHOT_FILENAME
        self._lru_size = lru_size
        # L1 key = "{tier}:{content_hash}" → <available_skills> 文本块
        self._l1: OrderedDict[str, str] = OrderedDict()

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def get_index_block(self, *, tier: int = 1) -> str:
        """收到索引请求：L1 → L2 → L3，返回 Hermes 分类缩进格式。"""
        content_hash = self._compute_content_hash()
        l1_key = f"{tier}:{content_hash}"

        # --- L1 命中 ---
        if l1_key in self._l1:
            self._l1.move_to_end(l1_key)
            return self._l1[l1_key]

        # --- L2 命中（快照 hash 与当前 skills 目录一致）---
        snapshot = self._load_snapshot()
        if snapshot and snapshot.get("content_hash") == content_hash:
            meta = snapshot.get("tier1_metadata") or []
            block = self._build_index_block(meta, tier=tier)
            self._l1_put(l1_key, block)
            return block

        # --- L3 冷扫描 ---
        t0 = time.perf_counter()
        meta = self._scan_tier1_metadata()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        block = self._build_index_block(meta, tier=tier)
        self._write_snapshot(content_hash, meta)
        self._l1_put(l1_key, block)
        # 同步完整 SkillLoader（Tier 2/3 执行依赖 _cache）
        self.loader.reload()
        if elapsed_ms > 10:
            print(f"  … [L3] 索引冷扫描 {len(meta)} 个技能 ({elapsed_ms:.0f}ms)", flush=True)
        return block

    def get_index_xml(self, *, tier: int = 1) -> str:
        """兼容旧方法名。"""
        return self.get_index_block(tier=tier)

    def invalidate_cache(self) -> None:
        """skill create/patch 后：清 L1 + 删 L2，下次请求走 L3。"""
        self._l1.clear()
        if self.snapshot_path.exists():
            self.snapshot_path.unlink()

    def last_build_info(self) -> Dict[str, Any]:
        """调试 / 书稿：查看缓存状态。"""
        snap = self._load_snapshot()
        return {
            "l1_entries": len(self._l1),
            "snapshot_exists": self.snapshot_path.exists(),
            "content_hash": self._compute_content_hash(),
            "snapshot": snap,
        }

    # ------------------------------------------------------------------
    # L1 LRU
    # ------------------------------------------------------------------

    def _l1_put(self, key: str, xml: str) -> None:
        self._l1[key] = xml
        self._l1.move_to_end(key)
        while len(self._l1) > self._lru_size:
            self._l1.popitem(last=False)  # 淘汰最久未用

    # ------------------------------------------------------------------
    # L2 磁盘快照
    # ------------------------------------------------------------------

    def _load_snapshot(self) -> Optional[Dict[str, Any]]:
        if not self.snapshot_path.exists():
            return None
        try:
            return json.loads(self.snapshot_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _write_snapshot(self, content_hash: str, tier1_metadata: List[Dict[str, Any]]) -> None:
        """原子写入 L2 快照（.tmp → replace）。"""
        payload = {
            "version": SNAPSHOT_VERSION,
            "content_hash": content_hash,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "skills_dir": str(self.skills_dir),
            "tier1_metadata": tier1_metadata,
        }
        tmp = self.snapshot_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.snapshot_path)

    # ------------------------------------------------------------------
    # L3 + content_hash 有效性
    # ------------------------------------------------------------------

    def _compute_content_hash(self) -> str:
        """skills 目录 .md/.json 文件指纹；任一变化 → hash 变 → L2 失效。"""
        h = hashlib.sha256()
        if not self.skills_dir.is_dir():
            return h.hexdigest()
        for path in sorted(self.skills_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix not in (".md", ".json"):
                continue
            rel = path.relative_to(self.skills_dir).as_posix()
            stat = path.stat()
            h.update(f"{rel}:{stat.st_mtime_ns}:{stat.st_size}\n".encode())
        return h.hexdigest()

    def _scan_tier1_metadata(self) -> List[Dict[str, Any]]:
        """
        L3：只解析 frontmatter（yaml.safe_load），不读正文。

        比 SkillLoader.reload() 轻量，专供索引构建。
        """
        import yaml

        results: List[Dict[str, Any]] = []
        if not self.skills_dir.is_dir():
            return results

        for path in sorted(self.skills_dir.glob("*.md")):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if not text.startswith("---"):
                continue
            parts = text.split("---", 2)
            if len(parts) < 2:
                continue
            meta = yaml.safe_load(parts[1]) or {}
            name = meta.get("name", path.stem)
            hermes = (meta.get("metadata") or {}).get("hermes") or {}
            version = str(meta.get("version", "1.0.0"))
            stats_path = self.skills_dir / f"{name}.stats.json"
            if stats_path.exists():
                try:
                    stats = json.loads(stats_path.read_text(encoding="utf-8"))
                    version = str(stats.get("version", version))
                except (json.JSONDecodeError, OSError):
                    pass
            results.append(
                {
                    "name": name,
                    "description": str(meta.get("description", "")),
                    "version": version,
                    "platforms": list(meta.get("platforms") or []),
                    "tags": list(hermes.get("tags") or []),
                }
            )
        return results

    # ------------------------------------------------------------------
    # 索引块渲染（注入 System Prompt 的 <available_skills>）
    # ------------------------------------------------------------------

    @staticmethod
    def _build_index_block(tier1_metadata: List[Dict[str, Any]], *, tier: int) -> str:
        """
        Hermes 分类缩进格式：

        <available_skills>
          devops:
            - deploy-nextjs: Deploy Next.js apps to Vercel...
        </available_skills>
        """
        if not tier1_metadata:
            return "<available_skills>\n  (none)\n</available_skills>"

        cats: Dict[str, List[Dict[str, Any]]] = {}
        for m in tier1_metadata:
            tags = m.get("tags") or []
            cat = tags[0] if tags else "general"
            cats.setdefault(cat, []).append(m)

        lines = ["<available_skills>", ""]
        for cat in sorted(cats.keys()):
            lines.append(f"  {cat}:")
            if tier == 0:
                for m in sorted(cats[cat], key=lambda x: x["name"]):
                    lines.append(f"    - {m['name']}")
            else:
                for m in sorted(cats[cat], key=lambda x: x["name"]):
                    desc = m.get("description", "").strip()
                    lines.append(f"    - {m['name']}: {desc}")
            lines.append("")
        lines.append("</available_skills>")
        return "\n".join(lines)

    @staticmethod
    def _build_xml(tier1_metadata: List[Dict[str, Any]], *, tier: int) -> str:
        """兼容旧名。"""
        return SkillsIndexCache._build_index_block(tier1_metadata, tier=tier)
