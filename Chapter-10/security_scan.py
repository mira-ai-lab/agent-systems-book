"""
security_scan.py — 技能内容安全扫描

在 skill_manage(create) 和 skill_manage(patch) 写入磁盘前调用，
拦截明显高风险内容（eval、破坏性命令、硬编码密码等）。

生产环境可替换为专用安全服务或更完整策略引擎。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass
class ScanResult:
    """扫描结果：passed=True 表示可写入。"""

    passed: bool
    issues: List[str]


# (正则模式, 违规说明)
_DANGEROUS_PATTERNS: List[tuple[str, str]] = [
    (r"\beval\s*\(", "禁止使用 eval()"),
    (r"\bexec\s*\(", "禁止使用 exec()"),
    (r"__import__\s*\(", "禁止动态 __import__"),
    (r"rm\s+-rf\s+/|del\s+/|format\s+c:", "禁止破坏性系统命令"),
    (r"ignore\s+(all\s+)?security|绕过\s*安全", "禁止绕过安全策略的表述"),
    (r"password\s*=\s*['\"][^'\"]+['\"]", "技能中不应硬编码密码"),
]


def scan_skill_text(text: str) -> ScanResult:
    """扫描 SKILL.md 全文或 patch 后的 Markdown 文本。"""
    issues: List[str] = []
    for pattern, message in _DANGEROUS_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            issues.append(message)
    return ScanResult(passed=len(issues) == 0, issues=issues)


def scan_skill_dict(skill_data: dict) -> ScanResult:
    """扫描 LLM 抽取的技能 JSON（create 前）。"""
    import json

    return scan_skill_text(json.dumps(skill_data, ensure_ascii=False))
