"""Chapter-7/Mcp 路径引导：定位书根目录、Chapter-6、加载 .env。"""

from __future__ import annotations

import sys
from pathlib import Path

# .../Chapter-7/Mcp/mcp_paths.py
MCP_DIR = Path(__file__).resolve().parent
BOOK_ROOT = MCP_DIR.parent.parent
CH6_DIR = BOOK_ROOT / "Chapter-6"


def bootstrap_paths(*, load_env: bool = True) -> None:
    """把 Mcp 目录与 Chapter-6 加入 sys.path，供 notebook / 脚本 import。"""
    for d in (MCP_DIR, CH6_DIR):
        s = str(d)
        if d.is_dir() and s not in sys.path:
            sys.path.insert(0, s)
    if load_env:
        from chapter6.paths import load_project_dotenv

        load_project_dotenv()
