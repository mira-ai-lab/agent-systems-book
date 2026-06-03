"""Chapter-6 统一路径与 .env 加载（全项目唯一来源）。"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# chapter6/paths.py → Chapter-6/
CH6_DIR = Path(__file__).resolve().parent.parent
BOOK_ROOT = CH6_DIR.parent
CHROMA_DIR = CH6_DIR / "chroma_memory"


def load_project_dotenv(*, override: bool = False) -> None:
    """依次尝试 Chapter-6/.env 与书仓库根 .env。"""
    load_dotenv(CH6_DIR / ".env", override=override)
    load_dotenv(BOOK_ROOT / ".env", override=override)


def ensure_ch6_on_path() -> Path:
    """脚本直跑时把 Chapter-6 加入 sys.path，便于 `import chapter6`。"""
    import sys

    root = str(CH6_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)
    return CH6_DIR
