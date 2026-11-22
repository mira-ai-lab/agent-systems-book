"""Chapter-7/Mcp path bootstrap.

Chapter 7 reuses the hotel search implementation from Chapter 5 and the
HotelAgent capability from Chapter 6. This module is the only place that
registers those cross-chapter paths and loads the project environment.

Main capabilities:

- Register Chapter-5 and Chapter-6 import paths.
- Load the shared project environment for the MCP server and client.

:author: zhanghong, liuhui
:date: 2026-11-22
"""

from __future__ import annotations

import sys
from pathlib import Path

# .../Chapter-7/Mcp/paths.py
MCP_DIR = Path(__file__).resolve().parent
CHAPTER7_DIR = MCP_DIR.parent
BOOK_ROOT = CHAPTER7_DIR.parent
CH5_DIR = BOOK_ROOT / "Chapter-5"
CH6_DIR = BOOK_ROOT / "Chapter-6"


def bootstrap_paths() -> None:
    """Register reusable Chapter 5/6 modules and load project environment."""
    for d in (MCP_DIR, CH5_DIR, CH6_DIR, BOOK_ROOT):
        s = str(d)
        if d.is_dir() and s not in sys.path:
            sys.path.insert(0, s)
    # Cross-chapter reuse: Chapter-6/config.py
    from config import load_project_dotenv

    load_project_dotenv()
