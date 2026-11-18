"""Centralized path and environment bootstrap for Chapter 6.

This module defines the chapter's canonical directory constants, registers the
repository root and the Chapter 2/5 directories on ``sys.path``, and loads the
repository-level and chapter-level ``.env`` files. Other Chapter 6 modules use
these constants and loading helpers instead of constructing paths locally.

Main responsibilities:

- Provide shared paths for Chapter 6, Chapter 2, Chapter 5, and Chroma storage.
- Make cross-chapter imports resolve consistently from notebooks and scripts.
- Load model, search, and memory configuration from the expected ``.env`` files.

:author: zhanghong, liuhui
:date: 2026-11-16
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv


CH6_DIR = Path(__file__).resolve().parent
BOOK_ROOT = CH6_DIR.parent
CH2_DIR = BOOK_ROOT / "Chapter-2"
CH5_DIR = BOOK_ROOT / "Chapter-5"
CHROMA_DIR = CH6_DIR / "chroma_memory"
CH6_ENV_FILE = CH6_DIR / ".env"
BOOK_ENV_FILE = BOOK_ROOT / ".env"
if str(BOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOK_ROOT))
if str(CH2_DIR) not in sys.path:
    sys.path.insert(0, str(CH2_DIR))
if str(CH5_DIR) not in sys.path:
    sys.path.insert(0, str(CH5_DIR))


def load_project_dotenv(*, override: bool = False) -> Path:
    """Load Chapter-6 and repository-level ``.env`` files in order.

    Returns:
        The repository-level environment file when present; otherwise the
        Chapter-6 environment file.
    """
    load_dotenv(CH6_ENV_FILE, override=override)
    load_dotenv(BOOK_ENV_FILE, override=override)
    return BOOK_ENV_FILE if BOOK_ENV_FILE.is_file() else CH6_ENV_FILE
