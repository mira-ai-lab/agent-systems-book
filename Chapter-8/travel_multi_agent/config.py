"""Project configuration: paths and environment loading.

Registers the repository root and Chapter-6 on ``sys.path`` so the chapter can
reuse the root ``llm.py`` helpers and the flat Chapter-6 modules (``task_plan``,
``agent_routing``, ``domain_agents``).

:author: zhanghong, liuhui, xingrui
:date: 2026-11-22
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

# config.py lives at Chapter-8/travel_multi_agent/config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # Chapter-8
BOOK_ROOT = PROJECT_ROOT.parent                          # repository root
CHAPTER6_DIR = BOOK_ROOT / "Chapter-6"
for _path in (BOOK_ROOT, CHAPTER6_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

CHROMA_DIR = PROJECT_ROOT / "chroma_memory"
AGENT_CARDS_DIR = PROJECT_ROOT / "travel_multi_agent" / "agents" / "agent_cards"
GRAPH_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "orchestration" / "fixed_graph" / "output"
)
TRACES_DIR = PROJECT_ROOT / "traces"

_dotenv_loaded = False


def load_project_dotenv(*, override: bool = False) -> None:
    """Load Chapter-8/.env and then the repository-root .env (idempotent)."""
    global _dotenv_loaded
    load_dotenv(PROJECT_ROOT / ".env", override=override)
    load_dotenv(BOOK_ROOT / ".env", override=override)
    _dotenv_loaded = True
