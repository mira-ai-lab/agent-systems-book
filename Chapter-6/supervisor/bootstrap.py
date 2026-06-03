"""Supervisor 包启动：把 Chapter-6 加入 sys.path 并加载 .env。"""

from __future__ import annotations

import sys
from pathlib import Path

_CH6 = Path(__file__).resolve().parent.parent
if str(_CH6) not in sys.path:
    sys.path.insert(0, str(_CH6))

from chapter6.paths import load_project_dotenv  # noqa: E402


def setup() -> Path:
    load_project_dotenv()
    return _CH6
