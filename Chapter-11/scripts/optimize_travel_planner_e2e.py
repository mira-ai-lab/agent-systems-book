#!/usr/bin/env python3
"""Travel Planner E2E graph 优化（Phase B2 快捷入口）。

等价于 ``optimize_travel_planner.py --backend textgrad_graph --objective e2e``。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    argv = list(sys.argv)
    if "--backend" not in argv:
        argv.extend(["--backend", "textgrad_graph"])
    if "--objective" not in argv:
        argv.extend(["--objective", "e2e"])
    sys.argv = argv

    target = _ROOT / "scripts" / "optimize_travel_planner.py"
    spec = importlib.util.spec_from_file_location("optimize_travel_planner", target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {target}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["optimize_travel_planner"] = module
    spec.loader.exec_module(module)
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
