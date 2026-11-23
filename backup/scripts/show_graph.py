"""仅展示 LangGraph 工作流结构（不调用 LLM，无需有效 API Key）。"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许在 scripts/ 下直接运行：python show_graph.py
_CHAPTER8_ROOT = Path(__file__).resolve().parent.parent
if str(_CHAPTER8_ROOT) not in sys.path:
    sys.path.insert(0, str(_CHAPTER8_ROOT))

if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

from agent_framework.orchestration.fixed_graph.visualize import GraphVisualizer


def main() -> None:
    print("正在编译 StateGraph（仅结构，不执行）...\n", flush=True)
    viz = GraphVisualizer.standalone()
    viz.print_all()
    paths = viz.save_all()
    print("\n提示: 用浏览器打开 output/central_agent_graph.png 查看流程图", flush=True)
    if "png" in paths:
        print(f"      PNG 路径: {paths['png'].resolve()}", flush=True)


if __name__ == "__main__":
    main()
