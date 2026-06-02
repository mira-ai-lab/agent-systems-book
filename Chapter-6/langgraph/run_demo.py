"""LangGraph 中心智能体演示入口"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

CHAPTER6_DIR = Path(__file__).resolve().parent.parent
LG_DIR = Path(__file__).resolve().parent
if str(LG_DIR) not in sys.path:
    sys.path.insert(0, str(LG_DIR))

from orchestrator import LangGraphOrchestrator  # noqa: E402


async def main() -> None:
    orchestrator = LangGraphOrchestrator(enable_memory=False)

    print("\n--- LangGraph 工作流结构 ---")
    orchestrator.show_graph()
    orchestrator.save_graph()
    print()

    query = """
你能帮我规划一个下周的多城市旅行吗？我还没想好行程顺序……
大概是上海、苏州、杭州这几个地方？需要包含行程路线、酒店推荐、
天气情况和美食攻略。我喜欢住安静的酒店，预算每晚不超过800元。
"""

    result = await orchestrator.process_request(query, thread_id="langgraph_demo")
    print(f"\n子任务数: {len(result.get('subtask_results') or {})}")
    print(f"最终回复长度: {len(result.get('final_response') or '')} 字符")


if __name__ == "__main__":
    asyncio.run(main())
