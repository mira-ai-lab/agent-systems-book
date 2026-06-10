"""完整演示：LangGraph 中心智能体多智能体协作。"""

from __future__ import annotations

import asyncio
import sys

if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

from travel_multi_agent.config import load_project_dotenv
from travel_multi_agent.orchestration.fixed_graph.orchestrator import LangGraphOrchestrator
from travel_multi_agent.tracing import get_logger, setup_observability

load_project_dotenv()
setup_observability()
logger = get_logger(__name__)


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

    result = await orchestrator.process_request(query, thread_id="fixed_graph")
    from travel_multi_agent.tracing import log_info
    log_info(
        logger,
        "demo.summary",
        subtask_count=len(result.get("subtask_results") or {}),
        final_length=len(result.get("final_response") or ""),
        trace_id=result.get("trace_id"),
    )


if __name__ == "__main__":
    asyncio.run(main())
