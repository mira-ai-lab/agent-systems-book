"""完整演示：LangGraph 中心智能体多智能体协作。"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

# 允许在 scripts/ 下直接运行：python run_demo.py
_CHAPTER8_ROOT = Path(__file__).resolve().parent.parent
if str(_CHAPTER8_ROOT) not in sys.path:
    sys.path.insert(0, str(_CHAPTER8_ROOT))

if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

from agent_framework.config import load_project_dotenv
from agent_framework.orchestration.fixed_graph.orchestrator import LangGraphOrchestrator
from agent_framework.tracing import get_logger, log_info, setup_observability

load_project_dotenv()
setup_observability()
logger = get_logger(__name__)

DEFAULT_QUERY = """
你能帮我规划一个下周的多城市旅行吗？我还没想好行程顺序……
大概是上海、苏州、杭州这几个地方？需要包含行程路线、酒店推荐、
天气情况和美食攻略。我喜欢住安静的酒店，预算每晚不超过800元。
""".strip()
# DEFAULT_QUERY = """
# 北京未来2周的天气如何
# """.strip()

async def run_chat(orchestrator: LangGraphOrchestrator, stream: bool) -> None:
    mode = "流式" if stream else "批量"
    print(f"旅行中心智能体 · {mode}对话 · 输入 quit 退出")
    while True:
        try:
            query = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not query or query.lower() in ("quit", "exit", "q", "退出"):
            print("再见。")
            break
        thread_id = f"chat-{uuid.uuid4().hex[:8]}"
        if stream:
            await orchestrator.process_request_stream(query, thread_id=thread_id)
        else:
            result = await orchestrator.process_request(query, thread_id=thread_id)
            print("\n" + "=" * 80)
            print(result.get("final_response", ""))
            print("=" * 80)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Chapter-8 LangGraph 旅行多智能体演示")
    parser.add_argument("-q", "--query", help="单条问题（默认使用内置多城旅行示例）")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="流式输出：阶段进度 + 最终回复逐字显示（推荐）",
    )
    parser.add_argument("--chat", action="store_true", help="交互对话模式")
    parser.add_argument("--no-graph", action="store_true", help="跳过打印/保存图结构")
    args = parser.parse_args()

    orchestrator = LangGraphOrchestrator(
        enable_memory=False,
        enable_guess_agent=True,
    )

    if not args.no_graph:
        print("\n--- LangGraph 工作流结构 ---")
        orchestrator.show_graph()
        orchestrator.save_graph()
        print()

    if args.chat:
        await run_chat(orchestrator, stream=args.stream)
        return

    query = (args.query or DEFAULT_QUERY).strip()
    if args.stream:
        result = await orchestrator.process_request_stream(query, thread_id="fixed_graph")
    else:
        result = await orchestrator.process_request(query, thread_id="fixed_graph")
        print("\n" + "=" * 80)
        print(result.get("final_response", ""))
        print("=" * 80)

    log_info(
        logger,
        "demo.summary",
        subtask_count=len(result.get("subtask_results") or {}),
        final_length=len(result.get("final_response") or ""),
        trace_id=result.get("trace_id"),
        stream=args.stream,
    )


if __name__ == "__main__":
    asyncio.run(main())
