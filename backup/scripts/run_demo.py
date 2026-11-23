"""企业路由引擎 CLI 演示。

默认（书稿 travel 多城市样例）::

    python scripts/run_demo.py
    # travel + profile=auto + TRAVEL_SAMPLE_QUERY

能力展示（Fixed Graph 全链路）::

    python scripts/run_demo.py --legacy-graph --stream
    # 直连 LangGraphOrchestrator，完整 Ch2 预调查 + Ch4 TaskPlanner 多 Agent

Router 统一入口::

    python scripts/run_demo.py --profile workflow --show-graph
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

_CHAPTER8_ROOT = Path(__file__).resolve().parent.parent
if str(_CHAPTER8_ROOT) not in sys.path:
    sys.path.insert(0, str(_CHAPTER8_ROOT))

if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

from agent_framework.bootstrap import route
from agent_framework.bootstrap.platform import create_runtime
from agent_framework.config import load_project_dotenv
from agent_framework.stream.events import public_event
from agent_framework.tracing import get_logger, log_info, setup_observability

load_project_dotenv()
setup_observability()
logger = get_logger(__name__)

DEFAULT_DOMAIN = "travel"
DEFAULT_PROFILE = "auto"

TRAVEL_SAMPLE_QUERY = """
你能帮我规划一个下周的多城市旅行吗？我还没想好行程顺序……
大概是上海、苏州、杭州这几个地方？需要包含行程路线、酒店推荐、
天气情况和美食攻略。我喜欢住安静的酒店，预算每晚不超过800元。
""".strip()


def _print_routing_summary(result: dict) -> None:
    print("\n--- 路由摘要 ---")
    print(f"domain: {result.get('resolved_domain') or result.get('domain')}")
    print(f"resolved_profile: {result.get('resolved_profile')}")
    matches = result.get("knowledge_matches") or []
    if matches:
        print(f"knowledge_matches: {len(matches)} 条")
    plan = result.get("routing_plan") or {}
    candidates = plan.get("candidates") or []
    if candidates:
        top = ", ".join(f"{c['name']}({c['score']:.2f})" for c in candidates[:3])
        print(f"candidates: {top}")


def _print_legacy_summary(result: dict) -> None:
    subtasks = result.get("subtask_results") or {}
    plan = result.get("execution_plan") or {}
    subtask_list = plan.get("subtasks") or []
    print("\n--- Fixed Graph 摘要 ---")
    print(f"subtasks: {len(subtask_list)}")
    if subtask_list:
        agents = ", ".join(
            f"{item.get('task_id')}→{item.get('agent')}"
            for item in subtask_list
            if item.get("agent")
        )
        if agents:
            print(f"agents: {agents}")
    print(f"completed: {len(subtasks)}")


def _print_stream_event(payload: dict, ctx: dict) -> bool:
    """打印 SSE/iter_request_stream 事件；aggregate 阶段 token 流式输出。

    ctx 维护 subtask_token_keys 与 aggregate_streaming 状态。
    返回 aggregate_streaming 标志。
    """
    event_type = payload.get("type") or ""
    data = payload.get("data") or {}
    aggregate_streaming = bool(ctx.get("aggregate_streaming"))
    subtask_token_keys: set[tuple[str, str]] = ctx.setdefault("subtask_token_keys", set())

    if event_type == "graph.progress":
        print(data.get("message") or "", flush=True)
        return aggregate_streaming
    if event_type == "graph.subtask.token":
        task_id = str(data.get("task_id") or "?")
        agent = str(data.get("agent") or "?")
        token = data.get("token") or ""
        key = (task_id, agent)
        if key not in subtask_token_keys:
            print(f"\n[{task_id} → {agent}] ", end="", flush=True)
            subtask_token_keys.add(key)
        print(token, end="", flush=True)
        return aggregate_streaming
    if event_type == "graph.subtask.completed":
        task_id = str(data.get("task_id") or "?")
        agent = str(data.get("agent") or "?")
        summary = data.get("summary") or ""
        key = (task_id, agent)
        if key in subtask_token_keys:
            print(f"\n✓ [{task_id} → {agent}] done", flush=True)
        else:
            print(f"\n✓ [{task_id} → {agent}] {summary}", flush=True)
        return aggregate_streaming
    if event_type == "graph.token":
        print(data.get("token") or "", end="", flush=True)
        ctx["aggregate_streaming"] = True
        return True
    if event_type.startswith("router."):
        print(f"[{event_type}]", flush=True)
        return aggregate_streaming
    if event_type == "graph.node" and data.get("node") == "aggregate":
        print("\n📝 聚合结果...", flush=True)
        return aggregate_streaming
    return aggregate_streaming


def _create_legacy_orchestrator(domain: str):
    from agent_framework.orchestration.fixed_graph.orchestrator import LangGraphOrchestrator

    return LangGraphOrchestrator(
        domain=domain,
        enable_memory=False,
        enable_guess_agent=True,
    )


async def run_legacy_chat(*, domain: str, stream: bool) -> None:
    from agent_framework.orchestration.fixed_graph.orchestrator import LangGraphOrchestrator

    orchestrator: LangGraphOrchestrator = _create_legacy_orchestrator(domain)
    mode = "流式" if stream else "批量"
    print(f"书稿 Fixed Graph · {domain} · {mode}对话 · 输入 quit 退出")
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
            result = await orchestrator.process_request_stream(query, thread_id=thread_id)
        else:
            result = await orchestrator.process_request(query, thread_id=thread_id)
            print("\n" + "=" * 80)
            print(result.get("final_response", ""))
            print("=" * 80)
        _print_legacy_summary(result)


async def run_router_chat(*, domain: str, profile: str, stream: bool) -> None:
    mode = "流式" if stream else "批量"
    print(f"企业路由引擎 · {domain} · profile={profile} · {mode}对话 · 输入 quit 退出")
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
            runtime = create_runtime(domain, profile=profile, enable_memory=False)
            stream_ctx: dict = {}
            async for event in runtime.iter_request_stream(query, thread_id=thread_id):
                payload = public_event(event)
                if payload["type"] == "final":
                    _print_routing_summary(payload.get("data") or {})
                    if not stream_ctx.get("aggregate_streaming"):
                        print("\n" + "=" * 80)
                        print((payload.get("data") or {}).get("final_response", ""))
                    else:
                        print("\n" + "=" * 80, flush=True)
                    print("=" * 80)
                else:
                    _print_stream_event(payload, stream_ctx)
        else:
            result = await route(query, domain=domain, profile=profile, thread_id=thread_id)
            _print_routing_summary(result)
            print("\n" + "=" * 80)
            print(result.get("final_response", ""))
            print("=" * 80)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="企业路由引擎 CLI 演示（默认 travel + TRAVEL_SAMPLE_QUERY + profile=auto）"
    )
    parser.add_argument("-q", "--query", help="单条问题")
    parser.add_argument(
        "--domain",
        default=DEFAULT_DOMAIN,
        help=f"领域名（默认 {DEFAULT_DOMAIN}）",
    )
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help=f"执行 Profile（默认 {DEFAULT_PROFILE}；--legacy-graph 时忽略）",
    )
    parser.add_argument(
        "--legacy-graph",
        action="store_true",
        help="直连 LangGraphOrchestrator（书稿 Ch2+Ch4 TaskPlanner，不经 RouterEngine）",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="流式输出：Router 阶段 / Fixed Graph 进度",
    )
    parser.add_argument("--chat", action="store_true", help="交互对话模式")
    parser.add_argument(
        "--show-graph",
        action="store_true",
        help="打印 LangGraph 结构",
    )
    args = parser.parse_args()

    domain = args.domain.strip()
    profile = args.profile.strip()
    if args.query:
        query = args.query.strip()
    else:
        query = TRAVEL_SAMPLE_QUERY

    if args.legacy_graph and profile != DEFAULT_PROFILE:
        print(f"提示：--legacy-graph 忽略 --profile={profile}，走完整 Fixed Graph。")

    if args.show_graph:
        if args.legacy_graph:
            orchestrator = _create_legacy_orchestrator(domain)
            print("\n--- LangGraph 工作流结构 ---")
            orchestrator.show_graph()
            orchestrator.save_graph()
            print()
        elif profile != "workflow":
            print("提示：--show-graph（Router 路径）仅适用于 profile=workflow，已跳过。")
        else:
            from agent_framework.orchestration.protocol import MODE_FIXED_GRAPH

            runtime = create_runtime(domain, profile="workflow", enable_memory=False)
            backend = await runtime._get_backend(MODE_FIXED_GRAPH)
            print("\n--- LangGraph 工作流结构 ---")
            backend.show_graph()
            backend.save_graph()
            print()

    if args.chat:
        if args.legacy_graph:
            await run_legacy_chat(domain=domain, stream=args.stream)
        else:
            await run_router_chat(domain=domain, profile=profile, stream=args.stream)
        return

    thread_id = f"demo-{uuid.uuid4().hex[:8]}"
    if args.legacy_graph:
        orchestrator = _create_legacy_orchestrator(domain)
        if args.stream:
            result = await orchestrator.process_request_stream(query, thread_id=thread_id)
        else:
            result = await orchestrator.process_request(query, thread_id=thread_id)
            print("\n" + "=" * 80)
            print(result.get("final_response", ""))
            print("=" * 80)
        _print_legacy_summary(result)
    elif args.stream:
        runtime = create_runtime(domain, profile=profile, enable_memory=False)
        result: dict = {}
        stream_ctx: dict = {}
        async for event in runtime.iter_request_stream(query, thread_id=thread_id):
            payload = public_event(event)
            if payload["type"] == "final":
                result = dict(payload.get("data") or {})
                _print_routing_summary(result)
                if not stream_ctx.get("aggregate_streaming"):
                    print("\n" + "=" * 80)
                    print(result.get("final_response", ""))
                else:
                    print("\n" + "=" * 80, flush=True)
                print("=" * 80)
            else:
                _print_stream_event(payload, stream_ctx)
    else:
        result = await route(query, domain=domain, profile=profile, thread_id=thread_id)
        _print_routing_summary(result)
        print("\n" + "=" * 80)
        print(result.get("final_response", ""))
        print("=" * 80)

    log_info(
        logger,
        "demo.summary",
        domain=domain,
        profile=profile if not args.legacy_graph else "legacy-graph",
        final_length=len(result.get("final_response") or ""),
        trace_id=result.get("trace_id"),
        stream=args.stream,
        legacy_graph=args.legacy_graph,
    )


if __name__ == "__main__":
    asyncio.run(main())
