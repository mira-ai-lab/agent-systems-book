"""
Quick connectivity test for a single A2A agent (Hotel Recommendation Agent by default).

What it does:
1) Fetches /.well-known/agent-card.json via A2A client
2) Sends a test user message (streaming)
3) Polls tasks/get until completed (or timeout)

Examples (PowerShell):
  python check_a2a_call.py --base-url http://127.0.0.1:9012/
  python check_a2a_call.py --query "推荐一个大同近古城的酒店，预算不超过500/晚"
  python check_a2a_call.py --timing
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from typing import Any

import httpx
from a2a.client import ClientConfig, create_client
from a2a.helpers import get_stream_response_text, new_text_message
from a2a.types import GetTaskRequest, Role, SendMessageRequest, TaskState
from google.protobuf.json_format import MessageToDict


def _dump(obj: Any) -> Any:
    if hasattr(obj, "DESCRIPTOR"):
        return MessageToDict(obj, preserving_proto_field_name=True)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


_TERMINAL_STATES = {
    TaskState.TASK_STATE_COMPLETED,
    TaskState.TASK_STATE_FAILED,
    TaskState.TASK_STATE_INPUT_REQUIRED,
    TaskState.TASK_STATE_REJECTED,
    TaskState.TASK_STATE_CANCELED,
}


def _state_name(state: int | None) -> str:
    if state is None:
        return "<none>"
    try:
        return TaskState.Name(int(state))
    except Exception:
        return str(state)


async def _amain(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Call an A2A agent to verify it responds.")
    p.add_argument("--base-url", default="http://127.0.0.1:9012/", help="Agent base URL")
    p.add_argument(
        "--query",
        default="推荐一个大同近古城的酒店，预算不超过500/晚，2026-06-04入住 2026-06-05退房",
        help="User query to send to the agent",
    )
    p.add_argument("--timeout", type=float, default=120.0, help="Overall timeout seconds")
    p.add_argument("--poll", type=float, default=1.0, help="Poll interval seconds")
    p.add_argument("--history-length", type=int, default=50, help="Task history_length for get_task")
    p.add_argument("--timing", action="store_true", help="Print timing summary")
    args = p.parse_args(argv)

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    base_url = args.base_url.rstrip("/") + "/"

    http_timeout = httpx.Timeout(timeout=float(args.timeout) + 10.0, connect=10.0)
    async with httpx.AsyncClient(timeout=http_timeout) as httpx_client:
        try:
            client = await create_client(
                base_url,
                ClientConfig(httpx_client=httpx_client, streaming=True),
            )
        except Exception as e:
            print(f"FAIL: create_client error: {e}", file=sys.stderr)
            return 2

        card = client._card  # noqa: SLF001
        print("OK: agent-card.json fetched")
        print(f"agent: {card.name or '<unknown>'}")

        context_id = f"test_hr_{uuid.uuid4()}"
        user_msg = new_text_message(
            args.query,
            context_id=context_id,
            role=Role.ROLE_USER,
        )
        send_req = SendMessageRequest(message=user_msg)

        task_id: str | None = None
        streamed_text: list[str] = []
        last_state: int | None = None
        first_chunk_at: float | None = None

        try:
            async for event in client.send_message(send_req):
                chunk = get_stream_response_text(event)
                if chunk:
                    if first_chunk_at is None:
                        first_chunk_at = time.perf_counter()
                    streamed_text.append(chunk)
                    print(chunk, end="", flush=True)
                if event.HasField("task"):
                    task_id = event.task.id or task_id
                    st = event.task.status.state
                    if st != last_state:
                        print(f"\ntask_state: {_state_name(st)}", flush=True)
                        last_state = st
                if event.HasField("status_update"):
                    st = event.status_update.status.state
                    if st != last_state:
                        print(f"\ntask_state: {_state_name(st)}", flush=True)
                        last_state = st
                    if st in _TERMINAL_STATES:
                        timings["total_to_completed_s"] = time.perf_counter() - t0
                        print("\nOK: task finished (stream)")
                        if args.timing:
                            _print_timing(timings, t0, first_chunk_at)
                        return 0
            print()
        except Exception as e:
            print(f"FAIL: send_message error: {e}", file=sys.stderr)
            return 2

        timings["send_message_wall_s"] = time.perf_counter() - t0
        if not task_id:
            if streamed_text:
                print("OK: received streamed text (no task_id)")
                return 0
            print("FAIL: no task_id and no streamed output", file=sys.stderr)
            return 2

        print(f"OK: task submitted: task_id={task_id} context_id={context_id}")

        deadline = time.time() + float(args.timeout)
        while time.time() < deadline:
            try:
                task = await client.get_task(
                    GetTaskRequest(id=task_id, history_length=int(args.history_length))
                )
            except Exception as e:
                print(f"WARN: get_task error: {e}", file=sys.stderr)
                await asyncio.sleep(float(args.poll))
                continue

            state = task.status.state
            if state != last_state:
                print(f"task_state: {_state_name(state)}")
                last_state = state

            if state in _TERMINAL_STATES:
                timings["total_to_completed_s"] = time.perf_counter() - t0
                print("OK: task finished")
                if args.timing:
                    _print_timing(timings, t0, first_chunk_at)
                print(json.dumps(_dump(task), ensure_ascii=False, indent=2))
                return 0

            await asyncio.sleep(float(args.poll))

        if args.timing:
            print("\n--- TIMING (timeout before completed) ---", file=sys.stderr)
            print(f"  send_message : {timings.get('send_message_wall_s', 0):.2f}s", file=sys.stderr)
            print(f"  total so far : {time.perf_counter() - t0:.2f}s", file=sys.stderr)
        print(f"FAIL: timeout after {args.timeout}s waiting for task completion", file=sys.stderr)
        return 3


def _print_timing(timings: dict[str, float], t0: float, first_chunk_at: float | None) -> None:
    print("\n--- TIMING ---", file=sys.stderr)
    print(f"  send_message (wall)     : {timings.get('send_message_wall_s', 0):.2f}s", file=sys.stderr)
    if first_chunk_at is not None:
        print(f"  time to first chunk     : {first_chunk_at - t0:.2f}s", file=sys.stderr)
    print(f"  total to completed      : {timings.get('total_to_completed_s', time.perf_counter() - t0):.2f}s", file=sys.stderr)
    print("---", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_amain()))
