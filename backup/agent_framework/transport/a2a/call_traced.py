"""A2A 远程调用（带 latc span 与 W3C trace 传播）。"""

from __future__ import annotations

import time
from typing import Optional

from agent_framework.observability.metrics import record_a2a_call
from agent_framework.tracing.trace_provider import (
    current_trace_add_event,
    inject_trace_context,
    span_name,
    trace_span,
)


def _require_a2a_sdk() -> None:
    try:
        import a2a.client  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "A2A transport 需要安装：pip install 'agent-platform[a2a]'"
        ) from exc


@trace_span(
    name=span_name("a2a.call"),
    attrs_args=["endpoint", "query", "context_id"],
    record_result=False,
)
async def a2a_call_remote(
    endpoint: str,
    query: str,
    context_id: Optional[str] = None,
) -> tuple[str, Optional[str], bool]:
    """调用远程 A2A Agent。返回 (response_text, new_context_id, success)。"""
    from a2a.helpers import get_stream_response_text, new_text_message
    from a2a.types import Role, SendMessageRequest, TaskState

    _require_a2a_sdk()
    import httpx
    from a2a.client import ClientConfig, create_client

    terminal = {
        TaskState.TASK_STATE_COMPLETED,
        TaskState.TASK_STATE_FAILED,
        TaskState.TASK_STATE_INPUT_REQUIRED,
        TaskState.TASK_STATE_REJECTED,
        TaskState.TASK_STATE_CANCELED,
    }
    url = endpoint.rstrip("/") + "/"
    headers: dict[str, str] = {}
    inject_trace_context(headers)
    t0 = time.perf_counter()
    terminal_state: Optional[str] = None
    httpx_client = httpx.AsyncClient(
        timeout=httpx.Timeout(timeout=120.0, connect=10.0),
        headers=headers,
    )
    try:
        client = await create_client(
            url,
            ClientConfig(httpx_client=httpx_client, streaming=True),
        )
        user_msg = new_text_message(
            query,
            context_id=context_id,
            role=Role.ROLE_USER,
        )
        send_req = SendMessageRequest(message=user_msg)
        chunks: list[str] = []
        new_context_id = context_id or user_msg.context_id

        async for event in client.send_message(send_req):
            text = get_stream_response_text(event)
            if text:
                chunks.append(text)
            if event.HasField("task"):
                new_context_id = event.task.context_id or new_context_id
            if event.HasField("status_update"):
                new_context_id = event.status_update.context_id or new_context_id
                state = event.status_update.status.state
                if state in terminal:
                    terminal_state = TaskState.Name(state)
                    break

        response_text = "".join(chunks).strip()
        duration_ms = int((time.perf_counter() - t0) * 1000)
        if not response_text:
            current_trace_add_event(
                "a2a.empty_response",
                {
                    "endpoint": url,
                    "context_id": new_context_id or "",
                    "duration_ms": duration_ms,
                    "task_state": terminal_state or "",
                },
            )
            record_a2a_call(url, status="empty", duration_sec=duration_ms / 1000.0)
            return "A2A 服务未返回文本内容", new_context_id, False

        current_trace_add_event(
            "sub_agent_conversation",
            {
                "transport": "a2a",
                "endpoint": url,
                "query": query[:500],
                "response": response_text[:500],
                "status": "completed",
                "context_id": new_context_id or "",
                "duration_ms": duration_ms,
                "task_state": terminal_state or "",
            },
        )
        record_a2a_call(url, status="success", duration_sec=duration_ms / 1000.0)
        return response_text, new_context_id, True
    except Exception as exc:
        duration_ms = int((time.perf_counter() - t0) * 1000)
        current_trace_add_event(
            "a2a.error",
            {
                "endpoint": url,
                "context_id": context_id or "",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
                "duration_ms": duration_ms,
            },
        )
        record_a2a_call(url, status="error", duration_sec=duration_ms / 1000.0)
        return f"调用 A2A 服务失败 ({url}): {exc}", context_id, False
    finally:
        await httpx_client.aclose()
