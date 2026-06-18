"""将平台 registry 子 Agent 暴露为 A2A Server Executor。"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from agent_framework.orchestration.supervisor.invoke_traced import _parse_agent_result
from agent_framework.tracing.trace_provider import span_name, trace_span


def _require_a2a_server() -> None:
    try:
        import a2a.server.agent_execution  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "A2A Server 需要安装：pip install 'agent-platform[a2a]'"
        ) from exc


def _proto_timestamp(dt: datetime | None = None):
    from google.protobuf.timestamp_pb2 import Timestamp

    ts = Timestamp()
    ts.FromDatetime(dt or datetime.now(timezone.utc))
    return ts


@trace_span(
    name=span_name("a2a.server.invoke"),
    attrs_args=["factory_name", "query"],
    record_result=False,
)
async def invoke_registry_agent(
    invoke_fn: Callable[[str, str], Any],
    *,
    factory_name: str,
    query: str,
    context_id: str,
) -> str:
    result = await invoke_fn(query, context_id)
    if isinstance(result, str):
        return result
    return _parse_agent_result(result)


class RegistrySubAgentExecutor:
    """A2A AgentExecutor：委托 registry 子 Agent 处理请求。"""

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(
        self,
        *,
        factory_name: str,
        invoke_fn: Callable[[str, str], Any],
        display_name: str = "",
        description: str = "",
    ) -> None:
        _require_a2a_server()
        self.factory_name = factory_name
        self.display_name = display_name or factory_name
        self.description = description or factory_name
        self._invoke_fn = invoke_fn

    async def execute(self, context, event_queue) -> None:
        from a2a.helpers import new_text_artifact
        from a2a.types import (
            Task,
            TaskArtifactUpdateEvent,
            TaskState,
            TaskStatus,
            TaskStatusUpdateEvent,
        )

        if not context.message:
            raise RuntimeError("No message provided")

        task_id = context.message.task_id or context.task_id or str(uuid.uuid4())
        context_id = context.message.context_id or context.context_id or str(uuid.uuid4())
        metadata = context.metadata or {}

        await event_queue.enqueue_event(
            Task(
                id=task_id,
                context_id=context_id,
                status=TaskStatus(
                    state=TaskState.TASK_STATE_SUBMITTED,
                    timestamp=_proto_timestamp(),
                ),
                history=[context.message],
                metadata=metadata,
            )
        )

        query = context.get_user_input()
        if not query:
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    context_id=context_id,
                    task_id=task_id,
                    status=TaskStatus(
                        state=TaskState.TASK_STATE_FAILED,
                        timestamp=_proto_timestamp(),
                    ),
                    metadata=metadata,
                )
            )
            return

        try:
            response_text = await invoke_registry_agent(
                self._invoke_fn,
                factory_name=self.factory_name,
                query=query,
                context_id=context_id,
            )
            artifact_id = str(uuid.uuid4())
            await event_queue.enqueue_event(
                TaskArtifactUpdateEvent(
                    context_id=context_id,
                    task_id=task_id,
                    artifact=new_text_artifact(
                        name="",
                        text=response_text,
                        artifact_id=artifact_id,
                    ),
                    metadata=metadata,
                    last_chunk=True,
                    append=False,
                )
            )
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    context_id=context_id,
                    task_id=task_id,
                    status=TaskStatus(
                        state=TaskState.TASK_STATE_COMPLETED,
                        timestamp=_proto_timestamp(),
                    ),
                    metadata=metadata,
                )
            )
        except Exception:
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    context_id=context_id,
                    task_id=task_id,
                    status=TaskStatus(
                        state=TaskState.TASK_STATE_FAILED,
                        timestamp=_proto_timestamp(),
                    ),
                    metadata=metadata,
                )
            )
            raise

    async def cancel(self, context, event_queue) -> None:
        from a2a.types import UnsupportedOperationError

        raise UnsupportedOperationError()
