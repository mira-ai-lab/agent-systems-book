import logging
import os
import time
import uuid
from datetime import datetime

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    UnsupportedOperationError,
)
from a2a.utils import new_agent_text_message, new_artifact, new_text_artifact
from a2a.utils.errors import ServerError
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from agents.logging_utils import get_executor_logger

from .agent import HotelRecommendationAgent

load_dotenv()
VALID_STATUSES = {"completed", "input-required", "rejected"}


async def get_status(response: str) -> str:
    if not response or not response.strip():
        return "completed"
    prompt = f"""Classify as one word: completed, input-required, or rejected. completed=done; input-required=asking for info; rejected=out of scope (not hotel recommendation).

---
{response[:2000]}
---
One word."""
    try:
        model = ChatOpenAI(
            model=os.getenv("DEPLOYMENT_NAME"),
            openai_api_key=os.getenv("CHAT_API_KEY"),
            openai_api_base=os.getenv("CHAT_ENDPOINT"),
            temperature=0,
        )
        msg = await model.ainvoke([{"role": "user", "content": prompt}])
        text = (msg.content or "").strip().lower()
        for s in VALID_STATUSES:
            if s in text or s.replace("-", " ") in text:
                return s
        return "completed"
    except Exception as e:
        logging.warning("get_status failed: %s", e)
        return "completed"


class MarkLastAsyncIterator:
    def __init__(self, aiterable):
        self.aiterable, self.buffer, self.iterator, self.index = aiterable, [], None, 0

    def __aiter__(self):
        self.iterator = self.aiterable.__aiter__()
        return self

    async def __anext__(self):
        if not self.buffer:
            try:
                self.buffer.append(await self.iterator.__anext__())
            except StopAsyncIteration:
                raise
        try:
            next_item = await self.iterator.__anext__()
            current = self.buffer.pop(0)
            self.buffer.append(next_item)
            result = (self.index, current, False)
            self.index += 1
            return result
        except StopAsyncIteration:
            result = (self.index, self.buffer.pop(0), True)
            self.index += 1
            return result


logging.basicConfig(level=logging.INFO)


class HotelRecommendationAgentExecutor(AgentExecutor):
    def __init__(self):
        self.agent = HotelRecommendationAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        log = get_executor_logger("HOTEL_RECOMMENDATION_AGENT", "HotelRecommendationAgent")
        t0 = time.perf_counter()
        if not context.message:
            raise Exception("No message provided")
        current_task_id = context.message.task_id or str(uuid.uuid4())
        current_context_id = context.message.context_id or str(uuid.uuid4())
        metadata = context._params.metadata or {}

        await event_queue.enqueue_event(
            Task(
                id=current_task_id,
                contextId=current_context_id,
                status=TaskStatus(state=TaskState.submitted, timestamp=datetime.now().isoformat()),
                history=[context.message],
                metadata=context._params.metadata,
            )
        )
        query = context.get_user_input()
        if not query:
            await event_queue.enqueue_event(new_agent_text_message("User input is empty."))
            return

        preview = (query[:200] + "…") if len(query) > 200 else query
        log.info("request_start task_id=%s context_id=%s query_len=%s query_preview=%r", current_task_id, current_context_id, len(query), preview)

        artifact_id = str(uuid.uuid4())
        concatenated = ""
        first_chunk_at: float | None = None
        emitted_chars = 0

        async for index, item, is_last in MarkLastAsyncIterator(self.agent.stream(query, current_context_id)):
            concatenated += item.content
            if item.content:
                emitted_chars += len(item.content)
                if first_chunk_at is None:
                    first_chunk_at = time.perf_counter()
            if not is_last:
                a = new_text_artifact(name="", text=item.content)
                a.artifact_id = artifact_id
                await event_queue.enqueue_event(
                    TaskArtifactUpdateEvent(contextId=current_context_id, taskId=current_task_id, artifact=a, metadata=metadata, lastChunk=False, append=index != 0)
                )
            else:
                a = new_artifact(parts=[], name="")
                if item.content:
                    a = new_text_artifact(name="", text=item.content)
                a.artifact_id = artifact_id
                await event_queue.enqueue_event(
                    TaskArtifactUpdateEvent(contextId=current_context_id, taskId=current_task_id, artifact=a, metadata=metadata, lastChunk=True, append=index != 0)
                )

        status_str = await get_status(concatenated)
        task_state = TaskState.input_required if status_str == "input-required" else (getattr(TaskState, "rejected", TaskState.failed) if status_str == "rejected" else (TaskState.completed if status_str == "completed" else TaskState.failed))
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(contextId=current_context_id, taskId=current_task_id, status=TaskStatus(state=task_state, timestamp=datetime.now().isoformat()), final=True, metadata=metadata)
        )

        total_ms = int((time.perf_counter() - t0) * 1000)
        first_chunk_ms = int((first_chunk_at - t0) * 1000) if first_chunk_at is not None else None
        log.info(
            "request_end task_id=%s context_id=%s state=%s total_ms=%s first_chunk_ms=%s emitted_chars=%s",
            current_task_id,
            current_context_id,
            getattr(task_state, "value", str(task_state)),
            total_ms,
            first_chunk_ms,
            emitted_chars,
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())

