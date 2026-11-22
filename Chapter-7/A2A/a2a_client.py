"""A2A client helpers for Chapter 7.

The module provides Agent Card discovery and a persistent streaming client
used by ``supervisor.py`` and the Notebook. Every request is required to reach
a terminal A2A task state before returning a result.

Main capabilities:

- Discover the Agent Card published by an A2A service.
- Send messages and collect streamed text.
- Validate that each task reaches a terminal A2A state.

:author: zhanghong, liuhui
:date: 2026-11-22
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from a2a.client import ClientConfig, create_client
from a2a.helpers import get_stream_response_text, new_text_message
from a2a.types import Role, SendMessageRequest, TaskState


_TERMINAL_STATES = {
    TaskState.TASK_STATE_COMPLETED,
    TaskState.TASK_STATE_FAILED,
    TaskState.TASK_STATE_INPUT_REQUIRED,
    TaskState.TASK_STATE_REJECTED,
    TaskState.TASK_STATE_CANCELED,
}
_FAILED_STATES = {
    TaskState.TASK_STATE_FAILED,
    TaskState.TASK_STATE_REJECTED,
    TaskState.TASK_STATE_CANCELED,
}


class A2AClientError(RuntimeError):
    """Raised when an A2A service cannot complete a request."""


@dataclass(frozen=True)
class A2ACallResult:
    """Completed result returned by one A2A request."""

    text: str
    task_id: str | None
    context_id: str | None
    state: int


async def discover_agent_card(base_url: str) -> dict[str, Any]:
    """Fetch and return the Agent Card published by an A2A service."""
    card_url = base_url.rstrip("/") + "/.well-known/agent-card.json"
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(15.0, connect=5.0)
    ) as client:
        response = await client.get(card_url)
        response.raise_for_status()
        payload = response.json()
    if not isinstance(payload, dict):
        raise A2AClientError(f"Invalid Agent Card returned by {card_url}")
    return payload


class A2AClient:
    """Persistent connection to one A2A service."""

    def __init__(self, endpoint: str, *, timeout_seconds: float = 180.0) -> None:
        self.endpoint = endpoint.rstrip("/") + "/"
        self.timeout_seconds = timeout_seconds
        self._httpx_client: httpx.AsyncClient | None = None
        self._client = None

    async def _ensure_client(self):
        if self._client is None:
            self._httpx_client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    timeout=self.timeout_seconds,
                    connect=10.0,
                )
            )
            self._client = await create_client(
                self.endpoint,
                ClientConfig(
                    httpx_client=self._httpx_client,
                    streaming=True,
                ),
            )
        return self._client

    async def call(
        self,
        query: str,
        context_id: str | None = None,
    ) -> A2ACallResult:
        """Send one message and require a terminal A2A task state."""
        try:
            client = await self._ensure_client()
            user_message = new_text_message(
                query,
                context_id=context_id,
                role=Role.ROLE_USER,
            )
            request = SendMessageRequest(message=user_message)

            chunks: list[str] = []
            task_id: str | None = None
            resolved_context_id = context_id or user_message.context_id
            terminal_state: int | None = None
            last_state: int | None = None

            async for event in client.send_message(request):
                text = get_stream_response_text(event)
                if text:
                    chunks.append(text)
                if event.HasField("task"):
                    task_id = event.task.id or task_id
                    resolved_context_id = (
                        event.task.context_id or resolved_context_id
                    )
                if event.HasField("status_update"):
                    task_id = event.status_update.task_id or task_id
                    resolved_context_id = (
                        event.status_update.context_id
                        or resolved_context_id
                    )
                    last_state = event.status_update.status.state
                    if last_state in _TERMINAL_STATES:
                        terminal_state = last_state
                        break

            response_text = "".join(chunks).strip()
            state_name = (
                TaskState.Name(terminal_state)
                if terminal_state is not None
                else (
                    TaskState.Name(last_state)
                    if last_state is not None
                    else "NO_TERMINAL_STATE"
                )
            )
            if terminal_state is None:
                raise A2AClientError(
                    f"{self.endpoint} ended without a terminal task state "
                    f"(last state: {state_name})"
                )
            if terminal_state in _FAILED_STATES:
                raise A2AClientError(
                    f"{self.endpoint} ended in {state_name}: "
                    f"{response_text or 'no response text'}"
                )
            if terminal_state == TaskState.TASK_STATE_INPUT_REQUIRED:
                raise A2AClientError(
                    f"{self.endpoint} requires additional user input: "
                    f"{response_text or 'no question returned'}"
                )
            if terminal_state != TaskState.TASK_STATE_COMPLETED:
                raise A2AClientError(
                    f"{self.endpoint} ended in {state_name}"
                )
            if not response_text:
                raise A2AClientError(
                    f"{self.endpoint} completed without response text"
                )
            return A2ACallResult(
                text=response_text,
                task_id=task_id,
                context_id=resolved_context_id,
                state=terminal_state,
            )
        except A2AClientError:
            raise
        except Exception as exc:
            raise A2AClientError(
                f"Failed to call A2A service {self.endpoint}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    async def close(self) -> None:
        """Close the underlying HTTP connection."""
        if self._httpx_client is not None:
            await self._httpx_client.aclose()
        self._httpx_client = None
        self._client = None
