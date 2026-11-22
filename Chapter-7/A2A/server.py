"""Publish business agents as A2A services for Chapter 7.

This module loads an Agent Card, maps its ``name`` to a business agent, and
builds the Starlette application that exposes the Agent Card discovery route
and the JSON-RPC task routes. The A2A SDK owns message parsing and HTTP
routing; this module owns the business executor and process startup.

Main capabilities:

- Load an Agent Card and select the matching business agent.
- Execute business agents and publish artifacts and task states.
- Build and run the A2A HTTP service.

:author: zhanghong, liuhui
:date: 2026-11-22
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

import click
import uvicorn
from a2a.helpers import (
    new_task_from_user_message,
    new_text_message,
    new_text_part,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCard, UnsupportedOperationError
from google.protobuf.json_format import ParseDict
from starlette.applications import Starlette

import settings
from agents import (
    HotelRecommendationAgent,
    TravelPlannerAgent,
)


logger = logging.getLogger(__name__)


class GenericAgentExecutor(AgentExecutor):
    """Collect a business-agent response and publish one A2A artifact."""

    def __init__(self, agent: Any, artifact_name: str) -> None:
        self.agent = agent
        self.artifact_name = artifact_name

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        task = context.current_task
        if task is None:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        query = context.get_user_input()
        if not query.strip():
            await updater.requires_input(
                new_text_message(
                    "请提供需要该智能体处理的任务文本。",
                    context_id=task.context_id,
                    task_id=task.id,
                )
            )
            return

        await updater.start_work()
        chunks: list[str] = []
        failed = False
        try:
            async for item in self.agent.stream(query, task.context_id):
                chunks.append(item.content)
                failed = failed or item.is_error
        except Exception as exc:
            chunks.append(f"{type(exc).__name__}: {exc}")
            failed = True

        text = "".join(chunks).strip()
        await updater.add_artifact(
            [new_text_part(text or "Agent execution failed.")],
            artifact_id=str(uuid.uuid4()),
            name=self.artifact_name,
            last_chunk=True,
        )
        if failed:
            await updater.failed()
        else:
            await updater.complete()

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        raise UnsupportedOperationError()


def get_agent(agent_name: str) -> Any:
    """Map a public Agent Card name to its business implementation."""
    agents: dict[str, Any] = {
        "TravelPlannerAgent": TravelPlannerAgent,
        "HotelAgent": HotelRecommendationAgent,
    }
    try:
        return agents[agent_name]()
    except KeyError as exc:
        raise ValueError(
            f"No implementation registered for Agent Card: {agent_name}"
        ) from exc


def load_agent_card(path: Path, public_url: str) -> AgentCard:
    """Load a standard Agent Card JSON file and set its advertised URL."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    interfaces = payload.get("supportedInterfaces") or []
    if not interfaces:
        raise ValueError(f"{path} must define supportedInterfaces")
    interfaces[0]["url"] = public_url.rstrip("/") + "/"
    return ParseDict(payload, AgentCard())


def build_agent_app(
    agent_card: AgentCard,
    agent: Any,
) -> Starlette:
    """Build the HTTP application for one Card and implementation."""
    request_handler = DefaultRequestHandler(
        agent_executor=GenericAgentExecutor(
            agent,
            f"{agent_card.name}-result",
        ),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )
    return Starlette(
        routes=[
            *create_agent_card_routes(agent_card),
            *create_jsonrpc_routes(
                request_handler,
                rpc_url="/",
                enable_v0_3_compat=True,
            ),
        ]
    )


@click.command()
@click.option(
    "--agent-card",
    "agent_card_path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option("--host", default=settings.DEFAULT_HOST, show_default=True)
@click.option("--port", required=True, type=int)
@click.option("--public-url", default=None)
def main(
    agent_card_path: Path,
    host: str,
    port: int,
    public_url: str | None,
) -> None:
    """Start the implementation selected by an Agent Card."""
    logging.basicConfig(level=logging.INFO)
    advertised_url = public_url or settings.advertised_url(host, port)
    card = load_agent_card(agent_card_path, advertised_url)
    logger.info("%s listening on %s", card.name, advertised_url)
    for logger_name in ("httpx", "httpx2", "httpcore", "openai"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    uvicorn.run(
        build_agent_app(card, get_agent(card.name)),
        host=host,
        port=port,
    )


if __name__ == "__main__":
    main()
