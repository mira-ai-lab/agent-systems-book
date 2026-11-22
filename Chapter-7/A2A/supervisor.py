"""Build a from-scratch Supervisor that routes tasks to remote A2A agents.

Main capabilities:

- Define handoff tools that transfer control to a worker agent.
- Build a Supervisor Agent with LangGraph.
- Invoke remote Planner and Hotel agents through A2A and summarize results.

:author: zhanghong, liuhui
:date: 2026-11-22
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any, Awaitable, Callable

import agents  # noqa: F401 - initializes Chapter-5/6 import paths
from a2a_client import A2AClient, discover_agent_card
from a2a_models import PlannerRequest, TravelPlan, capability_from_card
from config import load_project_dotenv
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.types import Command, Send
from llm import create_llm
import settings


load_project_dotenv()

PREPARE_QUERY = Callable[[str], Awaitable[str]]


def create_handoff_tool(
    *,
    agent_name: str,
    description: str | None = None,
):
    """Create a tool that transfers control to one worker agent."""
    name = f"transfer_to_{agent_name}"
    tool_description = description or f"Ask {agent_name} for help."

    @tool(name, description=tool_description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState],
        task_description: Annotated[
            str,
            "Description of what the next agent should do, including context.",
        ],
    ) -> Command:
        task_message = HumanMessage(
            content=task_description,
            name="task",
        )
        agent_input = {
            **state,
            "messages": [task_message],
        }
        return Command(
            goto=[Send(agent_name, agent_input)],
            graph=Command.PARENT,
        )

    return handoff_tool


def _latest_task_text(messages: list[Any]) -> str:
    """Return the latest user task text passed to the remote agent."""
    for message in reversed(messages):
        if isinstance(message, dict):
            content = message.get("content")
            if message.get("role") == "user" and content:
                return str(content).strip()
        elif isinstance(message, HumanMessage) and message.content:
            return str(message.content).strip()

    texts = [
        str(message.content).strip()
        for message in messages
        if isinstance(message, HumanMessage) and message.content
    ]
    return "\n".join(texts[-3:]) or "请根据当前对话完成任务。"


def _parse_travel_plan(text: str) -> TravelPlan:
    """Parse Planner JSON, tolerating an optional code fence."""
    candidates = [text.strip()]
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in dict.fromkeys(candidates):
        try:
            return TravelPlan.model_validate_json(candidate)
        except ValueError:
            continue
    raise ValueError("Planner returned text that is not a valid TravelPlan.")


def _format_planner_result(text: str) -> str:
    """Reduce Planner output to an ordered task list for the Supervisor."""
    plan = _parse_travel_plan(text)
    tasks_by_id = {task.task_id: task for task in plan.tasks}
    ordered_ids = list(dict.fromkeys([*plan.execution_order]))
    ordered_ids.extend(
        task.task_id
        for task in plan.tasks
        if task.task_id not in ordered_ids
    )
    lines = []
    for index, task_id in enumerate(ordered_ids, start=1):
        task = tasks_by_id.get(task_id)
        if task is None:
            continue
        dependencies = (
            "，依赖：" + "、".join(task.depends_on)
            if task.depends_on
            else ""
        )
        lines.append(
            f"{index}. {task.task_id} {task.description}{dependencies}"
        )
    return (
        "Planner 已按依赖关系重排子任务，按执行顺序逐个处理：\n"
        + "\n".join(lines)
    )


async def _prepare_planner_query(query: str) -> str:
    """Package the request and discovered Agent capabilities for Planner."""
    capabilities = []
    for service in settings.TASK_AGENT_SERVICES:
        card = await discover_agent_card(
            settings.service_base_url(service)
        )
        capabilities.append(capability_from_card(card))
    return PlannerRequest(
        query=query,
        available_agents=capabilities,
    ).model_dump_json()


def create_remote_agent_node(
    name: str,
    service: str,
    prepare_query: PREPARE_QUERY | None = None,
):
    """Return an async node that invokes one remote A2A agent."""

    async def invoke_remote(
        state: MessagesState,
        config: RunnableConfig,
    ) -> dict[str, list[AIMessage]]:
        query = _latest_task_text(state["messages"])
        if prepare_query is not None:
            query = await prepare_query(query)
        thread_id = (
            config.get("configurable", {}).get("thread_id")
            or uuid.uuid4().hex
        )
        client = A2AClient(settings.service_base_url(service))
        try:
            result = await client.call(
                query,
                context_id=f"{thread_id}:{name}",
            )
            text = result.text
            if name == "planner_agent":
                text = _format_planner_result(text)
        except Exception as exc:
            text = f"A2A call failed: {type(exc).__name__}: {exc}"
        finally:
            await client.close()
        return {
            "messages": [
                AIMessage(
                    content=text,
                    name=name,
                )
            ]
        }

    return invoke_remote


def build_supervisor():
    """Build a from-scratch Supervisor graph over the remote A2A agents."""
    llm = create_llm()
    transfer_to_planner = create_handoff_tool(
        agent_name="planner_agent",
        description=(
            "把旅行规划请求交给 Planner agent。Planner 会返回已按依赖关系"
            "重排、并给出执行顺序的酒店推荐子任务列表。"
        ),
    )
    transfer_to_hotel = create_handoff_tool(
        agent_name="hotel_agent",
        description=(
            "把单个城市的酒店推荐子任务交给 Hotel agent。Hotel 会返回"
            "该城市符合预算与位置偏好的酒店列表。"
        ),
    )
    handoff_tools = [transfer_to_planner, transfer_to_hotel]

    supervisor_agent = create_react_agent(
        model=llm,
        tools=handoff_tools,
        prompt=(
            "你是旅行任务 Supervisor，负责协调 planner_agent 与 "
            "hotel_agent。planner_agent 会返回按执行顺序排列的"
            "子任务列表，hotel_agent 负责处理单个城市的"
            "酒店推荐。请根据对话内容合理调度这两个 Agent。"
        ),
        name="travel_supervisor",
    )

    planner_node = create_remote_agent_node(
        "planner_agent",
        "planner",
        _prepare_planner_query,
    )
    hotel_node = create_remote_agent_node(
        "hotel_agent",
        "hotel",
    )

    graph = StateGraph(MessagesState)
    graph.add_node(
        supervisor_agent,
        destinations=("planner_agent", "hotel_agent", END),
    )
    graph.add_node("planner_agent", planner_node)
    graph.add_node("hotel_agent", hotel_node)
    graph.add_edge(START, "travel_supervisor")
    graph.add_edge("planner_agent", "travel_supervisor")
    graph.add_edge("hotel_agent", "travel_supervisor")
    return graph.compile(checkpointer=InMemorySaver())
