"""将远程 A2A 智能体包装为 LangGraph 子图，供 create_supervisor handoff 调度。"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import httpx
from a2a.client import ClientConfig, create_client
from a2a.helpers import get_stream_response_text, new_text_message
from a2a.types import Role, SendMessageRequest, TaskState
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, MessagesState, StateGraph

# (handoff 节点名, A2A base URL, 描述) — 增删行即可扩展多个 A2A 智能体
A2A_AGENT_SPECS: List[Tuple[str, str, str]] = [
    ("hotel_agent", "http://127.0.0.1:9012/", "酒店推荐（A2A）"),
    # ("weather_agent", "http://127.0.0.1:9013/", "天气查询（A2A）"),
]

_TERMINAL_STATES = {
    TaskState.TASK_STATE_COMPLETED,
    TaskState.TASK_STATE_FAILED,
    TaskState.TASK_STATE_INPUT_REQUIRED,
    TaskState.TASK_STATE_REJECTED,
    TaskState.TASK_STATE_CANCELED,
}

# thread_id -> agent_name -> a2a context_id
_a2a_context_store: Dict[str, Dict[str, str]] = {}


class A2AClient:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint.rstrip("/") + "/"
        self._httpx_client: httpx.AsyncClient | None = None
        self._client = None

    async def _ensure_client(self):
        if self._client is None:
            self._httpx_client = httpx.AsyncClient(
                timeout=httpx.Timeout(timeout=120.0, connect=10.0)
            )
            self._client = await create_client(
                self.endpoint,
                ClientConfig(httpx_client=self._httpx_client, streaming=True),
            )
        return self._client

    async def call(
        self,
        query: str,
        context_id: str | None = None,
    ) -> tuple[str, str | None]:
        try:
            client = await self._ensure_client()
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
                    new_context_id = (
                        event.status_update.context_id or new_context_id
                    )
                    if event.status_update.status.state in _TERMINAL_STATES:
                        break

            response_text = "".join(chunks).strip()
            if not response_text:
                return "A2A 服务未返回文本内容", new_context_id
            return response_text, new_context_id
        except Exception as exc:
            return f"调用 A2A 服务失败 ({self.endpoint}): {exc}", context_id

    async def close(self):
        if self._httpx_client is not None:
            await self._httpx_client.aclose()
            self._httpx_client = None
            self._client = None


def _extract_query(messages: List[Any]) -> str:
    parts: List[str] = []
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and msg.content:
            parts.append(str(msg.content))
            if len(parts) >= 2:
                break
    return "\n".join(reversed(parts)) if parts else ""


def build_a2a_agent_graph(node_name: str, endpoint: str, description: str):
    """单个 A2A 远程 Agent → 单节点 LangGraph，供 create_supervisor 调度。"""

    async def run_a2a_agent(state: MessagesState, config) -> Dict[str, Any]:
        query = _extract_query(state["messages"])
        thread_id = (config or {}).get("configurable", {}).get("thread_id", "default")
        ctx_bucket = _a2a_context_store.setdefault(thread_id, {})
        context_id = ctx_bucket.get(node_name)

        client = A2AClient(endpoint)
        print(f"  ▶ [{node_name}] 调用 A2A {endpoint} ...", flush=True)
        try:
            content, new_context_id = await client.call(query or "请根据上下文完成任务", context_id)
            if new_context_id:
                ctx_bucket[node_name] = new_context_id
        finally:
            await client.close()
        print(f"  ✓ [{node_name}] A2A 返回 ({len(content)} 字)", flush=True)

        return {
            "messages": [
                AIMessage(
                    content=content,
                    name=node_name,
                    additional_kwargs={"a2a_endpoint": endpoint, "description": description},
                )
            ]
        }

    graph = StateGraph(MessagesState)
    graph.add_node(node_name, run_a2a_agent)
    graph.set_entry_point(node_name)
    graph.add_edge(node_name, END)
    return graph.compile(name=node_name)


def build_all_a2a_agent_graphs() -> List[Any]:
    return [
        build_a2a_agent_graph(node, url, desc)
        for node, url, desc in A2A_AGENT_SPECS
    ]
