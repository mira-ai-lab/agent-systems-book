"""Supervisor 编排后端：动态 handoff 调度子 Agent。"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from agent_framework.config import (
    CHROMA_DIR,
    REQUEST_SLOT_WAIT_SEC,
    REQUEST_TIMEOUT_SEC,
    create_llm,
    load_project_dotenv,
)
from agent_framework.domain.a2a_spec import A2AEndpoint
from agent_framework.domain.agent_factory import SubAgentFactory
from agent_framework.domain.agent_registry import SubAgentRegistry
from agent_framework.domain.domain_prompts import DomainPrompts
from agent_framework.domain.pipeline import PipelineConfig
from agent_framework.infra.checkpoint_factory import resolve_checkpointer
from agent_framework.infra.concurrency import acquire_request_slot
from agent_framework.infra.memory.aggregation_helpers import (
    direct_response_from_results,
    is_single_direct_response,
)
from agent_framework.infra.memory.memory_factory import create_long_term_memory, resolve_memory_backend
from agent_framework.i18n.agent_locale_context import agent_locale_context
from agent_framework.observability.metrics import record_handoff
from agent_framework.observability.request_context import request_metrics_context
from agent_framework.orchestration.protocol import MODE_SUPERVISOR, TRANSPORT_LOCAL, AgentTransport
from agent_framework.orchestration.supervisor.graph import (
    build_default_supervisor_prompt,
    build_supervisor_app,
    resolve_supervisor_subgraphs,
)
from agent_framework.orchestration.thread_stage_context import get_thread_stage_store
from agent_framework.orchestration.supervisor.stage_summary import StageSummarizer
from agent_framework.orchestration.supervisor.step_summary import StepSummarizer
from agent_framework.stream.events import final_event, handoff_event
from agent_framework.tracing import get_logger, get_current_span_context, log_info, setup_observability, trace_span
from agent_framework.tracing.trace_provider import current_trace_add_event, span_name

logger = get_logger(__name__)


def _last_ai_content(messages: List[Any]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            return str(msg.content).strip()
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content).strip()
    return ""


def _extract_subtask_results(messages: List[Any], handoff_node_names: set[str]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    idx = 0
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.name in handoff_node_names:
            idx += 1
            tid = f"T{idx}"
            results[tid] = {
                "task_id": tid,
                "agent": msg.name,
                "status": "completed",
                "agent_summary": str(msg.content or ""),
            }
    return results


def _collect_messages_from_chunk(chunk: Any) -> List[Any]:
    if not isinstance(chunk, dict):
        return []
    messages: List[Any] = []
    for value in chunk.values():
        if isinstance(value, dict):
            block = value.get("messages")
            if isinstance(block, list):
                messages.extend(block)
    return messages


def _detect_new_handoffs(
    messages: List[Any],
    handoff_node_names: set[str],
    a2a_node_names: set[str],
    seen: set[tuple[str, str]],
) -> List[tuple[str, str, str]]:
    found: List[tuple[str, str, str]] = []
    for msg in messages:
        if not isinstance(msg, AIMessage) or msg.name not in handoff_node_names:
            continue
        preview = str(msg.content or "")
        key = (msg.name, preview[:120])
        if key in seen:
            continue
        seen.add(key)
        transport = "a2a" if msg.name in a2a_node_names else "local"
        found.append((msg.name, transport, preview))
    return found


class SupervisorOrchestrator:
    """LangGraph Supervisor 模式编排器（实现 OrchestrationBackend 契约）。"""

    mode = MODE_SUPERVISOR

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        *,
        domain: Optional[str] = None,
        user_id: str = "default",
        registry: SubAgentRegistry,
        prompts: DomainPrompts,
        pipeline: Optional[PipelineConfig] = None,
        enable_memory: bool = True,
        long_term_backend: Optional[Union[str, Any]] = None,
        transport: AgentTransport = TRANSPORT_LOCAL,
        a2a_endpoints: tuple[A2AEndpoint, ...] = (),
        locale: str = "zh",
    ) -> None:
        load_project_dotenv()
        setup_observability()
        self.domain = (domain or "").strip() or None
        self.user_id = (user_id or "default").strip() or "default"
        self.llm = llm or create_llm()
        self.request_timeout_sec = REQUEST_TIMEOUT_SEC
        self.registry = registry
        self.transport = transport
        self.a2a_endpoints = a2a_endpoints
        self.locale = (locale or "zh").strip() or "zh"
        SubAgentFactory.use_registry(registry)
        self.prompts = prompts
        self.pipeline = pipeline or PipelineConfig(enable_pre_survey=False, enable_memory=enable_memory)
        self.enable_memory = self.pipeline.enable_memory
        self.long_term_backend = resolve_memory_backend(long_term_backend)
        self.memory_system: Optional[Any] = None
        self.langgraph_store: Optional[Any] = None

        if self.enable_memory:
            try:
                self.memory_system, self.langgraph_store = create_long_term_memory(
                    self.long_term_backend,
                    user_id=self.user_id,
                    llm=self.llm,
                    persist_directory=str(CHROMA_DIR),
                )
                log_info(logger, "memory.enabled", mode=self.mode)
            except Exception as exc:
                log_info(logger, "memory.init_failed", error=str(exc))

        _, handoff_meta = resolve_supervisor_subgraphs(
            registry,
            transport=transport,
            a2a_endpoints=a2a_endpoints,
        )
        self._handoff_node_names = {node for node, _ in handoff_meta}
        self._a2a_node_names = {
            ep.node_name for ep in a2a_endpoints if ep.is_configured()
        }

        supervisor_prompt = (prompts.supervisor_system or "").strip()
        if not supervisor_prompt:
            supervisor_prompt = build_default_supervisor_prompt(registry, handoff_meta=handoff_meta)

        self.app = build_supervisor_app(
            self.llm,
            registry,
            supervisor_prompt=supervisor_prompt,
            transport=transport,
            a2a_endpoints=a2a_endpoints,
            checkpointer=resolve_checkpointer(),
            store=self.langgraph_store,
        )
        self._step_summarizer: Optional[StepSummarizer] = None
        if self.pipeline.enable_step_summary:
            self._step_summarizer = StepSummarizer(
                self.llm,
                locale=self.locale,
                min_chars=self.pipeline.step_summary_min_chars,
            )
        self._stage_summarizer: Optional[StageSummarizer] = None
        if self.pipeline.enable_stage_summary:
            self._stage_summarizer = StageSummarizer(
                self.llm,
                locale=self.locale,
                min_steps=self.pipeline.stage_summary_min_steps,
            )

    def _build_initial_message(self, user_query: str, thread_id: str) -> str:
        parts = [user_query.strip()]
        if (
            self.pipeline.enable_thread_stage_context
            and self.domain
        ):
            last_stage = get_thread_stage_store().get_last_stage_summary(
                self.domain,
                thread_id,
            )
            if last_stage:
                parts.append(f"\n【先前阶段累计进度】\n{last_stage}")
        if self.memory_system and self.enable_memory:
            hits = self.memory_system.search_memories(user_query)
            memories = self.memory_system.format_memories_for_plan(hits)
            if memories:
                parts.append("\n【相关长期记忆】")
                for m in memories:
                    line = m.get("summary") or m.get("content") or str(m)
                    parts.append(f"- {line}")
            stm = self.memory_system.short_term.format_recent(thread_id, last_n=4)
            if stm and stm != "（无历史对话）":
                parts.append(f"\n【本会话近期对话】\n{stm}")
        return "\n".join(parts)

    async def _finalize_supervisor_result(
        self,
        messages: List[Any],
        user_query: str,
        thread_id: str,
        last_stage_summary: str,
    ) -> Dict[str, Any]:
        subtask_results = _extract_subtask_results(messages, self._handoff_node_names)
        stage_summary = ""
        if self._step_summarizer and subtask_results:
            subtask_results = await self._step_summarizer.summarize_subtask_results(
                messages,
                subtask_results,
                self._handoff_node_names,
                user_query=user_query,
            )
        if self._stage_summarizer and subtask_results:
            stage_summary = await self._stage_summarizer.summarize_stage(
                user_query=user_query,
                messages=messages,
                subtask_results=subtask_results,
                handoff_node_names=self._handoff_node_names,
                last_stage_summary=last_stage_summary,
            )
        if (
            stage_summary
            and self.pipeline.enable_thread_stage_context
            and self.domain
        ):
            get_thread_stage_store().set_last_stage_summary(
                self.domain,
                thread_id,
                stage_summary,
            )
        for _tid, info in subtask_results.items():
            target = str(info.get("agent") or "")
            handoff_transport = "a2a" if target in self._a2a_node_names else "local"
            current_trace_add_event(
                "handoff.completed",
                {
                    "target": target,
                    "transport": handoff_transport,
                    "status": info.get("status", ""),
                    "response_preview": str(info.get("agent_summary") or "")[:200],
                },
            )
            record_handoff(target, handoff_transport, domain=self.domain)
        final_response = _last_ai_content(messages)
        if is_single_direct_response(subtask_results):
            direct = direct_response_from_results(subtask_results)
            if direct:
                final_response = direct
        if self.memory_system and self.enable_memory:
            self.memory_system.record_turn(thread_id, user_query, final_response)
            await self.memory_system.ingest(
                f"用户请求: {user_query.strip()}\n偏好摘要: {final_response[:500]}",
                memory_type="preference",
            )
        trace_id, span_id = get_current_span_context()
        return {
            "final_response": final_response,
            "subtask_results": subtask_results,
            "stage_summary": stage_summary,
            "last_stage_summary": last_stage_summary,
            "execution_plan": None,
            "logs": [],
            "trace_id": trace_id,
            "span_id": span_id,
            "orchestration_mode": self.mode,
            "agent_transport": self.transport,
        }

    @trace_span(name=span_name("request"), attrs_args=["user_query", "thread_id"])
    async def process_request(
        self,
        user_query: str,
        thread_id: str = "default",
        timeout_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        log_info(
            logger,
            "request.start",
            mode=self.mode,
            transport=self.transport,
            thread_id=thread_id,
            query_preview=user_query.strip()[:120],
        )
        from opentelemetry import trace

        root_span = trace.get_current_span()
        if root_span.get_span_context().is_valid:
            root_span.set_attribute("orchestration.mode", self.mode)
            root_span.set_attribute("agent.transport", self.transport)
            root_span.set_attribute("handoff.target_count", len(self._handoff_node_names))
        enriched = self._build_initial_message(user_query, thread_id)
        config = {"configurable": {"thread_id": thread_id}}
        deadline = timeout_sec if timeout_sec is not None else self.request_timeout_sec
        last_stage_summary = ""
        if self.pipeline.enable_thread_stage_context and self.domain:
            last_stage_summary = get_thread_stage_store().get_last_stage_summary(
                self.domain,
                thread_id,
            )

        with agent_locale_context(self.locale):
            with request_metrics_context(
                domain=self.domain or "",
                mode=self.mode,
                transport=self.transport,
            ):
                async with acquire_request_slot(wait_timeout_sec=REQUEST_SLOT_WAIT_SEC):
                    result = await asyncio.wait_for(
                        self.app.ainvoke(
                            {"messages": [HumanMessage(content=enriched)]},
                            config,
                        ),
                        timeout=deadline,
                    )

                messages = result.get("messages", [])
                payload = await self._finalize_supervisor_result(
                    messages,
                    user_query,
                    thread_id,
                    last_stage_summary,
                )

        log_info(
            logger,
            "request.done",
            mode=self.mode,
            transport=self.transport,
            thread_id=thread_id,
            subtask_count=len(payload.get("subtask_results") or {}),
        )
        return payload

    @trace_span(
        name=span_name("request.stream"),
        attrs_args=["user_query", "thread_id"],
        record_result=False,
    )
    async def iter_request_stream(
        self,
        user_query: str,
        thread_id: str = "default",
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        log_info(
            logger,
            "request.start",
            mode=self.mode,
            transport=self.transport,
            thread_id=thread_id,
            query_preview=user_query.strip()[:120],
            stream=True,
        )
        from opentelemetry import trace

        root_span = trace.get_current_span()
        if root_span.get_span_context().is_valid:
            root_span.set_attribute("orchestration.mode", self.mode)
            root_span.set_attribute("agent.transport", self.transport)
            root_span.set_attribute("handoff.target_count", len(self._handoff_node_names))
        enriched = self._build_initial_message(user_query, thread_id)
        config = {"configurable": {"thread_id": thread_id}}
        deadline = kwargs.get("timeout_sec")
        if deadline is None:
            deadline = self.request_timeout_sec
        last_stage_summary = ""
        if self.pipeline.enable_thread_stage_context and self.domain:
            last_stage_summary = get_thread_stage_store().get_last_stage_summary(
                self.domain,
                thread_id,
            )
        seen_handoffs: set[tuple[str, str]] = set()
        messages: List[Any] = []

        with agent_locale_context(self.locale):
            with request_metrics_context(
                domain=self.domain or "",
                mode=self.mode,
                transport=self.transport,
            ):
                async with acquire_request_slot(wait_timeout_sec=REQUEST_SLOT_WAIT_SEC):
                    async for chunk in self.app.astream(
                        {"messages": [HumanMessage(content=enriched)]},
                        config,
                        stream_mode="updates",
                    ):
                        chunk_messages = _collect_messages_from_chunk(chunk)
                        if chunk_messages:
                            messages = chunk_messages
                        for target, transport, preview in _detect_new_handoffs(
                            chunk_messages,
                            self._handoff_node_names,
                            self._a2a_node_names,
                            seen_handoffs,
                        ):
                            yield handoff_event(
                                target=target,
                                transport=transport,
                                preview=preview,
                            )
                    snapshot = await self.app.aget_state(config)
                    if snapshot and snapshot.values:
                        messages = list(snapshot.values.get("messages") or messages)
                    payload = await asyncio.wait_for(
                        self._finalize_supervisor_result(
                            messages,
                            user_query,
                            thread_id,
                            last_stage_summary,
                        ),
                        timeout=deadline,
                    )

        log_info(
            logger,
            "request.done",
            mode=self.mode,
            transport=self.transport,
            thread_id=thread_id,
            subtask_count=len(payload.get("subtask_results") or {}),
            stream=True,
        )
        yield final_event(payload)
