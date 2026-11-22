"""LangGraph orchestration pipeline nodes (fixed-graph mode).

Each node maps to an earlier book chapter: pre_survey (Ch2),
retrieve_memory / save_memory (Ch3), build_plan (Ch4), route (Ch6), execute_layer
(Ch5+ A2A domain agents), and aggregate.

:author: zhanghong, liuhui, xingrui
:date: 2026-11-22
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from opentelemetry import trace

from travel_multi_agent.agents.registry import DomainAgentRegistry
from travel_multi_agent.a2a.client import A2ARegistry
from travel_multi_agent.orchestration.pipeline import PipelineConfig
from travel_multi_agent.planning import TaskPlanner
from travel_multi_agent.orchestration.aggregation_helpers import (
    compose_final_response,
    is_single_direct_response,
)
from travel_multi_agent.memory import LongTermMemory
from travel_multi_agent.tracing import (
    current_trace_add_event,
    get_logger,
    log_info,
    record_exception,
    trace_span,
)
from travel_multi_agent.tracing.trace_provider import span_name

from agent_routing import AgentRouter
from domain_agents import build_domain_agent_user_message

from .state import CentralAgentState
from .stream_sink import StreamSink

logger = get_logger(__name__)


def _streaming(state: CentralAgentState) -> bool:
    """Return whether streaming output is enabled."""
    return bool(state.get("enable_stream"))


def _append_log(state: CentralAgentState, message: str, *, trace: bool = True) -> List[str]:
    """Append one human-readable log line to state.logs and, optionally, tracing."""
    logs = list(state.get("logs") or [])
    logs.append(message)
    if trace and not _streaming(state):
        log_info(logger, message.replace("\n", " "), thread_id=state.get("thread_id"))
    return logs


def _topological_layers(subtasks: List[Dict[str, Any]]) -> List[List[str]]:
    """Build dependency-safe topological layers from task ``depends_on``."""
    by_id = {task["task_id"]: task for task in subtasks}
    order = [task["task_id"] for task in subtasks]
    done: set = set()
    layers: List[List[str]] = []
    remaining = list(order)
    while remaining:
        layer = [
            tid for tid in remaining
            if all(d in done for d in by_id[tid].get("depends_on", []))
        ]
        if not layer:
            layer = [remaining[0]]
        layers.append(layer)
        for tid in layer:
            done.add(tid)
            remaining.remove(tid)
    return layers


def _format_result(result: Any) -> str:
    """Format a subtask result as JSON, falling back to str()."""
    try:
        return json.dumps(result, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(result)


def _log_subtask_start(
    state: CentralAgentState,
    logs: List[str],
    task: Dict[str, Any],
) -> List[str]:
    """Log the start of one subtask."""
    task_id = task["task_id"]
    agent_name = task.get("agent") or ""
    description = (task.get("description") or "")[:60]
    log_info(
        logger,
        "subtask.start",
        task_id=task_id,
        agent=agent_name,
        description=description,
        thread_id=state.get("thread_id"),
    )
    return _append_log(
        {**state, "logs": logs},
        f"  🔄 {task_id} → {agent_name}: {description}{'...' if len(task.get('description') or '') > 60 else ''}",
    )


def _log_subtask_done(
    state: CentralAgentState,
    logs: List[str],
    result: Dict[str, Any],
) -> List[str]:
    """Log the completion of one subtask."""
    agent_name = result.get("agent", "")
    status = result.get("status", "")
    if _streaming(state):
        logs = _append_log(
            {**state, "logs": logs},
            f"     ✓ {result.get('task_id')} {agent_name} {status}",
        )
    else:
        logs = _append_log({**state, "logs": logs}, f"     ✓ {agent_name} 执行完成")
        logs = _append_log({**state, "logs": logs}, "     " + "-" * 56, trace=False)
        for line in _format_result(result).splitlines():
            logs = _append_log({**state, "logs": logs}, f"     {line}", trace=False)
        logs = _append_log({**state, "logs": logs}, "     " + "-" * 56, trace=False)
    if not _streaming(state):
        preview = _format_result(result).replace("\n", " ")
        if len(preview) > 500:
            preview = preview[:500] + "..."
        log_info(
            logger,
            "subtask.done",
            task_id=result.get("task_id"),
            agent=agent_name,
            status=status,
            result_preview=preview,
            thread_id=state.get("thread_id"),
        )
    return logs


def _print_agent_message(
    *,
    sender: str,
    receiver: str,
    task_id: str,
    content: str,
) -> None:
    """Print one natural-language message exchanged with a domain-agent to stderr."""
    text = (content or "").strip() or "（无自然语言内容）"
    message = (
        f"\n{'=' * 72}\n"
        f"[多智能体自然语言协作] {task_id}\n"
        f"发送方：{sender}\n"
        f"接收方：{receiver}\n"
        f"消息内容：\n{text}\n"
        f"{'=' * 72}"
    )
    print(message, file=sys.stderr, flush=True)


@trace_span(
    name=span_name("agent.invoke"),
    attrs_args=["task", "thread_id"],
    parent_arg="trace_parent",
)
async def _invoke_domain_agent(
    task: Dict[str, Any],
    prior_results: Dict[str, Any],
    thread_id: str,
    trace_parent: Optional[Any] = None,
    a2a_registry: Optional[A2ARegistry] = None,
) -> Dict[str, Any]:
    """Invoke one domain-agent through its A2A service."""
    task_id = task["task_id"]
    agent_name = task.get("agent")
    routing_error = task.get("routing_error")

    if not agent_name or routing_error:
        error_msg = routing_error or "routing_failed"
        current_trace_add_event(
            "domain_agent_conversation",
            {
                "query": (task.get("description") or "")[:500],
                "agent": agent_name or "",
                "response": "",
                "status": "failed",
                "routing_error": error_msg,
            },
        )
        return {
            "task_id": task_id,
            "agent": agent_name or "",
            "status": "failed",
            "routing_error": error_msg,
            "agent_summary": "",
        }

    query_text = build_domain_agent_user_message(task, prior_results)
    _print_agent_message(
        sender="Supervisor",
        receiver=agent_name,
        task_id=task_id,
        content=query_text,
    )

    try:
        client = a2a_registry.client_for(agent_name)
        result = await client.call(query_text, context_id=f"{thread_id}_{task_id}")
        text = (result.text or "").strip()
        _print_agent_message(
            sender=agent_name,
            receiver="Supervisor",
            task_id=task_id,
            content=text,
        )
        current_trace_add_event(
            "domain_agent_conversation",
            {
                "query": query_text[:500],
                "agent": agent_name,
                "response": text[:500],
                "status": "completed",
            },
        )
        return {
            "task_id": task_id,
            "agent": agent_name,
            "status": "completed",
            "agent_summary": text,
        }
    except Exception as exc:
        record_exception(
            exc,
            step=span_name("agent.invoke"),
            agent_name=agent_name,
            task_id=task_id,
            thread_id=thread_id,
        )
        current_trace_add_event(
            "domain_agent_conversation",
            {
                "query": query_text[:500],
                "agent": agent_name,
                "response": "",
                "status": "failed",
            },
        )
        return {
            "task_id": task_id,
            "agent": agent_name,
            "status": "failed",
            "agent_summary": "",
            "error": str(exc),
        }


class GraphContext:
    """Shared dependencies (LLM, planner, router, registry, memory, sink) for nodes."""

    def __init__(
        self,
        llm: ChatOpenAI,
        memory_system: LongTermMemory | None = None,
        stream_sink: StreamSink | None = None,
        registry: DomainAgentRegistry | None = None,
        a2a_registry: A2ARegistry | None = None,
        pipeline: PipelineConfig | None = None,
    ) -> None:
        from travel_multi_agent.agents.registry import create_travel_registry

        self.llm = llm
        self.registry = registry or create_travel_registry()
        self.a2a_registry = a2a_registry
        self.pipeline = pipeline or PipelineConfig()
        self.planner = TaskPlanner(llm, self.registry)
        self.router = AgentRouter(llm, self.registry)
        self.memory_system = memory_system
        self.stream_sink = stream_sink or StreamSink()


def make_nodes(ctx: GraphContext):
    """Build and return the LangGraph node callables bound to ctx."""
    # --- Ch2 fact survey ---
    @trace_span(
        name=span_name("orchestration.pre_survey"),
        attrs_args=["state"],
        record_result=False,
    )
    async def pre_survey_node(state: CentralAgentState) -> Dict[str, Any]:
        """Ch2: run the fact survey and write state.pre_survey."""
        if _streaming(state):
            ctx.stream_sink.emit_progress("\n🔍 [Ch2] 预调查进行中...")
        logs = _append_log(state, "\n🔍 [Ch2] 预调查进行中...")
        pre_survey = await ctx.planner.run_pre_survey(state["user_query"])
        summary = {k: v for k, v in pre_survey.items() if k != "raw_text"}
        if _streaming(state):
            ctx.stream_sink.emit_progress("✓ 预调查完成")
            logs = _append_log({**state, "logs": logs}, "✓ 预调查完成")
        else:
            logs = _append_log(
                {**state, "logs": logs},
                "✓ 预调查完成\n" + json.dumps(summary, ensure_ascii=False, indent=2),
            )
        return {"pre_survey": pre_survey, "logs": logs}

    # --- Ch3 long-term memory retrieval ---
    @trace_span(
        name=span_name("orchestration.retrieve_memory"),
        attrs_args=["state"],
        record_result=False,
    )
    async def retrieve_memory_node(state: CentralAgentState) -> Dict[str, Any]:
        """Ch3: retrieve long-term memories and write state.retrieved_memories."""
        logs = list(state.get("logs") or [])
        memories: List[Dict[str, Any]] = []
        if ctx.memory_system and state.get("enable_memory", True):
            hits = ctx.memory_system.search_memories(state["user_query"])
            memories = ctx.memory_system.format_memories_for_plan(hits)
            logs = _append_log(
                {**state, "logs": logs},
                f"\n🧠 [Ch3] 检索到 {len(memories)} 条相关记忆",
            )
            if memories:
                logs = _append_log(
                    {**state, "logs": logs},
                    json.dumps(memories, ensure_ascii=False, indent=2),
                )
            else:
                logs = _append_log(
                    {**state, "logs": logs},
                    "  （暂无历史记忆，将仅使用当前对话信息）",
                )
        else:
            reason = "未启用" if not state.get("enable_memory", True) else "初始化失败"
            logs = _append_log({**state, "logs": logs}, f"\n🧠 [Ch3] 记忆已跳过（{reason}）")
        current_trace_add_event("memory.retrieved", {"memory.count": len(memories)})
        return {"retrieved_memories": memories, "logs": logs}

    # --- Ch4 task decomposition ---
    @trace_span(
        name=span_name("orchestration.build_plan"),
        attrs_args=["state"],
        record_result=False,
    )
    async def build_plan_node(state: CentralAgentState) -> Dict[str, Any]:
        """Ch4: build the dependency-safe task plan (no agent assignment)."""
        if _streaming(state):
            ctx.stream_sink.emit_progress("\n📋 [Ch4] 任务拆解 → 依赖分析...")
        logs = _append_log(state, "\n📋 [Ch4] 任务拆解 → 依赖分析...")
        plan = await ctx.planner.build_execution_plan(
            state["user_query"],
            state.get("pre_survey") or {},
            state.get("retrieved_memories") or [],
        )
        tasks = plan["tasks"]
        execution_order = plan["execution_order"]
        layers = _topological_layers(tasks)
        summary = f"✓ 共 {len(tasks)} 个子任务，{len(layers)} 个执行层"
        order_line = f"  执行顺序: {' → '.join(execution_order)}"
        if _streaming(state):
            ctx.stream_sink.emit_progress(summary)
            ctx.stream_sink.emit_progress(order_line)
        logs = _append_log({**state, "logs": logs}, summary)
        logs = _append_log({**state, "logs": logs}, order_line)
        log_info(
            logger,
            "plan.built",
            subtask_count=len(tasks),
            layer_count=len(layers),
            thread_id=state.get("thread_id"),
        )
        current_trace_add_event(
            "plan.built",
            {
                "subtask.count": len(tasks),
                "layer.count": len(layers),
                "execution.order": ",".join(execution_order),
            },
        )
        return {
            "execution_plan": plan,
            "total_goal": plan.get("total_goal", ""),
            "tasks": tasks,
            "execution_order": execution_order,
            "logs": logs,
        }

    # --- Ch6 domain-agent routing ---
    @trace_span(
        name=span_name("orchestration.route"),
        attrs_args=["state"],
        record_result=False,
    )
    async def route_node(state: CentralAgentState) -> Dict[str, Any]:
        """Ch6: route every task to one domain-agent (separate routing step)."""
        if _streaming(state):
            ctx.stream_sink.emit_progress("\n🔀 [Ch6] 领域智能体路由...")
        logs = _append_log(state, "\n🔀 [Ch6] 领域智能体路由...")
        tasks = state.get("tasks") or []
        execution_order = state.get("execution_order") or []
        pre_survey = state.get("pre_survey") or {}
        depends_map = {
            task["task_id"]: list(task.get("depends_on") or [])
            for task in tasks
        }
        subtasks = await ctx.router.route_to_agents(
            tasks,
            execution_order,
            depends_map,
            pre_survey,
        )
        layers = _topological_layers(subtasks)
        routed_line = " → ".join(
            f"{s.get('task_id')}:{s.get('agent') or '??'}" for s in subtasks
        )
        logs = _append_log({**state, "logs": logs}, f"  路由结果: {routed_line}")
        log_info(
            logger,
            "plan.routed",
            agent_count=sum(1 for s in subtasks if s.get("agent")),
            routing_error_count=sum(1 for s in subtasks if s.get("routing_error")),
            thread_id=state.get("thread_id"),
        )
        return {
            "subtasks": subtasks,
            "pending_layers": layers,
            "current_layer_index": 0,
            "subtask_results": {},
            "logs": logs,
        }

    # --- Ch5+ layered A2A execution ---
    @trace_span(
        name=span_name("orchestration.execute_layer"),
        attrs_args=["state"],
        record_result=False,
    )
    async def execute_layer_node(state: CentralAgentState) -> Dict[str, Any]:
        """execute one dependency layer in parallel via A2A domain-agents."""
        layers = state.get("pending_layers") or []
        idx = state.get("current_layer_index", 0)
        subtasks = {t["task_id"]: t for t in state.get("subtasks") or []}
        results = dict(state.get("subtask_results") or {})
        thread_id = state.get("thread_id", "default")
        logs = list(state.get("logs") or [])

        if idx >= len(layers):
            return {}

        layer = layers[idx]
        layer_span = trace.get_current_span()
        layer_span.set_attribute("layer.index", idx + 1)
        layer_span.set_attribute("layer.tasks", ",".join(layer))
        layer_span.set_attribute("thread.id", thread_id)

        layer_msg = f"\n执行第 {idx + 1}/{len(layers)} 层: {layer}"
        if _streaming(state):
            ctx.stream_sink.emit_progress(layer_msg)
        logs = _append_log({**state, "logs": logs}, layer_msg)

        tasks = [subtasks[tid] for tid in layer]
        for task in tasks:
            if _streaming(state):
                tid = task["task_id"]
                agent_name = task.get("agent") or ""
                ctx.stream_sink.emit_progress(f"  🔄 {tid} → {agent_name}")
            logs = _log_subtask_start({**state, "logs": logs}, logs, task)

        if len(tasks) == 1:
            layer_results = [
                await _invoke_domain_agent(
                    tasks[0],
                    results,
                    thread_id,
                    trace_parent=layer_span,
                    a2a_registry=ctx.a2a_registry,
                )
            ]
        else:
            layer_results = list(await asyncio.gather(*[
                _invoke_domain_agent(
                    t,
                    results,
                    thread_id,
                    trace_parent=layer_span,
                    a2a_registry=ctx.a2a_registry,
                )
                for t in tasks
            ]))

        failed = [r for r in layer_results if r.get("status") == "failed"]
        if failed:
            log_info(
                logger,
                "layer.partial_failure",
                layer_index=idx + 1,
                failed_tasks=",".join(r["task_id"] for r in failed),
                thread_id=thread_id,
            )
            current_trace_add_event(
                "layer.partial_failure",
                {
                    "layer.index": idx + 1,
                    "failed_tasks": ",".join(r["task_id"] for r in failed),
                },
            )

        for res in layer_results:
            results[res["task_id"]] = res
            logs = _log_subtask_done({**state, "logs": logs}, logs, res)

        logs = _append_log(
            {**state, "logs": logs},
            f"  ✓ 第 {idx + 1} 层全部完成（{len(layer_results)}/{len(layer)} 个子任务）",
        )

        return {
            "subtask_results": results,
            "current_layer_index": idx + 1,
            "logs": logs,
        }

    # --- aggregation ---
    @trace_span(
        name=span_name("orchestration.aggregate"),
        attrs_args=["state"],
        record_result=False,
    )
    async def aggregate_node(state: CentralAgentState) -> Dict[str, Any]:
        """Select the final response from subtask results (no LLM aggregation)."""
        if _streaming(state):
            ctx.stream_sink.emit_progress("\n📝 汇总结果...")
        logs = _append_log(state, "\n📝 汇总结果...")
        results = state.get("subtask_results") or {}
        execution_order = state.get("execution_order") or []
        if is_single_direct_response(results):
            title = "📋 最终回复"
        else:
            title = "📋 最终旅行规划"
        final_text = compose_final_response(results, execution_order)

        logs = _append_log({**state, "logs": logs}, "\n" + "=" * 80)
        logs = _append_log({**state, "logs": logs}, title)
        logs = _append_log({**state, "logs": logs}, "=" * 80)
        logs = _append_log({**state, "logs": logs}, final_text, trace=not _streaming(state))
        logs = _append_log({**state, "logs": logs}, "=" * 80)
        trace.get_current_span().set_attribute("final_response.length", len(final_text or ""))
        return {"final_response": final_text, "logs": logs}

    @trace_span(
        name=span_name("orchestration.save_memory"),
        attrs_args=["state"],
        record_result=False,
    )
    async def save_memory_node(state: CentralAgentState) -> Dict[str, Any]:
        """Ch3: write the turn and preference summary back to long-term memory."""
        logs = list(state.get("logs") or [])
        if ctx.memory_system and state.get("enable_memory", True):
            thread_id = state.get("thread_id", "default")
            ctx.memory_system.record_turn(
                thread_id, state["user_query"], state.get("final_response", "")
            )
            await ctx.memory_system.ingest(
                f"用户请求: {state['user_query'].strip()}\n"
                f"偏好摘要: {(state.get('final_response') or '')[:500]}",
                memory_type="preference",
            )
            logs = _append_log(
                {**state, "logs": logs},
                "\n💾 [Ch3] 已写入长期记忆（对话轮次 + 偏好摘要）",
            )
            current_trace_add_event("memory.saved", {"memory.type": "preference"})
        else:
            reason = "未启用" if not state.get("enable_memory", True) else "初始化失败"
            logs = _append_log(
                {**state, "logs": logs},
                f"\n💾 [Ch3] 记忆写入已跳过（{reason}）",
            )
        return {"logs": logs}

    return {
        "pre_survey": pre_survey_node,
        "retrieve_memory": retrieve_memory_node,
        "build_plan": build_plan_node,
        "route": route_node,
        "execute_layer": execute_layer_node,
        "aggregate": aggregate_node,
        "save_memory": save_memory_node,
    }


def has_more_layers(state: CentralAgentState) -> str:
    """Conditional edge: return to execute_layer or proceed to aggregate."""
    layers = state.get("pending_layers") or []
    idx = state.get("current_layer_index", 0)
    if idx < len(layers):
        return "execute_layer"
    return "aggregate"
