"""LangGraph 固定图节点：预调查 → 记忆检索 → 规划 → 分层执行 → 聚合 → 写回记忆。

各节点通过 make_nodes() 工厂生成，由 graph.py 组装为完整编排图。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from opentelemetry import trace

from travel_multi_agent.agents.factory import SubAgentFactory
from travel_multi_agent.domain.agent_registry import SubAgentRegistry
from travel_multi_agent.domain.plan_context import format_time_anchor_block
from travel_multi_agent.domain.prompts import AGGREGATION_PROMPT
from travel_multi_agent.domain.task_planner import TaskPlanner
from travel_multi_agent.infra.memory.aggregation_helpers import (
    MEMORY_AGGREGATION_INSTRUCTION,
    direct_response_from_results,
    is_single_direct_response,
)
from travel_multi_agent.infra.memory.memory_system import LongTermMemory
from travel_multi_agent.tracing import (
    current_trace_add_event,
    get_logger,
    log_info,
    record_exception,
    record_tool_event,
    trace_span,
)

from .state import CentralAgentState
from .stream_sink import StreamSink

logger = get_logger(__name__)


def _streaming(state: CentralAgentState) -> bool:
    return bool(state.get("enable_stream"))


def _append_log(state: CentralAgentState, message: str, *, trace: bool = True) -> List[str]:
    """向 state.logs 追加一行；流式模式下默认不写 tracing（避免刷屏）。"""
    logs = list(state.get("logs") or [])
    logs.append(message)
    if trace and not _streaming(state):
        log_info(logger, message.replace("\n", " "), thread_id=state.get("thread_id"))
    return logs


async def _stream_llm_text(llm: ChatOpenAI, messages: list, sink: StreamSink) -> str:
    """调用 LLM 并逐 token 转发到 StreamSink。"""
    parts: List[str] = []
    async for chunk in llm.astream(messages):
        token = chunk.content if isinstance(chunk.content, str) else ""
        if token:
            parts.append(token)
            sink.emit_token(token)
    return "".join(parts)


def _topological_layers(execution_plan: Dict[str, Any]) -> List[List[str]]:
    """将 execution_order 按 depends_on 拆成可并行执行的层。

    每层内任务依赖均已满足，可 asyncio.gather 并行；层与层之间串行。
    若依赖信息缺失导致无法推进，强制取 remaining[0] 避免死循环。
    """
    subtasks = {t["task_id"]: t for t in execution_plan.get("subtasks", [])}
    order = execution_plan.get("execution_order", list(subtasks.keys()))
    done: set = set()
    layers: List[List[str]] = []
    remaining = list(order)
    while remaining:
        layer = [
            tid for tid in remaining
            if all(d in done for d in subtasks[tid].get("depends_on", []))
        ]
        if not layer:
            layer = [remaining[0]]
        layers.append(layer)
        for tid in layer:
            done.add(tid)
            remaining.remove(tid)
    return layers


def _format_result(result: Any) -> str:
    """将子任务结果格式化为 JSON 字符串，不可序列化时回退 str()。"""
    try:
        return json.dumps(result, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(result)


def _log_subtask_start(
    state: CentralAgentState,
    logs: List[str],
    task: Dict[str, Any],
) -> List[str]:
    """记录子任务开始：task_id、目标 Agent、描述摘要（最多 60 字符）。"""
    task_id = task["task_id"]
    agent_name = task.get("agent", "ItineraryAgent")
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
    """记录子任务完成；流式模式仅一行摘要，非流式保留完整 JSON。"""
    agent_name = result.get("agent", "?")
    status = result.get("status", "?")
    if _streaming(state):
        logs = _append_log(
            {**state, "logs": logs},
            f"     ✓ {result.get('task_id')} {agent_name} 完成（{status}）",
        )
    else:
        logs = _append_log({**state, "logs": logs}, f"     ✓ {agent_name} 执行完成")
        logs = _append_log({**state, "logs": logs}, "     " + "-" * 56)
        for line in _format_result(result).splitlines():
            logs = _append_log({**state, "logs": logs}, f"     {line}")
        logs = _append_log({**state, "logs": logs}, "     " + "-" * 56)
    if not _streaming(state):
        log_info(
            logger,
            "subtask.done",
            task_id=result.get("task_id"),
            agent=agent_name,
            status=status,
            thread_id=state.get("thread_id"),
        )
    return logs


def _tool_has_error(content: Any) -> bool:
    """检测工具返回值是否含 error 字段（支持 dict、JSON 字符串或含 "error" 的纯文本）。"""
    if content is None:
        return False
    if isinstance(content, dict):
        return bool(content.get("error"))
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and parsed.get("error"):
                return True
        except (json.JSONDecodeError, TypeError):
            pass
        return '"error"' in content.lower()
    return False


def _build_sub_agent_query(
    task: Dict[str, Any],
    prior_results: Dict[str, Any],
) -> str:
    """构造子 Agent 用户消息（通用框架层）。

    只拼接通用的上下文信息，具体工具调用指令由：
    - 子 Agent 的 System Prompt（agents/*.py）
    - Planner 生成的 task.description
    负责，不在编排层硬编码。
    """
    lines: List[str] = [format_time_anchor_block()]

    desc = (task.get("description") or "").strip()
    lines.append(desc or "请完成任务。")

    params = task.get("params") or {}
    if params:
        lines.append(f"参数: {json.dumps(params, ensure_ascii=False)}")

    for dep_id in task.get("depends_on", []):
        if dep_id in prior_results:
            dep_json = json.dumps(prior_results[dep_id], ensure_ascii=False)
            if len(dep_json) > 2000:
                dep_json = dep_json[:2000] + "..."
            lines.append(f"依赖 {dep_id} 的结果: {dep_json}")

    return "\n".join(lines)


def _evaluate_subtask_status(
    agent_name: str,
    tool_outputs: List[Any],
    registry: Any = None,
) -> str:
    """执行状态判断：优先从 registry 查询 requires_tool，registry 未提供时回退为「有 error 才 failed」。"""
    requires_tool = False
    if registry is not None and hasattr(registry, "requires_tool"):
        requires_tool = registry.requires_tool(agent_name)
    if requires_tool and not tool_outputs:
        return "failed"
    if tool_outputs and _tool_has_error(tool_outputs[-1]):
        return "failed"
    return "completed"


def _pack_tool_data(tool_outputs: List[Any]) -> Any:
    if not tool_outputs:
        return None
    if len(tool_outputs) == 1:
        return tool_outputs[0]
    return {"calls": tool_outputs, "count": len(tool_outputs)}


@trace_span(
    name="latc.travel-multi-agent.agent.invoke",
    attrs_args=["task", "thread_id"],
    parent_arg="trace_parent",
)
async def _invoke_sub_agent(
    task: Dict[str, Any],
    prior_results: Dict[str, Any],
    thread_id: str,
    trace_parent: Optional[Any] = None,
    registry: Any = None,
) -> Dict[str, Any]:
    """调用单个子 Agent：拼装描述 + 参数 + 上游依赖结果，解析 tool/ai 消息。

    返回 {task_id, agent, status, tool_data, agent_summary}。
    异常不向上抛出，统一封装为 status="failed" 以便同层其他任务继续执行；
    依赖结果超过 2000 字符会截断，避免 prompt 过长。
    """
    task_id = task["task_id"]
    agent_name = task.get("agent", "ItineraryAgent")

    query_text = _build_sub_agent_query(task, prior_results)
    query_preview = query_text[:500]

    try:
        agent = SubAgentFactory.get_agent(agent_name)
        agent_state = await agent.ainvoke(
            {"messages": [("user", query_text)]},
            {"configurable": {"thread_id": f"{thread_id}_{task_id}"}},
        )

        tool_outputs: List[Any] = []
        agent_text = ""
        for msg in agent_state.get("messages", []):
            if not hasattr(msg, "type"):
                continue
            if msg.type == "tool" and hasattr(msg, "content"):
                content = msg.content
                tool_name = getattr(msg, "name", None) or f"{agent_name}.tool"
                parsed_content: Any = content
                try:
                    if isinstance(content, str):
                        parsed_content = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    parsed_content = content
                tool_outputs.append(parsed_content)
                record_tool_event(
                    tool_name,
                    task_id=task_id,
                    agent_name=agent_name,
                    has_error=_tool_has_error(parsed_content),
                    output_preview=str(content)[:200] if content else None,
                )
            elif msg.type == "ai" and getattr(msg, "content", None):
                agent_text = msg.content

        tool_data = _pack_tool_data(tool_outputs)
        status = _evaluate_subtask_status(agent_name, tool_outputs, registry=registry)
        if status == "failed":
            log_info(
                logger,
                "agent.tool_missing" if not tool_outputs else "agent.tool_error",
                agent=agent_name,
                task_id=task_id,
                thread_id=thread_id,
                tool_call_count=len(tool_outputs),
            )

        current_trace_add_event(
            "sub_agent_conversation",
            {
                "query": query_preview,
                "agent": agent_name,
                "response": (agent_text or "")[:500],
                "status": status,
                "tool_call_count": len(tool_outputs),
            },
        )
        return {
            "task_id": task_id,
            "agent": agent_name,
            "status": status,
            "tool_data": tool_data,
            "tool_call_count": len(tool_outputs),
            "agent_summary": agent_text,
        }
    except Exception as exc:
        record_exception(
            exc,
            step="latc.travel-multi-agent.agent.invoke",
            agent_name=agent_name,
            task_id=task_id,
            thread_id=thread_id,
        )
        current_trace_add_event(
            "sub_agent_conversation",
            {
                "query": query_preview,
                "agent": agent_name,
                "response": "",
                "status": "failed",
            },
        )
        return {
            "task_id": task_id,
            "agent": agent_name,
            "status": "failed",
            "tool_data": {"error": str(exc), "error_type": type(exc).__name__},
            "agent_summary": "",
        }


class GraphContext:
    """图节点共享上下文：LLM、Agent 注册表、TaskPlanner、长期记忆系统。"""

    def __init__(
        self,
        llm: ChatOpenAI,
        memory_system: LongTermMemory | None = None,
        stream_sink: StreamSink | None = None,
    ) -> None:
        """初始化编排依赖；memory_system 为 None 时记忆相关节点自动跳过。"""
        self.llm = llm
        self.registry = SubAgentRegistry()
        self.planner = TaskPlanner(llm, self.registry)
        self.memory_system = memory_system
        self.stream_sink = stream_sink or StreamSink()


def make_nodes(ctx: GraphContext):
    """工厂：绑定 GraphContext 后返回 LangGraph 可用的节点函数字典。

    返回键：pre_survey / retrieve_memory / build_plan / execute_layer /
            aggregate / save_memory。
    """

    @trace_span(
        name="latc.travel-multi-agent.orchestration.pre_survey",
        attrs_args=["state"],
        record_result=False,
    )
    async def pre_survey_node(state: CentralAgentState) -> Dict[str, Any]:
        """Ch2 节点：对用户请求做思维链预调查，写入 state.pre_survey。"""
        if _streaming(state):
            ctx.stream_sink.emit_progress("\n🔍 [Ch2] 思维链预调查...")
        logs = _append_log(state, "\n🔍 [Ch2] 思维链预调查...")
        pre_survey = await ctx.planner.run_pre_survey(state["user_query"])
        summary = {k: v for k, v in pre_survey.items() if k != "raw_text"}
        if _streaming(state):
            ctx.stream_sink.emit_progress("✓ [Ch2] 预调查完成")
            logs = _append_log({**state, "logs": logs}, "✓ 预调查完成")
        else:
            logs = _append_log(
                {**state, "logs": logs},
                "✓ 预调查完成\n" + json.dumps(summary, ensure_ascii=False, indent=2),
            )
        return {"pre_survey": pre_survey, "logs": logs}

    @trace_span(
        name="latc.travel-multi-agent.orchestration.retrieve_memory",
        attrs_args=["state"],
        record_result=False,
    )
    async def retrieve_memory_node(state: CentralAgentState) -> Dict[str, Any]:
        """Ch3 节点：按 user_query 检索长期记忆，供后续规划与聚合使用。"""
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

    @trace_span(
        name="latc.travel-multi-agent.orchestration.build_plan",
        attrs_args=["state"],
        record_result=False,
    )
    async def build_plan_node(state: CentralAgentState) -> Dict[str, Any]:
        """Ch4 节点：拆解子任务、分析依赖、路由 Agent，并预计算 pending_layers。"""
        if _streaming(state):
            ctx.stream_sink.emit_progress("\n📋 [Ch4] 任务拆解 → 依赖分析 → 子智能体路由...")
        logs = _append_log(state, "\n📋 [Ch4] 任务拆解 → 依赖分析 → 子智能体路由...")
        plan = await ctx.planner.build_execution_plan(
            state["user_query"],
            state.get("pre_survey") or {},
            state.get("retrieved_memories") or [],
        )
        layers = _topological_layers(plan)
        summary = f"✓ 共 {len(plan['subtasks'])} 个子任务，{len(layers)} 个执行层"
        order_line = f"  执行顺序: {' → '.join(plan['execution_order'])}"
        if _streaming(state):
            ctx.stream_sink.emit_progress(summary)
            ctx.stream_sink.emit_progress(order_line)
        logs = _append_log({**state, "logs": logs}, summary)
        logs = _append_log({**state, "logs": logs}, order_line)
        log_info(
            logger,
            "plan.built",
            subtask_count=len(plan.get("subtasks", [])),
            layer_count=len(layers),
            thread_id=state.get("thread_id"),
        )
        current_trace_add_event(
            "plan.built",
            {
                "subtask.count": len(plan.get("subtasks", [])),
                "layer.count": len(layers),
                "execution.order": ",".join(plan.get("execution_order", [])),
            },
        )
        return {
            "execution_plan": plan,
            "total_goal": plan.get("total_goal", ""),
            "subtasks": plan.get("subtasks", []),
            "execution_order": plan.get("execution_order", []),
            "pending_layers": layers,
            "current_layer_index": 0,
            "subtask_results": {},
            "logs": logs,
        }

    @trace_span(
        name="latc.travel-multi-agent.orchestration.execute_layer",
        attrs_args=["state"],
        record_result=False,
    )
    async def execute_layer_node(state: CentralAgentState) -> Dict[str, Any]:
        """Ch5+ 节点：执行当前层子任务；同层多任务并行，层内失败不阻断其余任务。"""
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

        layer_msg = f"\n⚙️ [Ch5+] 执行第 {idx + 1}/{len(layers)} 层: {layer}"
        if _streaming(state):
            ctx.stream_sink.emit_progress(layer_msg)
        logs = _append_log({**state, "logs": logs}, layer_msg)

        tasks = [subtasks[tid] for tid in layer]
        for task in tasks:
            if _streaming(state):
                tid = task["task_id"]
                agent_name = task.get("agent", "ItineraryAgent")
                ctx.stream_sink.emit_progress(f"  ▶ {tid} → {agent_name}")
            logs = _log_subtask_start({**state, "logs": logs}, logs, task)

        if len(tasks) == 1:
            layer_results = [
                await _invoke_sub_agent(tasks[0], results, thread_id, trace_parent=layer_span, registry=ctx.registry)
            ]
        else:
            layer_results = list(await asyncio.gather(*[
                _invoke_sub_agent(t, results, thread_id, trace_parent=layer_span, registry=ctx.registry) for t in tasks
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

    @trace_span(
        name="latc.travel-multi-agent.orchestration.aggregate",
        attrs_args=["state"],
        record_result=False,
    )
    async def aggregate_node(state: CentralAgentState) -> Dict[str, Any]:
        """聚合节点：单任务直出 / 带记忆 LLM 聚合 / 无记忆 AGGREGATION_PROMPT 三选一。"""
        if _streaming(state):
            ctx.stream_sink.emit_progress("\n📝 聚合结果...")
        logs = _append_log(state, "\n📝 聚合结果...")
        plan = state.get("execution_plan") or {}
        results = state.get("subtask_results") or {}
        user_query = state["user_query"]
        thread_id = state.get("thread_id", "default")
        stream = _streaming(state)
        sink = ctx.stream_sink
        title = "📋 最终回复" if is_single_direct_response(results) else "📋 最终旅行规划"
        header = "\n" + "=" * 80 + "\n" + title + "\n" + "=" * 80 + "\n"
        footer = "=" * 80

        if is_single_direct_response(results):
            final_text = direct_response_from_results(results)
            if stream:
                sink.emit_progress("  ✓ 单任务直出（跳过聚合 LLM）")
            logs = _append_log(
                {**state, "logs": logs},
                "  ✓ 单任务查询，直接使用子智能体回复（跳过旅行规划聚合）",
            )
        elif ctx.memory_system and state.get("enable_memory", True):
            if stream:
                sink.emit_progress("  🧠 聚合时注入长期记忆上下文")
            logs = _append_log(
                {**state, "logs": logs},
                "  🧠 聚合时注入长期记忆上下文",
            )
            prompt = ctx.memory_system.build_prompt(
                thread_id,
                user_query,
                ctx.memory_system.search_memories(user_query),
            )
            prompt += f"\n\n## 子任务执行结果\n{json.dumps(results, ensure_ascii=False, indent=2)}"
            prompt += f"\n\n{MEMORY_AGGREGATION_INSTRUCTION}"
            messages = [HumanMessage(content=prompt)]
            if stream:
                sink.emit_progress(header)
                final_text = await _stream_llm_text(ctx.llm, messages, sink)
                sink.emit_progress("\n" + footer)
            else:
                response = await ctx.llm.ainvoke(messages)
                final_text = response.content or ""
            logs = _append_log({**state, "logs": logs}, "  ✓ 聚合完成")
        else:
            if stream:
                sink.emit_progress("  🧠 聚合未使用记忆（记忆未启用）")
            logs = _append_log({**state, "logs": logs}, "  🧠 聚合未使用记忆（记忆未启用）")
            prompt = AGGREGATION_PROMPT.format(
                user_query=user_query,
                pre_survey=json.dumps(plan.get("pre_survey", {}), ensure_ascii=False, indent=2),
                memories=json.dumps(plan.get("retrieved_memories", []), ensure_ascii=False, indent=2),
                total_goal=plan.get("total_goal", ""),
                results=json.dumps(results, ensure_ascii=False, indent=2),
            )
            messages = [HumanMessage(content=prompt)]
            if stream:
                sink.emit_progress(header)
                final_text = await _stream_llm_text(ctx.llm, messages, sink)
                sink.emit_progress("\n" + footer)
            else:
                response = await ctx.llm.ainvoke(messages)
                final_text = response.content or ""
            logs = _append_log({**state, "logs": logs}, "  ✓ 聚合完成")

        if stream and is_single_direct_response(results):
            sink.emit_progress(header)
            sink.emit_token(final_text)
            sink.emit_progress("\n" + footer)

        logs = _append_log({**state, "logs": logs}, "\n" + "=" * 80)
        logs = _append_log({**state, "logs": logs}, title)
        logs = _append_log({**state, "logs": logs}, "=" * 80)
        logs = _append_log(
            {**state, "logs": logs},
            final_text,
            trace=not stream,
        )
        logs = _append_log({**state, "logs": logs}, "=" * 80)
        trace.get_current_span().set_attribute(
            "final_response.length",
            len(final_text or ""),
        )
        return {"final_response": final_text, "logs": logs}

    @trace_span(
        name="latc.travel-multi-agent.orchestration.save_memory",
        attrs_args=["state"],
        record_result=False,
    )
    async def save_memory_node(state: CentralAgentState) -> Dict[str, Any]:
        """Ch3 写回节点：记录对话轮次，并将用户请求 + 回复摘要写入长期偏好记忆。"""
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
        "execute_layer": execute_layer_node,
        "aggregate": aggregate_node,
        "save_memory": save_memory_node,
    }


def has_more_layers(state: CentralAgentState) -> str:
    """LangGraph 条件边：current_layer_index 未越界则继续 execute_layer，否则进入 aggregate。"""
    layers = state.get("pending_layers") or []
    idx = state.get("current_layer_index", 0)
    if idx < len(layers):
        return "execute_layer"
    return "aggregate"
