"""LangGraph 编排流水线节点（Chapter-8 固定图模式）。

各节点职责与书稿章节对应，由 graph.py 中 make_nodes() 注册到 StateGraph：

    pre_survey      — Ch2  思维链预调查，抽取事实与待查项
    retrieve_memory — Ch3  向量检索长期记忆
    build_plan      — Ch4  任务拆解 + 依赖分析 + 子 Agent 路由
    execute_layer   — Ch5+ 按层并行调用子 Agent，收集 tool 输出
    aggregate       —      汇总子任务结果 → final_response
    save_memory     — Ch3  对话轮次与偏好摘要写回长期记忆

设计要点：
- GraphContext 注入 LLM / Registry / Prompts / Memory，供各节点复用
- 子 Agent 收到的 query = description + params + 依赖任务结果（不展开 tool 名）
- execute_layer 同层子任务用 asyncio.gather 并行
- tracing：节点级 @trace_span + tool.completed / sub_agent_conversation 等事件
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from opentelemetry import trace

from agent_framework.domain.agent_registry import SubAgentRegistry
from agent_framework.domain.domain_config import DomainConfig
from agent_framework.domain.domain_prompts import DomainPrompts
from agent_framework.domain.pipeline import PipelineConfig
from agent_framework.domain.task_planner import TaskPlanner
from agent_framework.infra.memory.aggregation_helpers import (
    direct_response_from_results,
    is_single_direct_response,
)
from agent_framework.infra.memory.memory_system import LongTermMemory
from agent_framework.tracing import (
    current_trace_add_event,
    get_logger,
    log_info,
    record_exception,
    record_tool_event,
    trace_span,
)
from agent_framework.tracing.trace_provider import span_name

from .state import CentralAgentState
from .stream_sink import StreamSink

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 日志 / 工具 / 子 Agent 辅助函数
# ---------------------------------------------------------------------------

def _streaming(state: CentralAgentState) -> bool:
    """是否启用流式输出（聚合阶段 token 流 + 减少 tracing 日志刷屏）。"""
    return bool(state.get("enable_stream"))


def _append_log(state: CentralAgentState, message: str, *, trace: bool = True) -> List[str]:
    """追加一条人类可读日志到 state.logs；trace=True 时同步写入 tracing（宜用于单行摘要）。"""
    logs = list(state.get("logs") or [])
    logs.append(message)
    if trace and not _streaming(state):
        log_info(logger, message.replace("\n", " "), thread_id=state.get("thread_id"))
    return logs


async def _stream_llm_text(llm: ChatOpenAI, messages: list, sink: StreamSink) -> str:
    """流式调用 LLM，逐 token 经 StreamSink 推到 CLI。"""
    parts: List[str] = []
    async for chunk in llm.astream(messages):
        token = chunk.content if isinstance(chunk.content, str) else ""
        if token:
            parts.append(token)
            sink.emit_token(token)
    return "".join(parts)


def _topological_layers(execution_plan: Dict[str, Any]) -> List[List[str]]:
    """从 execution_order + depends_on 构建拓扑分层执行序列。

    同层内 remaining 中所有依赖已完成的 task_id 进入同一层；
    execute_layer_node 对每层内任务 asyncio.gather 并行执行。
    若出现循环依赖等异常，回退取 remaining[0] 打破僵局。
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
            # 循环依赖等异常：强制取队首打破 while 僵局
            layer = [remaining[0]]
        layers.append(layer)
        for tid in layer:
            done.add(tid)
            remaining.remove(tid)
    return layers


def _format_result(result: Any) -> str:
    """将子任务结果格式化为 JSON，失败则 str()。"""
    try:
        return json.dumps(result, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(result)


def _log_subtask_start(
    state: CentralAgentState,
    logs: List[str],
    task: Dict[str, Any],
) -> List[str]:
    """记录子任务开始：task_id + Agent 名 + description 前 60 字。"""
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
    """记录子任务完成：CLI/state.logs 可展示完整 JSON；tracing 仅一条 subtask.done 摘要。"""
    agent_name = result.get("agent", "")
    status = result.get("status", "")
    if _streaming(state):
        logs = _append_log(
            {**state, "logs": logs},
            f"     ✓ {result.get('task_id')} {agent_name} {status}",
        )
    else:
        logs = _append_log({**state, "logs": logs}, f"     ✓ {agent_name} 执行完成")
        # 详情写入 state.logs，不逐行刷 tracing（避免 14 天预报等大块 JSON 刷屏）
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
            tool_call_count=result.get("tool_call_count", 0),
            result_preview=preview,
            thread_id=state.get("thread_id"),
        )
    return logs


def _tool_has_error(content: Any) -> bool:
    """判断 tool 返回内容是否含 error 字段（dict 或 JSON 字符串）。"""
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
    context_builder: Optional[Any] = None,
) -> str:
    """组装传给子 Agent 的 query，不含具体 tool 名称。

    优先注入 context_builder() 产出的领域上下文块；
    主体为 task.description + params 及依赖任务结果摘要。
    不传 tool 清单，由子 Agent System Prompt + Planner 路由的 description 决定行为。
    """
    lines: List[str] = []
    if context_builder is not None:
        block = (context_builder() or "").strip()
        if block:
            lines.append(block)

    desc = (task.get("description") or "").strip()
    lines.append(desc or "执行子任务")

    params = task.get("params") or {}
    if params:
        lines.append(f"参数: {json.dumps(params, ensure_ascii=False)}")

    for dep_id in task.get("depends_on", []):
        if dep_id in prior_results:
            dep_json = json.dumps(prior_results[dep_id], ensure_ascii=False)
            if len(dep_json) > 2000:
                # 截断过长 JSON 避免撑爆 Agent 上下文
                dep_json = dep_json[:2000] + "..."
            lines.append(f"依赖 {dep_id} 的结果: {dep_json}")

    return "\n".join(lines)


def _evaluate_subtask_status(
    agent_name: str,
    tool_outputs: List[Any],
    registry: Any = None,
) -> str:
    """评估子任务执行状态。

    - requires_tool 且无 tool 输出 → failed
    - 有 tool 输出且至少一个无 error → completed，全部 error → failed
    - 无 tool 输出（非 requires_tool）→ completed
    """
    requires_tool = False
    if registry is not None and hasattr(registry, "requires_tool"):
        requires_tool = registry.requires_tool(agent_name)
    if requires_tool and not tool_outputs:
        return "failed"
    if tool_outputs:
        if any(not _tool_has_error(output) for output in tool_outputs):
            return "completed"
        return "failed"
    return "completed"


def _pack_tool_data(tool_outputs: List[Any]) -> Any:
    """合并多个 tool 输出；单个直接返回，多个包装为 {calls, count} 结构。"""
    if not tool_outputs:
        return None
    if len(tool_outputs) == 1:
        return tool_outputs[0]
    return {"calls": tool_outputs, "count": len(tool_outputs)}


@trace_span(
    name=span_name("agent.invoke"),
    attrs_args=["task", "thread_id"],
    parent_arg="trace_parent",
)
async def _invoke_sub_agent(
    task: Dict[str, Any],
    prior_results: Dict[str, Any],
    thread_id: str,
    trace_parent: Optional[Any] = None,
    registry: Any = None,
    context_builder: Optional[Any] = None,
) -> Dict[str, Any]:
    """调用单个子 Agent（LangGraph ReAct + 工具 + 收集 tool/ai 消息）。

    返回 {task_id, agent, status, tool_data, agent_summary}
    - routing_failed 时标记 failed，不调用 Agent
    - query 截断预览 2000 字符写入 prompt
    - 每个 tool 输出记录 tool.completed 级 event
    """
    task_id = task["task_id"]
    agent_name = task.get("agent")
    routing_status = task.get("routing_status")

    if not agent_name or routing_status == "routing_failed":
        # Planner 未能路由到有效 Agent
        error_msg = "未能路由到有效子智能体"
        current_trace_add_event(
            "sub_agent_conversation",
            {
                "query": (task.get("description") or "")[:500],
                "agent": agent_name or "",
                "response": "",
                "status": "failed",
                "routing_status": routing_status or "routing_failed",
            },
        )
        return {
            "task_id": task_id,
            "agent": agent_name or "",
            "status": "failed",
            "routing_status": routing_status or "routing_failed",
            "tool_data": {"error": "routing_failed", "message": error_msg},
            "tool_call_count": 0,
            "agent_summary": "",
        }

    query_text = _build_sub_agent_query(
        task,
        prior_results,
        context_builder=context_builder,
    )
    query_preview = query_text[:500]

    try:
        agent = registry.get_agent(agent_name)
        # 独立 thread_id 避免 LangGraph checkpoint 冲突
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
            step=span_name("agent.invoke"),
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
            "tool_call_count": 0,
            "agent_summary": "",
        }


class GraphContext:
    """图节点共享依赖：LLM、TaskPlanner、Registry、Memory、StreamSink。

    make_nodes(ctx) 返回 dict 供 graph.py 注册 StateGraph 各节点。
    默认绑定出行领域（create_travel_registry / TravelPrompts）；
    外部可注入自定义 registry / prompts / domain_config。
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        memory_system: LongTermMemory | None = None,
        stream_sink: StreamSink | None = None,
        registry: SubAgentRegistry | None = None,
        prompts: DomainPrompts | None = None,
        domain_config: DomainConfig | None = None,
        pipeline: PipelineConfig | None = None,
    ) -> None:
        from domains.travel.prompt_bundle import TravelPrompts
        from domains.travel.registry import create_travel_registry, travel_domain_config

        self.llm = llm
        # demo 默认绑定出行领域，生产环境可替换 registry / prompts
        self.registry = registry or create_travel_registry()
        self.prompts = prompts or TravelPrompts.build()
        self.domain_config = domain_config or travel_domain_config()
        self.pipeline = pipeline or PipelineConfig()
        self.planner = TaskPlanner(
            llm,
            self.registry,
            prompts=self.prompts,
            domain_config=self.domain_config,
        )
        self.memory_system = memory_system
        self.stream_sink = stream_sink or StreamSink()


def make_nodes(ctx: GraphContext):
    """工厂：绑定 GraphContext，返回 LangGraph 各节点 callable。

    返回 key 与 graph.py 中 add_node 名称一一对应：
        pre_survey / retrieve_memory / build_plan / execute_layer / aggregate / save_memory
    """

    # --- Ch2 思维链预调查 ---
    @trace_span(
        name=span_name("orchestration.pre_survey"),
        attrs_args=["state"],
        record_result=False,
    )
    async def pre_survey_node(state: CentralAgentState) -> Dict[str, Any]:
        """Ch2：对用户 query 做思维链预调查，写入 state.pre_survey。"""
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

    # --- Ch3 长期记忆检索 ---
    @trace_span(
        name=span_name("orchestration.retrieve_memory"),
        attrs_args=["state"],
        record_result=False,
    )
    async def retrieve_memory_node(state: CentralAgentState) -> Dict[str, Any]:
        """Ch3：按 user_query 检索长期记忆，写入 state.retrieved_memories。"""
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

    # --- Ch4 任务拆解 + Ch6 子 Agent 路由 ---
    @trace_span(
        name=span_name("orchestration.build_plan"),
        attrs_args=["state"],
        record_result=False,
    )
    async def build_plan_node(state: CentralAgentState) -> Dict[str, Any]:
        """Ch4：任务拆解、依赖分析与路由子 Agent，初始化 pending_layers 等执行状态。"""
        if _streaming(state):
            ctx.stream_sink.emit_progress("\n📋 [Ch4] 任务拆解 → 依赖分析 → 子 Agent 路由...")
        logs = _append_log(state, "\n📋 [Ch4] 任务拆解 → 依赖分析 → 子 Agent 路由...")
        plan = await ctx.planner.build_execution_plan(
            state["user_query"],
            state.get("pre_survey") or {},
            state.get("retrieved_memories") or [],
        )
        layers = _topological_layers(plan)
        # pending_layers 供 execute_layer 逐层消费；subtask_results 累积结果
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

    # --- Ch5+ 分层执行子 Agent ---
    @trace_span(
        name=span_name("orchestration.execute_layer"),
        attrs_args=["state"],
        record_result=False,
    )
    async def execute_layer_node(state: CentralAgentState) -> Dict[str, Any]:
        """Ch5+：取 pending_layers[current_layer_index] 同层并行调用子 Agent，合并 subtask_results。

        同层子任务 asyncio.gather 并行，完成后 current_layer_index++。
        部分失败时记录 layer.partial_failure 事件，不阻断后续层。
        """
        layers = state.get("pending_layers") or []
        idx = state.get("current_layer_index", 0)
        subtasks = {t["task_id"]: t for t in state.get("subtasks") or []}
        results = dict(state.get("subtask_results") or {})
        thread_id = state.get("thread_id", "default")
        logs = list(state.get("logs") or [])

        if idx >= len(layers):
            # 已无待执行层时返回空 dict，由 has_more_layers 路由到 aggregate
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
                agent_name = task.get("agent") or ""
                ctx.stream_sink.emit_progress(f"  🔄 {tid} → {agent_name}")
            logs = _log_subtask_start({**state, "logs": logs}, logs, task)

        if len(tasks) == 1:
            layer_results = [
                await _invoke_sub_agent(
                    tasks[0],
                    results,
                    thread_id,
                    trace_parent=layer_span,
                    registry=ctx.registry,
                    context_builder=ctx.domain_config.context_builder,
                )
            ]
        else:
            # 多任务同层 → 并行 invoke，trace parent 共享 layer span
            layer_results = list(await asyncio.gather(*[
                _invoke_sub_agent(
                    t,
                    results,
                    thread_id,
                    trace_parent=layer_span,
                    registry=ctx.registry,
                    context_builder=ctx.domain_config.context_builder,
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

    # --- 聚合：直答 / 记忆增强 LLM / 标准 aggregation prompt ---
    @trace_span(
        name=span_name("orchestration.aggregate"),
        attrs_args=["state"],
        record_result=False,
    )
    async def aggregate_node(state: CentralAgentState) -> Dict[str, Any]:
        """汇总 subtask_results 为 final_response。

        分支策略：
        1. 单任务直答 → 跳过 LLM 聚合
        2. 启用记忆 → build_prompt + 子任务结果 + memory_aggregation_instruction
        3. 否则 → prompts.aggregation 模板
        """
        if _streaming(state):
            ctx.stream_sink.emit_progress("\n📝 聚合结果...")
        logs = _append_log(state, "\n📝 聚合结果...")
        plan = state.get("execution_plan") or {}
        results = state.get("subtask_results") or {}
        user_query = state["user_query"]
        thread_id = state.get("thread_id", "default")
        stream = _streaming(state)
        sink = ctx.stream_sink
        prompts = ctx.prompts
        title = prompts.single_task_title if is_single_direct_response(results) else prompts.multi_task_title
        header = "\n" + "=" * 80 + "\n" + title + "\n" + "=" * 80 + "\n"
        footer = "=" * 80

        if is_single_direct_response(results):
            # 单 Agent 直答场景：取 agent_summary，无需 LLM 聚合
            final_text = direct_response_from_results(results)
            if stream:
                sink.emit_progress("  ✓ 单任务直答，跳过 LLM 聚合")
            logs = _append_log(
                {**state, "logs": logs},
                f"  ✓ {prompts.aggregation_skip_hint}",
            )
        elif ctx.memory_system and state.get("enable_memory", True):
            if stream:
                sink.emit_progress("  🧠 聚合时注入长期记忆上下文...")
            logs = _append_log(
                {**state, "logs": logs},
                "  🧠 聚合时注入长期记忆上下文...",
            )
            prompt = ctx.memory_system.build_prompt(
                thread_id,
                user_query,
                ctx.memory_system.search_memories(user_query),
            )
            prompt += f"\n\n## 子任务执行结果\n{json.dumps(results, ensure_ascii=False, indent=2)}"
            prompt += f"\n\n{prompts.memory_aggregation_instruction}"
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
                sink.emit_progress("  🧠 聚合使用标准 prompt...")
            logs = _append_log({**state, "logs": logs}, "  🧠 聚合使用标准 prompt...")
            prompt = prompts.aggregation.format(
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
        name=span_name("orchestration.save_memory"),
        attrs_args=["state"],
        record_result=False,
    )
    async def save_memory_node(state: CentralAgentState) -> Dict[str, Any]:
        """Ch3：将本轮对话与偏好摘要写回长期记忆（ingest + record_turn）。"""
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
    """execute_layer 条件边：还有未执行层 → execute_layer，否则 → aggregate。"""
    layers = state.get("pending_layers") or []
    idx = state.get("current_layer_index", 0)
    if idx < len(layers):
        return "execute_layer"
    return "aggregate"
