"""LangGraph 节点：复用 Chapter-6 的 TaskPlanner / Memory / SubAgent"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

LG_DIR = Path(__file__).resolve().parent
if str(LG_DIR) not in sys.path:
    sys.path.insert(0, str(LG_DIR))

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from _ch6_loader import load_ch6_module

_aggregation = load_ch6_module("aggregation_helpers")

_central = load_ch6_module("central_orchestrator")
_prompts = load_ch6_module("prompts")
_memory = load_ch6_module("memory_system")
_sub = load_ch6_module("sub_agents")
_planner = load_ch6_module("task_planner")

is_single_direct_response = _aggregation.is_single_direct_response
direct_response_from_results = _aggregation.direct_response_from_results
MEMORY_AGGREGATION_INSTRUCTION = _aggregation.MEMORY_AGGREGATION_INSTRUCTION
SubAgentRegistry = _central.SubAgentRegistry
AGGREGATION_PROMPT = _prompts.AGGREGATION_PROMPT
LongTermMemory = _memory.LongTermMemory
SubAgentFactory = _sub.SubAgentFactory
TaskPlanner = _planner.TaskPlanner

from state import CentralAgentState


def _append_log(state: CentralAgentState, message: str) -> List[str]:
    logs = list(state.get("logs") or [])
    logs.append(message)
    print(message, flush=True)
    return logs


def _topological_layers(execution_plan: Dict[str, Any]) -> List[List[str]]:
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
    try:
        return json.dumps(result, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(result)


def _log_subtask_start(
    state: CentralAgentState,
    logs: List[str],
    task: Dict[str, Any],
) -> List[str]:
    task_id = task["task_id"]
    agent_name = task.get("agent", "ItineraryAgent")
    description = (task.get("description") or "")[:60]
    logs = _append_log(
        {**state, "logs": logs},
        f"  🔄 {task_id} → {agent_name}: {description}{'...' if len(task.get('description') or '') > 60 else ''}",
    )
    return logs


def _log_subtask_done(
    state: CentralAgentState,
    logs: List[str],
    result: Dict[str, Any],
) -> List[str]:
    agent_name = result.get("agent", "?")
    logs = _append_log({**state, "logs": logs}, f"     ✓ {agent_name} 执行完成")
    logs = _append_log({**state, "logs": logs}, "     " + "-" * 56)
    for line in _format_result(result).splitlines():
        logs = _append_log({**state, "logs": logs}, f"     {line}")
    logs = _append_log({**state, "logs": logs}, "     " + "-" * 56)
    return logs


async def _invoke_sub_agent(
    task: Dict[str, Any],
    prior_results: Dict[str, Any],
    thread_id: str,
) -> Dict[str, Any]:
    task_id = task["task_id"]
    agent_name = task.get("agent", "ItineraryAgent")
    description = task.get("description", "")

    query_parts = [description]
    if task.get("params"):
        query_parts.append(f"参数: {json.dumps(task['params'], ensure_ascii=False)}")
    for dep_id in task.get("depends_on", []):
        if dep_id in prior_results:
            dep_json = json.dumps(prior_results[dep_id], ensure_ascii=False)
            if len(dep_json) > 2000:
                dep_json = dep_json[:2000] + "..."
            query_parts.append(f"依赖 {dep_id} 的结果: {dep_json}")

    agent = SubAgentFactory.get_agent(agent_name)
    state = await agent.ainvoke(
        {"messages": [("user", "\n".join(query_parts))]},
        {"configurable": {"thread_id": f"{thread_id}_{task_id}"}},
    )

    tool_outputs = []
    agent_text = ""
    for msg in state.get("messages", []):
        if hasattr(msg, "type"):
            if msg.type == "tool" and hasattr(msg, "content"):
                try:
                    tool_outputs.append(json.loads(msg.content))
                except (json.JSONDecodeError, TypeError):
                    tool_outputs.append(msg.content)
            elif msg.type == "ai" and getattr(msg, "content", None):
                agent_text = msg.content

    return {
        "task_id": task_id,
        "agent": agent_name,
        "status": "completed",
        "tool_data": tool_outputs[-1] if tool_outputs else None,
        "agent_summary": agent_text,
    }


class GraphContext:
    """图节点共享上下文（LLM、规划器、记忆）"""

    def __init__(
        self,
        llm: ChatOpenAI,
        memory_system: LongTermMemory | None = None,
    ) -> None:
        self.llm = llm
        self.registry = SubAgentRegistry()
        self.planner = TaskPlanner(llm, self.registry)
        self.memory_system = memory_system


def make_nodes(ctx: GraphContext):
    """工厂：绑定 LLM / Planner 后返回各节点函数"""

    async def pre_survey_node(state: CentralAgentState) -> Dict[str, Any]:
        logs = _append_log(state, "\n🔍 [Ch2] 思维链预调查...")
        pre_survey = await ctx.planner.run_pre_survey(state["user_query"])
        summary = {k: v for k, v in pre_survey.items() if k != "raw_text"}
        logs = _append_log(
            {**state, "logs": logs},
            "✓ 预调查完成\n" + json.dumps(summary, ensure_ascii=False, indent=2),
        )
        return {"pre_survey": pre_survey, "logs": logs}

    async def retrieve_memory_node(state: CentralAgentState) -> Dict[str, Any]:
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
        return {"retrieved_memories": memories, "logs": logs}

    async def build_plan_node(state: CentralAgentState) -> Dict[str, Any]:
        logs = _append_log(state, "\n📋 [Ch4] 任务拆解 → 依赖分析 → 子智能体路由...")
        plan = await ctx.planner.build_execution_plan(
            state["user_query"],
            state.get("pre_survey") or {},
            state.get("retrieved_memories") or [],
        )
        layers = _topological_layers(plan)
        logs = _append_log(
            {**state, "logs": logs},
            f"✓ 共 {len(plan['subtasks'])} 个子任务，{len(layers)} 个执行层",
        )
        logs = _append_log(
            {**state, "logs": logs},
            f"  执行顺序: {' → '.join(plan['execution_order'])}",
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

    async def execute_layer_node(state: CentralAgentState) -> Dict[str, Any]:
        layers = state.get("pending_layers") or []
        idx = state.get("current_layer_index", 0)
        subtasks = {t["task_id"]: t for t in state.get("subtasks") or []}
        results = dict(state.get("subtask_results") or {})
        thread_id = state.get("thread_id", "default")
        logs = list(state.get("logs") or [])

        if idx >= len(layers):
            return {}

        layer = layers[idx]
        logs = _append_log({**state, "logs": logs}, f"\n⚙️ [Ch5+] 执行第 {idx + 1}/{len(layers)} 层: {layer}")

        tasks = [subtasks[tid] for tid in layer]
        for task in tasks:
            logs = _log_subtask_start({**state, "logs": logs}, logs, task)

        if len(tasks) == 1:
            layer_results = [await _invoke_sub_agent(tasks[0], results, thread_id)]
        else:
            layer_results = await asyncio.gather(*[
                _invoke_sub_agent(t, results, thread_id) for t in tasks
            ])

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

    async def aggregate_node(state: CentralAgentState) -> Dict[str, Any]:
        logs = _append_log(state, "\n📝 聚合结果...")
        plan = state.get("execution_plan") or {}
        results = state.get("subtask_results") or {}
        user_query = state["user_query"]
        thread_id = state.get("thread_id", "default")

        if is_single_direct_response(results):
            final_text = direct_response_from_results(results)
            logs = _append_log(
                {**state, "logs": logs},
                "  ✓ 单任务查询，直接使用子智能体回复（跳过旅行规划聚合）",
            )
        elif ctx.memory_system and state.get("enable_memory", True):
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
            response = await ctx.llm.ainvoke([HumanMessage(content=prompt)])
            final_text = response.content or ""
            logs = _append_log({**state, "logs": logs}, "  ✓ 聚合完成")
        else:
            logs = _append_log({**state, "logs": logs}, "  🧠 聚合未使用记忆（记忆未启用）")
            prompt = AGGREGATION_PROMPT.format(
                user_query=user_query,
                pre_survey=json.dumps(plan.get("pre_survey", {}), ensure_ascii=False, indent=2),
                memories=json.dumps(plan.get("retrieved_memories", []), ensure_ascii=False, indent=2),
                total_goal=plan.get("total_goal", ""),
                results=json.dumps(results, ensure_ascii=False, indent=2),
            )
            response = await ctx.llm.ainvoke([HumanMessage(content=prompt)])
            final_text = response.content or ""
            logs = _append_log({**state, "logs": logs}, "  ✓ 聚合完成")

        title = "📋 最终回复" if is_single_direct_response(results) else "📋 最终旅行规划"
        logs = _append_log({**state, "logs": logs}, "\n" + "=" * 80)
        logs = _append_log({**state, "logs": logs}, title)
        logs = _append_log({**state, "logs": logs}, "=" * 80)
        logs = _append_log({**state, "logs": logs}, final_text)
        logs = _append_log({**state, "logs": logs}, "=" * 80)
        return {"final_response": final_text, "logs": logs}

    async def save_memory_node(state: CentralAgentState) -> Dict[str, Any]:
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
    """条件边：是否还有未执行的层"""
    layers = state.get("pending_layers") or []
    idx = state.get("current_layer_index", 0)
    if idx < len(layers):
        return "execute_layer"
    return "aggregate"
