"""Control request routing and centralized multi-agent execution for Chapter 6.

This module is the entry point for the Chapter 6 travel-planning application.
Its central system prompt classifies non-travel requests for direct model reply
and sends travel requests into a code-controlled pipeline. The orchestrator then
coordinates fact survey, long-term memory, task planning, independent agent
routing, dependency-layer execution, final itinerary selection, and memory
write-back.

Main capabilities:

- Separate direct requests from travel tasks before planning.
- Compose Chapter 2, 3, 4, and 5 services into one explicit execution flow.
- Execute dependency layers and enforce each agent's concurrency policy.
- Pass dependency results to downstream agents and preserve failure states.
- Select the final itinerary result and persist the completed travel turn.

Planning, routing, and execution are deliberately separate. The module keeps
the sequence and control decisions in code, while the language model is used
only for request classification and the specialized planning/routing prompts.

:author: zhanghong, liuhui
:date: 2026-11-18
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional

from langchain_openai import ChatOpenAI

from agent_routing import AgentRouter
from config import CHROMA_DIR, load_project_dotenv
from domain_agents import (
    DomainAgentRegistry,
    build_domain_agent_user_message,
    get_domain_agent,
    parse_domain_agent_invoke_result,
)
from llm import create_llm, invoke_json
from memory_adapter import MemoryAdapter
from task_plan import create_task_plan, survey_facts

load_project_dotenv()


CENTRAL_AGENT_SYSTEM_PROMPT = """你是旅行规划应用的中心编排智能体，负责请求分诊和后续流程控制。

## 任务类型判断
将用户请求分为两类：

1. 旅行任务
包括旅行行程规划，以及天气、景点、酒店、餐厅、航班、交通等旅行相关信息查询。这类任务需要后续的事实调查、任务规划、领域智能体分配和执行，不能直接凭模型知识作答。

2. 非旅行任务
包括问候、写作、编程、常识问答、数学计算以及与旅行无关的信息查询。这类任务不需要启动任务规划或多智能体执行。

## 处理规则
1. 如果是非旅行任务：直接使用你自己的通用能力回答；


2. 如果是旅行任务：
   - 不要直接生成最终答案；
   - 将任务交给后续流程：先完成事实调查、任务拆解和依赖分析，再匹配领域智能体，最后按执行计划调用领域智能体；
   - response 字段留空，等待后续流程生成最终答案。

## 输出格式
只输出 JSON 对象，不要输出解释文字：

非旅行任务：
{"route": "direct", "response": "直接给用户的完整回答"}

旅行任务：
{"route": "travel", "response": ""}
"""


def direct_response_from_results(results: Dict[str, Any]) -> str:
    """Extract the direct user-facing response from one subtask result."""
    result = next(iter(results.values()))
    summary = (result.get("agent_summary") or "").strip()
    if summary:
        return summary
    tool_data = result.get("tool_data")
    if tool_data is not None:
        return json.dumps(tool_data, ensure_ascii=False, indent=2)
    return json.dumps(result, ensure_ascii=False, indent=2)


async def run_task_layer(
    layer: List[str],
    subtasks: Dict[str, Dict[str, Any]],
    invoke: Callable[[str], Awaitable[Any]],
    *,
    is_parallel_safe: Callable[[str], bool],
) -> Dict[str, Any]:
    """Execute one dependency layer using the declared concurrency policy.

    Args:
        layer: Task IDs whose dependencies have already completed.
        subtasks: Mapping from task ID to task definition.
        invoke: Async callback that executes one task ID.
        is_parallel_safe: Predicate deciding whether an agent may run concurrently.

    Returns:
        A mapping from task ID to its execution result.
    """
    results: Dict[str, Any] = {}
    serial_task_ids = [
        task_id
        for task_id in layer
        if not is_parallel_safe(subtasks.get(task_id, {}).get("agent", ""))
    ]
    parallel_task_ids = [
        task_id for task_id in layer if task_id not in serial_task_ids
    ]

    for task_id in serial_task_ids:
        results[task_id] = await invoke(task_id)

    if len(parallel_task_ids) == 1:
        task_id = parallel_task_ids[0]
        results[task_id] = await invoke(task_id)
    elif parallel_task_ids:
        layer_results = await asyncio.gather(
            *[invoke(task_id) for task_id in parallel_task_ids]
        )
        for task_id, task_result in zip(parallel_task_ids, layer_results):
            results[task_id] = task_result
    return results


class CentralOrchestrator:
    """Coordinate the complete centralized travel-planning workflow."""

    def __init__(self, llm: Optional[ChatOpenAI] = None, enable_memory: bool = True) -> None:
        """Initialize the model, planner dependencies, router, and memory."""
        self.llm = llm or create_llm()
        self.agent_registry = DomainAgentRegistry()
        self.agent_routing = AgentRouter(self.llm, self.agent_registry)
        self.system_prompt = CENTRAL_AGENT_SYSTEM_PROMPT

        self.memory_adapter: Optional[MemoryAdapter] = None
        if enable_memory:
            try:
                self.memory_adapter = MemoryAdapter(
                    user_id="central_agent_user",
                    persist_directory=str(CHROMA_DIR),
                    llm=self.llm,
                )
                self._log("✓ 长期记忆系统已启用（Chapter-3）")
            except Exception as exc:
                self._log(f"⚠️ 长期记忆初始化失败: {exc}")
        else:
            self._log("ℹ️ 长期记忆已禁用")

    def _log(self, *args, **kwargs) -> None:
        """Print a flushed orchestration log message."""
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)

    async def decide_request_route(self, user_query: str) -> Dict[str, str]:
        """Classify a request and answer directly when it is not travel-related.

        Args:
            user_query: Original user request.

        Returns:
            A dictionary containing ``route`` set to ``direct`` or ``travel``
            and the direct response when applicable.
        """
        decision = await invoke_json(
            self.llm,
            user_query,
            system_prompt=self.system_prompt,
            require_object=True,
        )
        route = str(decision.get("route") or "").strip()
        response = str(decision.get("response") or "").strip()
        if route == "direct":
            if not response:
                raise ValueError("Direct route requires a non-empty response.")
            return {"route": route, "response": response}
        if route == "travel":
            return {"route": route, "response": ""}
        raise ValueError(f"Unsupported request route: {route!r}")

    async def process_request(self, user_query: str, thread_id: str = "default") -> Dict[str, Any]:
        """Execute request classification and, when needed, travel planning.

        Args:
            user_query: Natural-language user request.
            thread_id: Conversation identifier used for memory and checkpoints.

        Returns:
            A dictionary containing the route, execution plan, subtask results,
            and final response.
        """
        self._log("=" * 80)
        self._log(f"📥 用户请求: {user_query.strip()}")
        self._log("=" * 80)

        # Request routing: classify direct versus travel requests before planning.
        self._log("\n🧭 [中心编排] 判断请求类型...")
        request_route = await self.decide_request_route(user_query)
        if request_route["route"] == "direct":
            final_response = request_route["response"]
            self._log("✓ 非旅行任务，直接回复")
            self._log("\n" + "=" * 80)
            self._log("📋 最终回复")
            self._log("=" * 80)
            self._log(final_response)
            self._log("=" * 80)
            return {
                "route": "direct",
                "execution_plan": None,
                "subtask_results": {},
                "final_response": final_response,
            }
        self._log("✓ 旅行任务，进入多智能体编排流程")

        # Fact survey: reuse the Chapter 2 structured fact-survey framework.
        self._log("\n🔍 [Ch2] 思维链预调查...")
        try:
            pre_survey = await survey_facts(self.llm, user_query)
        except Exception as exc:
            self._log(f"⚠️ 预调查失败，已回退简化预调查: {type(exc).__name__}: {exc}")
            pre_survey = {
                "given_facts": [user_query.strip()],
                "facts_to_lookup": [],
                "facts_to_derive": [],
                "educated_guesses": [],
                "raw_text": user_query.strip(),
            }
        self._log("✓ 预调查完成")
        self._log(json.dumps({k: v for k, v in pre_survey.items() if k != "raw_text"}, ensure_ascii=False, indent=2))

        # Memory retrieval: reuse the Chapter 3 long-term memory search.
        memories: List[Dict[str, Any]] = []
        if self.memory_adapter:
            try:
                hits = await self.memory_adapter.search_memories_async(user_query)
                memories = self.memory_adapter.format_memories_for_plan(hits)
                self._log(f"\n🧠 [Ch3] 检索到 {len(memories)} 条相关记忆")
                if memories:
                    self._log(json.dumps(memories, ensure_ascii=False, indent=2))
                else:
                    self._log("  （暂无历史记忆，将仅使用当前对话信息）")
            except Exception as exc:
                self._log(
                    f"\n⚠️ [Ch3] 记忆检索失败，继续无记忆执行: "
                    f"{type(exc).__name__}: {exc}"
                )

        # Task planning: reuse Chapter 4 decomposition and dependency analysis.
        self._log("\n📋 [Ch4] 任务拆解 → 依赖分析 → 执行顺序...")
        task_plan = await create_task_plan(
            self.llm,
            user_query,
            pre_survey,
            memories,
            self.agent_registry,
        )
        self._log(
            f"✓ 共 {len(task_plan['tasks'])} 个任务，"
            f"执行顺序: {' → '.join(task_plan['execution_order'])}"
        )

        # Agent routing: assign one domain agent and tool parameters per task.
        self._log("\n🧩 分配领域智能体...")
        depends_map = {
            task["task_id"]: list(task.get("depends_on") or [])
            for task in task_plan["tasks"]
        }
        assigned_tasks = await self.agent_routing.route_to_agents(
            task_plan["tasks"],
            task_plan["execution_order"],
            depends_map,
            pre_survey,
        )
        execution_plan = {
            "pre_survey": task_plan["pre_survey"],
            "pre_survey_raw": task_plan["pre_survey_raw"],
            "retrieved_memories": task_plan["retrieved_memories"],
            "total_goal": task_plan["total_goal"],
            "plan_fallback": task_plan["plan_fallback"],
            "subtasks": assigned_tasks,
            "execution_order": task_plan["execution_order"],
        }
        self._log(f"✓ 共 {len(execution_plan['subtasks'])} 个子任务")
        self._log(f"  执行顺序: {' → '.join(execution_plan['execution_order'])}")
        self._log(json.dumps(execution_plan, ensure_ascii=False, indent=2))

        # Domain execution: reuse Chapter 5 search capabilities through agents.
        self._log("\n⚙️ [Ch5+] 执行领域智能体...")
        run_tag = uuid.uuid4().hex[:8]
        subtask_results = await self.execute_plan(
            execution_plan,
            thread_id,
            run_tag,
        )

        # Final response: select the planned itinerary result from task outputs.
        self._log("\n📝 生成最终旅行规划...")
        final_response = self.compose_final_response(
            subtask_results,
            execution_plan.get("execution_order", []),
        )

        # Memory write: reuse the Chapter 3 turn and preference ingestion flow.
        if self.memory_adapter:
            try:
                self.memory_adapter.record_turn(
                    thread_id,
                    user_query,
                    final_response,
                )
                await self.memory_adapter.ingest(
                    f"用户请求: {user_query.strip()}\n"
                    f"偏好摘要: {final_response[:500]}",
                    memory_type="preference",
                )
            except Exception as exc:
                self._log(
                    f"⚠️ [Ch3] 记忆写入失败，不影响本次回复: "
                    f"{type(exc).__name__}: {exc}"
                )

        self._log("\n" + "=" * 80)
        self._log("✅ 全部处理完成")
        self._log("=" * 80)

        return {
            "route": "travel",
            "execution_plan": execution_plan,
            "subtask_results": subtask_results,
            "final_response": final_response,
        }

    async def execute_plan(
        self,
        execution_plan: Dict[str, Any],
        thread_id: str,
        run_tag: str,
        *,
        message_records: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Execute all planned subtasks layer by layer."""
        results: Dict[str, Any] = {}
        subtasks = {t["task_id"]: t for t in execution_plan.get("subtasks", [])}

        layers = self._topological_layers(execution_plan)
        for layer in layers:
            layer_out = await run_task_layer(
                layer,
                subtasks,
                lambda tid: self._invoke_domain_agent(
                    subtasks[tid],
                    results,
                    thread_id,
                    run_tag,
                    message_records,
                ),
                is_parallel_safe=self.agent_registry.is_parallel_safe,
            )
            results.update(layer_out)

        return results

    def _topological_layers(self, execution_plan: Dict[str, Any]) -> List[List[str]]:
        """按依赖关系分层，同层可并行"""
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

    async def _invoke_domain_agent(
        self,
        task: Dict[str, Any],
        prior_results: Dict[str, Any],
        thread_id: str,
        run_tag: str,
        message_records: Optional[List[Dict[str, Any]]],
    ) -> Any:
        """Invoke one routed domain agent and parse its structured result."""
        task_id = task["task_id"]
        agent_name = task.get("agent")
        description = task.get("description", "")

        if not agent_name or task.get("routing_error"):
            err = task.get("routing_error") or "unrouted"
            self._log(f"\n  ⏭️ {task_id}: 路由未完成（{err}），已跳过")
            self._log(f"     {description[:60]}...")
            return {
                "task_id": task_id,
                "task_description": description,
                "agent": None,
                "tool_data": {"error": err, "message": "子任务未由路由 LLM 成功分配 Agent"},
                "agent_summary": "",
            }

        self._log(f"\n  🔄 {task_id}: {description[:60]}...")
        self._log(f"     → {agent_name}")

        user_message = build_domain_agent_user_message(task, prior_results)
        if message_records is not None:
            message_records.append(
                {
                    "task_id": task_id,
                    "agent": agent_name,
                    "message": user_message,
                }
            )
        agent_thread = f"{thread_id}_{run_tag}_{task_id}"
        try:
            agent = get_domain_agent(agent_name)
            state = await agent.ainvoke(
                {"messages": [("user", user_message)]},
                {"configurable": {"thread_id": agent_thread}},
            )
            result = parse_domain_agent_invoke_result(
                state,
                task_id=task_id,
                agent_name=agent_name,
            )
            result["task_description"] = description
        except Exception as exc:
            self._log(
                f"     ✗ 失败: {type(exc).__name__}: {exc}",
                flush=True,
            )
            result = {
                "task_id": task_id,
                "task_description": description,
                "agent": agent_name,
                "tool_data": {
                    "error": "domain_agent_failed",
                    "message": f"{type(exc).__name__}: {exc}",
                },
                "agent_summary": "",
            }

        self._log(f"     ✓ 完成")
        self._log("     " + "-" * 56)
        try:
            rendered_result = json.dumps(result, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            rendered_result = str(result)
        for line in rendered_result.splitlines():
            self._log(f"     {line}")
        self._log("     " + "-" * 56)
        return result

    def compose_final_response(
        self,
        results: Dict[str, Any],
        execution_order: List[str],
    ) -> str:
        """Select the final response from direct or ordered task results."""
        is_single_result = len(results) == 1
        if is_single_result:
            final_text = direct_response_from_results(results)
            self._log("  ✓ 单任务查询，直接使用领域智能体回复")
        else:
            final_result = next(
                (
                    results[task_id]
                    for task_id in reversed(execution_order)
                    if isinstance(results.get(task_id), dict)
                    and (results[task_id].get("agent_summary") or "").strip()
                ),
                None,
            )
            if final_result:
                final_text = final_result["agent_summary"].strip()
            else:
                summaries = [
                    (result.get("agent_summary") or "").strip()
                    for result in results.values()
                    if isinstance(result, dict)
                    and (result.get("agent_summary") or "").strip()
                ]
                final_text = "\n\n".join(summaries) or "未获得可用的领域结果。"

        title = "📋 最终回复" if is_single_result else "📋 最终旅行规划"
        self._log("\n" + "=" * 80)
        self._log(title)
        self._log("=" * 80)
        self._log(final_text)
        self._log("=" * 80)
        return final_text


async def main() -> None:
    """Run the default centralized travel-planning example."""
    orchestrator = CentralOrchestrator()
    query = """
你能帮我规划一个五一假期的多城市旅行吗？我还没想好行程顺序……
我想从北京出发，大概是上海、苏州、杭州这几个地方？需要包含行程路线、酒店推荐、
天气情况和美食攻略。我喜欢住安静的酒店，预算每晚不超过800元。
"""
    result = await orchestrator.process_request(query, thread_id="central_orchestration")
    print(f"\n子任务数: {len(result['subtask_results'])}")


if __name__ == "__main__":
    asyncio.run(main())
