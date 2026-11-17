"""Central orchestrator for Chapter 6 multi-agent travel planning.

The controller coordinates pre-survey, long-term memory, task decomposition,
dependency analysis, domain-agent execution, and final aggregation.

:author: zhanghong, liuhui
:date: 2026-11-17
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional

from langchain_openai import ChatOpenAI

from config import CHROMA_DIR, load_project_dotenv
from memory_adapter import MemoryAdapter
from domain_agents import (
    DomainAgentRegistry,
    build_domain_agent_user_message,
    get_domain_agent,
    inject_itinerary_context,
    parse_domain_agent_invoke_result,
)
from llm import create_llm, invoke_chat
from task_planner import TaskPlanner, collect_cities_from_subtasks
from travel_planning import build_trip_date_anchor_async

load_project_dotenv()

QualityEvaluator = Callable[[str, Any], str]

AGGREGATION_PROMPT = """你是专业的旅行助手。请根据以下信息，针对用户的原始请求生成回复。

## 用户原始请求
{user_query}

## 思维链预调查
{pre_survey}

## 检索到的长期记忆
{memories}

## 总体目标
{total_goal}

## 子任务执行结果
{results}

## 子任务结果字段说明
每项含 `tool_data`（工具原始 JSON，唯一可信事实来源）与 `agent_summary_note`（领域 Agent 文字说明，仅供参考）。若二者冲突，必须以 `tool_data` 为准。

## 要求
1. 综合所有子任务中 tool_data 已返回的真实字段。
2. 严格匹配用户请求范围，不擅自扩展。
3. 某类 tool_data 为空时说明缺失，不编造 POI、价格、天气或时刻表。
4. 使用中文，语气友好专业。
5. 使用已给出和工具返回的日期，不臆造年份。
6. 不得引用 tool_data 中不存在的事实。
7. 数据质量为 error、empty 或 invalid_poi 时明确说明工具未返回有效数据。
8. 酒店或餐厅名称为行政区划名等明显无效 POI 时不得写入推荐列表。
9. 天气工具失败时不得输出具体气温或降水概率。
10. 工具数据缺失时只说明缺失并给出补查建议。
"""


def is_single_direct_response(results: Dict[str, Any]) -> bool:
    """Return whether exactly one subtask should answer directly."""
    return len(results) == 1


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


def format_results_for_aggregation(
    results: Dict[str, Any],
    *,
    quality_evaluator: QualityEvaluator,
    summary_max_chars: int = 400,
) -> str:
    """Serialize subtask results with their quality markers.

    Args:
        results: Mapping from task ID to normalized subtask result.
        quality_evaluator: Callable resolving an agent's quality policy.
        summary_max_chars: Maximum retained length of each agent summary.

    Returns:
        A JSON string containing tool data, quality markers, and summaries.
    """
    blocks = []
    for task_id, result in results.items():
        if not isinstance(result, dict):
            blocks.append({"task_id": task_id, "raw": result})
            continue
        agent_name = result.get("agent", "")
        tool_data = result.get("tool_data")
        summary = (result.get("agent_summary") or "").strip()
        if len(summary) > summary_max_chars:
            summary = summary[:summary_max_chars] + "…"
        quality = quality_evaluator(agent_name, tool_data)
        summary_note = summary or "（无文字总结，请以 tool_data 为准）"
        if quality != "ok":
            summary_note = "（该子任务工具未返回有效数据，请忽略文字补充，仅以 tool_data 错误信息为准）"
        blocks.append(
            {
                "task_id": task_id,
                "agent": agent_name,
                "data_quality": quality,
                "tool_data": tool_data,
                "agent_summary_note": summary_note,
            }
        )
    return json.dumps(blocks, ensure_ascii=False, indent=2)


def format_date_anchor_constraint(date_anchor: Dict[str, Any] | None) -> str:
    """Build the hard date constraint used during final aggregation."""
    if not date_anchor:
        return ""
    trip_dates = date_anchor.get("trip_dates") or []
    trip_range = date_anchor.get("trip_range") or (
        trip_dates[0] if trip_dates else ""
    )
    dates_line = ", ".join(trip_dates) if trip_dates else "（未解析）"
    return (
        "\n\n## 日期约束（必须遵守，优于 pre_survey 中的任何示例年份）\n"
        f"- 今天是 {date_anchor.get('today')}\n"
        f"- 本请求出行日期: {trip_range}（{dates_line}）\n"
        "- 用户说的「下周/明天」等相对日期，只使用上述日期，禁止推断为 2024 或其他年份\n"
        "- 子任务 tool_data 中的日期若与上述一致，视为正确；不要捏造「系统时间基准与用户不一致」类说法\n"
        "- 无 tool_data 支撑时说明缺失，不要用其他年份的数据填补"
    )


def build_aggregation_prompt(
    *,
    user_query: str,
    execution_plan: Dict[str, Any],
    results: Dict[str, Any],
    quality_evaluator: QualityEvaluator,
    recent_dialogue: str = "",
    date_anchor: Dict[str, Any] | None = None,
) -> str:
    """Build the final aggregation task prompt.

    Args:
        user_query: Original user request.
        execution_plan: Planner output containing goals and context.
        results: Completed subtask results.
        quality_evaluator: Callable resolving an agent's quality policy.
        recent_dialogue: Optional recent conversation history.
        date_anchor: Optional authoritative trip-date context.

    Returns:
        A complete task prompt for the final aggregation model.
    """
    query_for_agg = execution_plan.get("enriched_query") or user_query
    anchor = date_anchor or execution_plan.get("date_anchor")
    prompt = AGGREGATION_PROMPT.format(
        user_query=query_for_agg,
        pre_survey=json.dumps(
            execution_plan.get("pre_survey", {}),
            ensure_ascii=False,
            indent=2,
        ),
        memories=json.dumps(
            execution_plan.get("retrieved_memories", []),
            ensure_ascii=False,
            indent=2,
        ),
        total_goal=execution_plan.get("total_goal", ""),
        results=format_results_for_aggregation(
            results,
            quality_evaluator=quality_evaluator,
        ),
    )
    prompt += format_date_anchor_constraint(anchor)
    dialogue = (recent_dialogue or "").strip()
    if dialogue:
        prompt += f"\n\n## 本线程最近对话（Chapter-3 短期记忆）\n{dialogue}"
    return prompt


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


def build_central_agent_system_prompt(agent_registry: DomainAgentRegistry) -> str:
    """Build the orchestration role prompt from the live capability catalog.

    Args:
        agent_registry: Registry containing the currently available domain agents.

    Returns:
        A system prompt describing the central orchestration role and workflow.
    """
    agent_catalog = agent_registry.get_all_agents_text()
    return f"""你是「旅行规划中心智能体」，负责协调以下能力：

## 1. 思维链预调查
在正式规划前，将请求信息分为已给出事实、需要查阅的事实、需要推导的事实和有根据的猜测。

## 2. 长期记忆
检索与用户偏好相关的历史记忆，并禁止编造未检索到的内容。

## 3. 任务拆解与依赖排序
按领域智能体能力拆解子任务，根据技能输入、输出定义识别依赖并生成执行顺序。

## 4. 可用领域智能体
{agent_catalog}

## 工作流程
用户请求 → 预调查 → 记忆检索 → 任务拆解 → 依赖排序 → 领域智能体调度 → 聚合回复 → 写入记忆

作为中心控制器，你需要确保每个子任务由正确的领域智能体执行，并将结果综合为用户可读的旅行规划。
"""


class CentralOrchestrator:
    """Coordinate the complete centralized travel-planning workflow."""

    def __init__(self, llm: Optional[ChatOpenAI] = None, enable_memory: bool = True) -> None:
        """Initialize the model, planner, agent registry, and optional memory."""
        self.llm = llm or create_llm()
        self.agent_registry = DomainAgentRegistry()
        self.planner = TaskPlanner(self.llm, self.agent_registry)
        self.system_prompt = build_central_agent_system_prompt(self.agent_registry)

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

    def _apply_date_anchor_to_subtasks(
        self,
        execution_plan: Dict[str, Any],
        date_anchor: Dict[str, Any],
    ) -> None:
        """按 Agent inputSchema 通用注入统一出行日期 params。"""
        trip_dates = date_anchor.get("trip_dates") or []
        if not trip_dates:
            return
        all_cities = collect_cities_from_subtasks(execution_plan.get("subtasks", []))
        for st in execution_plan.get("subtasks", []):
            agent = st.get("agent", "")
            input_fields = self.agent_registry.get_agent_input_fields(agent)
            params = dict(st.get("params") or {})

            if "dates" in input_fields:
                params.setdefault("dates", trip_dates)
            if "date" in input_fields:
                params.setdefault("date", date_anchor["trip_dates"][0])
            if "days" in input_fields:
                params.setdefault("days", len(trip_dates))
            if "cities" in input_fields and all_cities:
                params.setdefault("cities", all_cities)

            st["params"] = params

    async def process_request(self, user_query: str, thread_id: str = "default") -> Dict[str, Any]:
        """Execute the full centralized workflow for one user request.

        Args:
            user_query: Natural-language travel request.
            thread_id: Conversation identifier used for memory and checkpoints.

        Returns:
            A dictionary containing the execution plan, subtask results, and
            final response.
        """
        self._log("=" * 80)
        self._log(f"📥 用户请求: {user_query.strip()}")
        self._log("=" * 80)

        date_anchor = await build_trip_date_anchor_async(user_query, llm=self.llm)
        enriched_query = f"{user_query.strip()}\n\n{date_anchor['anchor_block']}"
        self._log(
            f"\n📅 日期锚定: 今天 {date_anchor['today']}；"
            f"出行 {date_anchor['trip_range']}（{', '.join(date_anchor['trip_dates'])}）"
        )
        # --- Chapter-2: 思维链预调查 ---
        self._log("\n🔍 [Ch2] 思维链预调查...")
        try:
            pre_survey = await self.planner.run_pre_survey(enriched_query)
        except Exception as exc:
            self._log(f"⚠️ 预调查失败，已回退简化预调查: {type(exc).__name__}: {exc}")
            pre_survey = {
                "given_facts": [user_query.strip()],
                "facts_to_lookup": [],
                "facts_to_derive": [],
                "educated_guesses": [],
                "trip_cities": [],
                "trip_dates": date_anchor.get("trip_dates") or [],
                "raw_text": user_query.strip(),
            }
        self._log("✓ 预调查完成")
        self._log(json.dumps({k: v for k, v in pre_survey.items() if k != "raw_text"}, ensure_ascii=False, indent=2))

        # --- Chapter-3: 长期记忆检索 ---
        memories: List[Dict[str, Any]] = []
        if self.memory_adapter:
            hits = await self.memory_adapter.search_memories_async(user_query)
            memories = self.memory_adapter.format_memories_for_plan(hits)
            self._log(f"\n🧠 [Ch3] 检索到 {len(memories)} 条相关记忆")
            if memories:
                self._log(json.dumps(memories, ensure_ascii=False, indent=2))
            else:
                self._log("  （暂无历史记忆，将仅使用当前对话信息）")

        # --- Chapter-4 + 路由: 拆解 → 依赖 → 选 Agent ---
        self._log("\n📋 [Ch4] 任务拆解 → 依赖分析 → 领域智能体路由...")
        execution_plan = await self.planner.build_execution_plan(enriched_query, pre_survey, memories)
        execution_plan["date_anchor"] = date_anchor
        execution_plan["enriched_query"] = enriched_query
        self._apply_date_anchor_to_subtasks(execution_plan, date_anchor)
        self._log(f"✓ 共 {len(execution_plan['subtasks'])} 个子任务")
        self._log(f"  执行顺序: {' → '.join(execution_plan['execution_order'])}")
        cv = execution_plan.get("city_validation") or {}
        if cv.get("expected_cities"):
            self._log(
                f"  城市校验: 预调查={cv.get('expected_cities')} "
                f"对齐={cv.get('aligned')} "
                f"补全={cv.get('patched')}"
                + (f" 缺失={cv.get('missing_cities')}" if cv.get("missing_cities") else "")
            )
        self._log(json.dumps(execution_plan, ensure_ascii=False, indent=2))

        # --- Chapter-5 扩展: 领域智能体执行 ---
        self._log("\n⚙️ [Ch5+] 执行领域智能体...")
        run_tag = uuid.uuid4().hex[:8]
        subtask_results = await self._execute_subtasks(
            execution_plan,
            thread_id,
            date_anchor,
            run_tag,
        )

        # --- 聚合最终回复 ---
        self._log("\n📝 聚合结果，生成最终旅行规划...")
        final_response = await self._aggregate_results(
            user_query, execution_plan, subtask_results, thread_id
        )

        # --- Chapter-3: 写入记忆 ---
        if self.memory_adapter:
            self.memory_adapter.record_turn(thread_id, user_query, final_response)
            await self.memory_adapter.ingest(
                f"用户请求: {user_query.strip()}\n偏好摘要: {final_response[:500]}",
                memory_type="preference",
            )

        self._log("\n" + "=" * 80)
        self._log("✅ 全部处理完成")
        self._log("=" * 80)

        return {
            "execution_plan": execution_plan,
            "subtask_results": subtask_results,
            "final_response": final_response,
        }

    async def _execute_subtasks(
        self,
        execution_plan: Dict[str, Any],
        thread_id: str,
        date_anchor: Dict[str, Any],
        run_tag: str,
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
                    date_anchor,
                    run_tag,
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
        date_anchor: Dict[str, Any],
        run_tag: str,
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
                "agent": None,
                "tool_data": {"error": err, "message": "子任务未由路由 LLM 成功分配 Agent"},
                "agent_summary": "",
            }

        self._log(f"\n  🔄 {task_id}: {description[:60]}...")
        self._log(f"     → {agent_name}")

        task = inject_itinerary_context(task, prior_results)
        user_message = build_domain_agent_user_message(task, prior_results)
        user_message = f"{user_message}\n\n{date_anchor['anchor_block']}"
        agent = get_domain_agent(agent_name)
        agent_thread = f"{thread_id}_{run_tag}_{task_id}"
        state = await agent.ainvoke(
            {"messages": [("user", user_message)]},
            {"configurable": {"thread_id": agent_thread}},
        )

        result = parse_domain_agent_invoke_result(
            state,
            task_id=task_id,
            agent_name=agent_name,
        )

        self._log(f"     ✓ 完成")
        self._log("     " + "-" * 56)
        for line in self._format_result(result).splitlines():
            self._log(f"     {line}")
        self._log("     " + "-" * 56)
        return result

    def _format_result(self, result: Any) -> str:
        """Format a subtask result for readable execution logs."""
        try:
            return json.dumps(result, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return str(result)

    async def _aggregate_results(
        self,
        user_query: str,
        execution_plan: Dict[str, Any],
        results: Dict[str, Any],
        thread_id: str,
    ) -> str:
        """Build the final response from direct or aggregated subtask results."""
        if is_single_direct_response(results):
            final_text = direct_response_from_results(results)
            self._log("  ✓ 单任务查询，直接使用领域智能体回复（跳过旅行规划聚合）")
        else:
            recent_dialogue = ""
            if self.memory_adapter:
                recent_dialogue = self.memory_adapter.short_term.format_recent(thread_id)
                self._log("  🧠 聚合使用 AGGREGATION_PROMPT + 长期/短期记忆上下文")
            prompt = build_aggregation_prompt(
                user_query=user_query,
                execution_plan=execution_plan,
                results=results,
                recent_dialogue=recent_dialogue,
                date_anchor=execution_plan.get("date_anchor"),
                quality_evaluator=self.agent_registry.evaluate_result,
            )
            final_text = await invoke_chat(
                self.llm,
                prompt,
                system_prompt=self.system_prompt,
            )

        title = "📋 最终回复" if is_single_direct_response(results) else "📋 最终旅行规划"
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
大概是上海、苏州、杭州这几个地方？需要包含行程路线、酒店推荐、
天气情况和美食攻略。我喜欢住安静的酒店，预算每晚不超过800元。
"""
    result = await orchestrator.process_request(query, thread_id="central_orchestration")
    print(f"\n子任务数: {len(result['subtask_results'])}")


if __name__ == "__main__":
    asyncio.run(main())
