"""
规划流水线：借鉴 Chapter-6/langgraph 的 build_plan + execute_layer 设计。

流程：pre_survey → build_plan → 按依赖分层 execute_layer → aggregate
用于复合旅行请求；单任务仍走 local_supervisor 的 Supervisor handoff。
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from _ch6_loader import load_ch6_module
from sub_agents import SubAgentFactory
from travel_common import build_trip_date_anchor

_central = load_ch6_module("central_orchestrator")
_planner_mod = load_ch6_module("task_planner")
_agg = load_ch6_module("aggregation_helpers")
_prompts = load_ch6_module("prompts")

SubAgentRegistry = _central.SubAgentRegistry
TaskPlanner = _planner_mod.TaskPlanner
is_single_direct_response = _agg.is_single_direct_response
direct_response_from_results = _agg.direct_response_from_results
AGGREGATION_PROMPT = _prompts.AGGREGATION_PROMPT

# supervisor 节点名（用于终端日志，与 local_supervisor.AGENT_SPECS 一致）
_FACTORY_TO_NODE: Dict[str, str] = {
    "WeatherAgent": "weather_agent",
    "AttractionAgent": "attraction_agent",
    "HotelAgent": "hotel_agent",
    "RestaurantAgent": "restaurant_agent",
    "FlightAgent": "flight_agent",
    "ItineraryAgent": "itinerary_agent",
}


def _topological_layers(execution_plan: Dict[str, Any]) -> List[List[str]]:
    subtasks = {t["task_id"]: t for t in execution_plan.get("subtasks", [])}
    order = execution_plan.get("execution_order", list(subtasks.keys()))
    done: set = set()
    layers: List[List[str]] = []
    remaining = list(order)
    while remaining:
        layer = [
            tid
            for tid in remaining
            if all(d in done for d in subtasks[tid].get("depends_on", []))
        ]
        if not layer:
            layer = [remaining[0]]
        layers.append(layer)
        for tid in layer:
            done.add(tid)
            remaining.remove(tid)
    return layers


def _agent_user_message(query: str, date_anchor: Optional[Dict[str, Any]] = None) -> str:
    if not query.strip():
        return "请根据上下文完成任务"
    parts = [query.strip()]
    if date_anchor:
        parts.append(date_anchor["anchor_block"])
    else:
        today = datetime.now().strftime("%Y-%m-%d")
        parts.append(
            f"[系统参考：当前日期 {today}。"
            f"get_weather 的 date 可传 YYYY-MM-DD 或相对词]"
        )
    return "\n\n".join(parts)


async def _invoke_sub_agent(
    task: Dict[str, Any],
    prior_results: Dict[str, Any],
    thread_id: str,
    date_anchor: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    task_id = task["task_id"]
    agent_name = task.get("agent", "ItineraryAgent")
    description = task.get("description", "")
    node_name = _FACTORY_TO_NODE.get(agent_name, agent_name)

    print(f"\n  ▶ [{node_name}] 执行 {agent_name}...", flush=True)

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
        {"messages": [("user", _agent_user_message("\n".join(query_parts), date_anchor))]},
        {"configurable": {"thread_id": f"{thread_id}_{task_id}_{uuid.uuid4().hex[:6]}"}},
    )

    tool_outputs: List[Any] = []
    agent_text = ""
    for msg in state.get("messages", []):
        if not hasattr(msg, "type"):
            continue
        if msg.type == "tool" and getattr(msg, "content", None):
            try:
                tool_outputs.append(json.loads(msg.content))
            except (json.JSONDecodeError, TypeError):
                tool_outputs.append(msg.content)
        elif msg.type == "ai" and getattr(msg, "content", None):
            agent_text = msg.content

    print(f"  ✓ [{node_name}] 完成", flush=True)
    return {
        "task_id": task_id,
        "agent": agent_name,
        "status": "completed",
        "tool_data": tool_outputs[-1] if tool_outputs else None,
        "agent_summary": agent_text,
    }


class PlannedPipeline:
    """TaskPlanner 驱动的固定编排（无 Supervisor handoff 循环）。"""

    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm = llm
        self.planner = TaskPlanner(llm, SubAgentRegistry())

    async def _build_plan_context(self, user_query: str) -> Dict[str, Any]:
        date_anchor = build_trip_date_anchor(user_query)
        enriched_query = f"{user_query.strip()}\n\n{date_anchor['anchor_block']}"

        pre_survey = await self.planner.run_pre_survey(enriched_query)
        plan = await self.planner.build_execution_plan(
            enriched_query,
            pre_survey,
            [],
        )
        for st in plan.get("subtasks", []):
            agent = st.get("agent", "")
            params = dict(st.get("params") or {})
            if agent == "WeatherAgent":
                params["dates"] = date_anchor["trip_dates"]
            elif agent == "FlightAgent" and date_anchor["trip_dates"]:
                params.setdefault("date", date_anchor["trip_dates"][0])
            st["params"] = params

        return {
            "user_query": user_query,
            "date_anchor": date_anchor,
            "enriched_query": enriched_query,
            "plan": plan,
        }

    async def classify_route(self, user_query: str) -> tuple[str, Optional[Dict[str, Any]]]:
        """
        用 TaskPlanner 子任务数量决定路由：
        - >1 个子任务 → planned（返回 plan_ctx，避免重复规划）
        - 否则 → supervisor
        """
        print("\n🧭 任务规划（build_plan 路由决策）...", flush=True)
        ctx = await self._build_plan_context(user_query)
        subtasks = ctx["plan"].get("subtasks", [])
        n = len(subtasks)
        agents = ", ".join(st.get("agent", "?") for st in subtasks) or "无"
        print(
            f"  📅 出行日期: {', '.join(ctx['date_anchor']['trip_dates'])}",
            flush=True,
        )
        print(f"  ✓ 拆解为 {n} 个子任务: {agents}", flush=True)
        if n > 1:
            return "planned", ctx
        return "supervisor", None

    async def run(
        self,
        user_query: str,
        thread_id: str = "planned",
        plan_ctx: Optional[Dict[str, Any]] = None,
    ) -> str:
        print("\n📋 [规划模式] 分层执行 → 聚合", flush=True)

        if plan_ctx is None:
            ctx = await self._build_plan_context(user_query)
        else:
            ctx = plan_ctx

        plan = ctx["plan"]
        date_anchor = ctx["date_anchor"]
        enriched_query = ctx["enriched_query"]
        layers = _topological_layers(plan)
        subtasks = {t["task_id"]: t for t in plan.get("subtasks", [])}

        print(
            f"  ✓ 共 {len(plan.get('subtasks', []))} 个子任务，"
            f"{len(layers)} 个执行层",
            flush=True,
        )
        print(
            f"  执行顺序: {' → '.join(plan.get('execution_order', []))}",
            flush=True,
        )

        results: Dict[str, Any] = {}
        for idx, layer in enumerate(layers):
            tasks = [subtasks[tid] for tid in layer]
            if len(tasks) == 1:
                layer_results = [
                    await _invoke_sub_agent(
                        tasks[0], results, thread_id, date_anchor
                    )
                ]
            else:
                layer_results = await asyncio.gather(*[
                    _invoke_sub_agent(t, results, thread_id, date_anchor)
                    for t in tasks
                ])
            for res in layer_results:
                results[res["task_id"]] = res

        print("\n📝 聚合结果...", flush=True)
        if is_single_direct_response(results):
            final_text = direct_response_from_results(results)
            print("  ✓ 单任务直达（跳过旅行规划模板）", flush=True)
        else:
            prompt = AGGREGATION_PROMPT.format(
                user_query=enriched_query,
                pre_survey=json.dumps(plan.get("pre_survey", {}), ensure_ascii=False, indent=2),
                memories=json.dumps(plan.get("retrieved_memories", []), ensure_ascii=False, indent=2),
                total_goal=plan.get("total_goal", ""),
                results=json.dumps(results, ensure_ascii=False, indent=2),
            )
            prompt += (
                f"\n\n## 日期约束（必须遵守）\n"
                f"- 今天是 {date_anchor['today']}\n"
                f"- 行程日期仅限: {', '.join(date_anchor['trip_dates'])}\n"
                f"- 禁止在回复中出现 2024 等与上述不一致的年份\n"
                f"- 若子任务未返回有效天气数据，说明无法查询并给出建议，不要编造历史气象公报\n"
            )
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            final_text = (response.content or "").strip()
            print("  ✓ 聚合完成", flush=True)

        return final_text
