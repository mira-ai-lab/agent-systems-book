"""
Chapter-6: 中心智能体 — 整合 Ch2 思维链 + Ch3 记忆 + Ch4 任务拆解 + Ch5 多子智能体
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from chapter6.paths import CHROMA_DIR, load_project_dotenv
from aggregation_helpers import (
    build_aggregation_prompt,
    direct_response_from_results,
    inject_itinerary_params,
    is_single_direct_response,
)
from execution_helpers import run_task_layer
from memory_system import LongTermMemory
from prompts import CENTRAL_AGENT_SYSTEM_PROMPT
from sub_agents import build_sub_agent_user_message, parse_sub_agent_invoke_result
from task_planner import TaskPlanner, collect_cities_from_subtasks
from travel_common import build_trip_date_anchor_async

load_project_dotenv()


class SubAgentRegistry:
    """子智能体注册表（Chapter-5 HotelAgent 扩展为 6 个专业 Agent）"""

    def __init__(self) -> None:
        self.agents = {
            "WeatherAgent": {
                "name": "WeatherAgent",
                "description": "查询指定城市、日期的天气预报，提供温度、天气状况和出行建议",
                "skills": [{
                    "name": "get_weather",
                    "inputSchema": ["city", "date", "cities", "dates"],
                    "outputSchema": ["forecast", "temperature", "condition", "advice"],
                }],
            },
            "AttractionAgent": {
                "name": "AttractionAgent",
                "description": "根据城市、兴趣偏好推荐旅游景点和必去打卡地",
                "skills": [{
                    "name": "recommend_attractions",
                    "inputSchema": ["city", "preferences", "limit"],
                    "outputSchema": ["attraction_list", "ratings", "locations"],
                }],
            },
            "HotelAgent": {
                "name": "HotelAgent",
                "description": "根据位置、预算、偏好（近景区/安静/品牌）推荐酒店；地图关键词与主观偏好分离（Chapter-5）",
                "skills": [{
                    "name": "recommend_hotel",
                    "inputSchema": ["city", "preferences", "budget_cny_per_night_max"],
                    "outputSchema": ["hotels", "prices", "ratings", "locations"],
                }],
            },
            "RestaurantAgent": {
                "name": "RestaurantAgent",
                "description": "根据菜系、位置、预算推荐当地特色餐厅和美食",
                "skills": [{
                    "name": "recommend_restaurant",
                    "inputSchema": ["location", "cuisine", "budget_cny_per_person"],
                    "outputSchema": ["restaurants", "cuisines", "prices", "ratings"],
                }],
            },
            "ItineraryAgent": {
                "name": "ItineraryAgent",
                "description": "根据景点 POI 与地图路线规划生成每日景点游览路线",
                "skills": [{
                    "name": "plan_itinerary",
                    "inputSchema": [
                        "departure_city", "destination_city", "days", "dates",
                        "cities", "attraction_list", "attractions_by_city", "preferences",
                    ],
                    "outputSchema": ["daily_plan", "local_route", "transportation"],
                }],
            },
            "FlightAgent": {
                "name": "FlightAgent",
                "description": "查询出发地到目的地的航班信息、价格和时刻表",
                "skills": [{
                    "name": "search_flights",
                    "inputSchema": ["departure", "arrival", "date"],
                    "outputSchema": ["flights", "prices", "times", "airlines"],
                }],
            },
        }

    def get_all_agents_text(self) -> str:
        return "\n".join(f"- {a['name']}: {a['description']}" for a in self.agents.values())

    def get_agent_parameters_text(self) -> str:
        lines = []
        for info in self.agents.values():
            lines.append(info["name"])
            for skill in info["skills"]:
                lines.append(
                    f"\t{skill['name']}, inputSchema:{skill['inputSchema']}, "
                    f"outputSchema:{skill['outputSchema']}"
                )
        return "\n".join(lines)

    def get_agent_input_fields(self, agent_name: str) -> set[str]:
        """返回某个 Agent 所有技能声明的输入字段，用于通用参数注入。"""
        info = self.agents.get(agent_name) or {}
        fields: set[str] = set()
        for skill in info.get("skills", []):
            for field in skill.get("inputSchema", []):
                fields.add(str(field))
        return fields


class CentralOrchestrator:
    """
    中心智能体编排器

    流程：Ch2 预调查 → Ch3 记忆检索 → Ch4 拆解+依赖 → 子智能体路由 → Ch5 Agent 执行 → 聚合 → Ch3 写记忆
    """

    def __init__(self, llm: Optional[ChatOpenAI] = None, enable_memory: bool = True) -> None:
        self.llm = llm or self._create_default_llm()
        self.agent_registry = SubAgentRegistry()
        self.planner = TaskPlanner(self.llm, self.agent_registry)
        self.system_prompt = CENTRAL_AGENT_SYSTEM_PROMPT

        self.memory_system: Optional[LongTermMemory] = None
        if enable_memory:
            try:
                self.memory_system = LongTermMemory(
                    user_id="central_agent_user",
                    persist_directory=str(CHROMA_DIR),
                    llm=self.llm,
                )
                self._log("✓ 长期记忆系统已启用（Chapter-3）")
            except Exception as exc:
                self._log(f"⚠️ 长期记忆初始化失败: {exc}")
        else:
            self._log("ℹ️ 长期记忆已禁用")

    def _create_default_llm(self) -> ChatOpenAI:
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY")
        return ChatOpenAI(
            model=os.getenv("DASHSCOPE_CHAT_MODEL", "qwen-plus"),
            temperature=0,
            api_key=api_key,
            base_url=os.getenv(
                "DASHSCOPE_CHAT_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ).rstrip("/"),
            http_client=httpx.Client(verify=False),
        )

    def _log(self, *args, **kwargs) -> None:
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

    async def _prewarm_weather_mcp(self, today: str) -> None:
        """Notebook/Windows 下提前拉起 MCP 子进程，避免首个 WeatherAgent 冷启动失败。"""
        import shutil
        from weather_mcp import close_weather_mcp, fetch_weather_via_mcp

        close_weather_mcp()
        npx = shutil.which("npx") or shutil.which("npx.cmd")
        key = (os.getenv("WEATHERAPI_KEY") or "").strip()
        if not npx:
            self._log("⚠️ 未找到 npx，WeatherAgent 将无法使用 MCP（请安装 Node.js 并加入 PATH）")
            return
        if not key:
            self._log("⚠️ WEATHERAPI_KEY 未配置，WeatherAgent 将回退高德/wttr")
            return
        if os.getenv("WEATHER_USE_MCP", "1").strip().lower() in ("0", "false", "no", "off"):
            self._log("ℹ️ WEATHER_USE_MCP=0，已跳过 MCP 预热")
            return
        sample = await fetch_weather_via_mcp("上海", today)
        if sample and not sample.get("error"):
            self._log(f"✓ WeatherAPI MCP 预热成功（{sample.get('data_source')}）")
        else:
            from weather_mcp import get_last_mcp_error
            self._log(
                f"⚠️ WeatherAPI MCP 预热失败: {get_last_mcp_error() or 'unknown'}；"
                "天气将回退高德/wttr"
            )

    async def process_request(self, user_query: str, thread_id: str = "default") -> Dict[str, Any]:
        self._log("=" * 80)
        self._log(f"📥 用户请求: {user_query.strip()}")
        self._log("=" * 80)

        date_anchor = await build_trip_date_anchor_async(user_query, llm=self.llm)
        enriched_query = f"{user_query.strip()}\n\n{date_anchor['anchor_block']}"
        self._log(
            f"\n📅 日期锚定: 今天 {date_anchor['today']}；"
            f"出行 {date_anchor['trip_range']}（{', '.join(date_anchor['trip_dates'])}）"
        )
        await self._prewarm_weather_mcp(date_anchor["today"])

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
        if self.memory_system:
            hits = self.memory_system.search_memories(user_query)
            memories = self.memory_system.format_memories_for_plan(hits)
            self._log(f"\n🧠 [Ch3] 检索到 {len(memories)} 条相关记忆")
            if memories:
                self._log(json.dumps(memories, ensure_ascii=False, indent=2))
            else:
                self._log("  （暂无历史记忆，将仅使用当前对话信息）")

        # --- Chapter-4 + 路由: 拆解 → 依赖 → 选 Agent ---
        self._log("\n📋 [Ch4] 任务拆解 → 依赖分析 → 子智能体路由...")
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

        # --- Chapter-5 扩展: 子智能体执行 ---
        self._log("\n⚙️ [Ch5+] 执行子智能体...")
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
        if self.memory_system:
            self.memory_system.record_turn(thread_id, user_query, final_response)
            await self.memory_system.ingest(
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
        from sub_agents import SubAgentFactory

        results: Dict[str, Any] = {}
        subtasks = {t["task_id"]: t for t in execution_plan.get("subtasks", [])}

        layers = self._topological_layers(execution_plan)
        for layer in layers:
            layer_out = await run_task_layer(
                layer,
                subtasks,
                lambda tid: self._invoke_sub_agent(
                    SubAgentFactory,
                    subtasks[tid],
                    results,
                    thread_id,
                    date_anchor,
                    run_tag,
                ),
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

    async def _invoke_sub_agent(
        self,
        factory: Any,
        task: Dict[str, Any],
        prior_results: Dict[str, Any],
        thread_id: str,
        date_anchor: Dict[str, Any],
        run_tag: str,
    ) -> Any:
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

        task = inject_itinerary_params(task, prior_results)
        user_message = build_sub_agent_user_message(task, prior_results)
        user_message = f"{user_message}\n\n{date_anchor['anchor_block']}"
        agent = factory.get_agent(agent_name)
        agent_thread = f"{thread_id}_{run_tag}_{task_id}"
        state = await agent.ainvoke(
            {"messages": [("user", user_message)]},
            {"configurable": {"thread_id": agent_thread}},
        )

        result = parse_sub_agent_invoke_result(
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
        if is_single_direct_response(results):
            final_text = direct_response_from_results(results)
            self._log("  ✓ 单任务查询，直接使用子智能体回复（跳过旅行规划聚合）")
        else:
            recent_dialogue = ""
            if self.memory_system:
                recent_dialogue = self.memory_system.short_term.format_recent(thread_id)
                self._log("  🧠 聚合使用 AGGREGATION_PROMPT + 长期/短期记忆上下文")
            prompt = build_aggregation_prompt(
                user_query=user_query,
                execution_plan=execution_plan,
                results=results,
                recent_dialogue=recent_dialogue,
                date_anchor=execution_plan.get("date_anchor"),
            )
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            final_text = response.content or ""

        title = "📋 最终回复" if is_single_direct_response(results) else "📋 最终旅行规划"
        self._log("\n" + "=" * 80)
        self._log(title)
        self._log("=" * 80)
        self._log(final_text)
        self._log("=" * 80)
        return final_text


async def main() -> None:
    orchestrator = CentralOrchestrator()
    query = """
你能帮我规划一个五一假期的多城市旅行吗？我还没想好行程顺序……
大概是上海、苏州、杭州这几个地方？需要包含行程路线、酒店推荐、
天气情况和美食攻略。我喜欢住安静的酒店，预算每晚不超过800元。
"""
    result = await orchestrator.process_request(query, thread_id="demo")
    print(f"\n子任务数: {len(result['subtask_results'])}")


if __name__ == "__main__":
    asyncio.run(main())
