"""
Chapter-6: 中心智能体 — 整合 Ch2 思维链 + Ch3 记忆 + Ch4 任务拆解 + Ch5 多子智能体
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent
CHAPTER6_DIR = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CHAPTER6_DIR) not in sys.path:
    sys.path.insert(0, str(CHAPTER6_DIR))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import httpx

from prompts import AGGREGATION_PROMPT, CENTRAL_AGENT_SYSTEM_PROMPT
from aggregation_helpers import (
    MEMORY_AGGREGATION_INSTRUCTION,
    direct_response_from_results,
    is_single_direct_response,
)
from memory_system import LongTermMemory
from task_planner import TaskPlanner

load_dotenv(PROJECT_ROOT / ".env")


class SubAgentRegistry:
    """子智能体注册表（Chapter-5 HotelAgent 扩展为 6 个专业 Agent）"""

    def __init__(self) -> None:
        self.agents = {
            "WeatherAgent": {
                "name": "WeatherAgent",
                "description": "查询指定城市、日期的天气预报，提供温度、天气状况和出行建议",
                "skills": [{
                    "name": "get_weather",
                    "inputSchema": ["city", "date"],
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
                "description": "综合天气、景点、交通、住宿信息，生成详细的每日行程安排",
                "skills": [{
                    "name": "plan_itinerary",
                    "inputSchema": [
                        "departure_city", "destination_city", "days",
                        "weather_summary", "attraction_list", "preferences",
                    ],
                    "outputSchema": ["daily_plan", "transportation", "stay_suggestion"],
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
                    persist_directory=str(CHAPTER6_DIR / "chroma_memory"),
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

    async def process_request(self, user_query: str, thread_id: str = "default") -> Dict[str, Any]:
        self._log("=" * 80)
        self._log(f"📥 用户请求: {user_query.strip()}")
        self._log("=" * 80)

        # --- Chapter-2: 思维链预调查 ---
        self._log("\n🔍 [Ch2] 思维链预调查...")
        pre_survey = await self.planner.run_pre_survey(user_query)
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
        execution_plan = await self.planner.build_execution_plan(user_query, pre_survey, memories)
        self._log(f"✓ 共 {len(execution_plan['subtasks'])} 个子任务")
        self._log(f"  执行顺序: {' → '.join(execution_plan['execution_order'])}")
        self._log(json.dumps(execution_plan, ensure_ascii=False, indent=2))

        # --- Chapter-5 扩展: 子智能体执行 ---
        self._log("\n⚙️ [Ch5+] 执行子智能体...")
        subtask_results = await self._execute_subtasks(execution_plan, thread_id)

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
        self, execution_plan: Dict[str, Any], thread_id: str
    ) -> Dict[str, Any]:
        from sub_agents import SubAgentFactory

        results: Dict[str, Any] = {}
        subtasks = {t["task_id"]: t for t in execution_plan.get("subtasks", [])}

        layers = self._topological_layers(execution_plan)
        for layer in layers:
            if len(layer) == 1:
                tid = layer[0]
                results[tid] = await self._invoke_sub_agent(
                    SubAgentFactory, subtasks[tid], results, thread_id
                )
            else:
                layer_results = await asyncio.gather(*[
                    self._invoke_sub_agent(SubAgentFactory, subtasks[tid], results, thread_id)
                    for tid in layer
                ])
                for tid, res in zip(layer, layer_results):
                    results[tid] = res

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
    ) -> Any:
        task_id = task["task_id"]
        agent_name = task.get("agent", "ItineraryAgent")
        description = task.get("description", "")

        self._log(f"\n  🔄 {task_id}: {description[:60]}...")
        self._log(f"     → {agent_name}")

        query_parts = [description]
        if task.get("params"):
            query_parts.append(f"参数: {json.dumps(task['params'], ensure_ascii=False)}")
        for dep_id in task.get("depends_on", []):
            if dep_id in prior_results:
                dep_json = json.dumps(prior_results[dep_id], ensure_ascii=False)
                if len(dep_json) > 2000:
                    dep_json = dep_json[:2000] + "..."
                query_parts.append(f"依赖 {dep_id} 的结果: {dep_json}")

        user_message = "\n".join(query_parts)
        agent = factory.get_agent(agent_name)
        state = await agent.ainvoke(
            {"messages": [("user", user_message)]},
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

        result = {
            "task_id": task_id,
            "agent": agent_name,
            "tool_data": tool_outputs[-1] if tool_outputs else None,
            "agent_summary": agent_text,
        }

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
        memories_text = json.dumps(execution_plan.get("retrieved_memories", []), ensure_ascii=False, indent=2)
        pre_survey_text = json.dumps(execution_plan.get("pre_survey", {}), ensure_ascii=False, indent=2)

        if is_single_direct_response(results):
            final_text = direct_response_from_results(results)
            self._log("  ✓ 单任务查询，直接使用子智能体回复（跳过旅行规划聚合）")
        elif self.memory_system:
            prompt = self.memory_system.build_prompt(
                thread_id,
                user_query,
                self.memory_system.search_memories(user_query),
            )
            prompt += f"\n\n## 子任务执行结果\n{json.dumps(results, ensure_ascii=False, indent=2)}"
            prompt += f"\n\n{MEMORY_AGGREGATION_INSTRUCTION}"
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            final_text = response.content or ""
        else:
            prompt = AGGREGATION_PROMPT.format(
                user_query=user_query,
                pre_survey=pre_survey_text,
                memories=memories_text,
                total_goal=execution_plan.get("total_goal", ""),
                results=json.dumps(results, ensure_ascii=False, indent=2),
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
