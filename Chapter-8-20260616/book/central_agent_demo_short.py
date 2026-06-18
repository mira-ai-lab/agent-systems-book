"""
Chapter-8 精简版 — LangGraph 固定图多智能体协作


固定图六步（与 agent_framework/orchestration/fixed_graph 同构）：

    pre_survey → retrieve_memory → build_plan
        → execute_layer → aggregate → save_memory → final_response

运行：
  cd Chapter-8/book
  python central_agent_demo_short.py

联网完整版：cd Chapter-8 && python scripts/run_demo.py --stream
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

GRAPH_CHAIN = (
    "pre_survey → retrieve_memory → build_plan → execute_layer → aggregate → save_memory"
)

# ── 1. 数据结构（对齐 CentralAgentState / execution_plan.subtasks）────────

@dataclass
class SubTask:
    task_id: str
    description: str
    agent: str
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)


@dataclass
class GraphRunState:
    user_query: str
    pre_survey: Dict[str, Any] = field(default_factory=dict)
    memories: List[Dict[str, Any]] = field(default_factory=list)
    execution_plan: Dict[str, Any] = field(default_factory=dict)
    subtask_results: Dict[str, str] = field(default_factory=dict)
    final_response: str = ""


# ── 2. 子 Agent 演示（真实项目：agents/factory.py → SubAgentFactory）──────

class SubAgentDemo:
    REPLIES = {
        "WeatherAgent": "【天气科】上海/苏州/杭州：多云 24–30°C，适宜出行。",
        "HotelAgent": "【酒店科】安静型酒店 2–3 家，≤800元/晚。",
        "RestaurantAgent": "【美食科】本帮菜、苏式面、龙井虾仁各 3 家推荐。",
        "ItineraryAgent": "【行程科】Day1 沪→苏；Day2 拙政园+平江路；Day3 杭西湖；返沪。",
    }

    def run(self, agent: str, instruction: str) -> str:
        print(f"      └─ [{agent}] {instruction[:70]}…")
        return self.REPLIES.get(agent, f"【{agent}】已处理")


# ── 3. 固定示例计划（完整版用 TaskPlannerDemo 动态拆解）────────────────────

def build_demo_plan(user_query: str) -> Dict[str, Any]:
    """教科书固定示例：上海→苏州→杭州，T4 依赖 T1–T3。"""
    subtasks = [
        SubTask(
            "T1",
            "WeatherAgent：查询三地逐日天气预报与出行建议",
            "WeatherAgent",
            {"cities": ["上海", "苏州", "杭州"], "date": "2026-06-18"},
        ),
        SubTask(
            "T2",
            "HotelAgent：推荐安静酒店，≤800元/晚",
            "HotelAgent",
            {"cities": ["上海", "苏州", "杭州"], "budget_cny_per_night_max": 800},
        ),
        SubTask(
            "T3",
            "RestaurantAgent：推荐三地本地美食及餐厅",
            "RestaurantAgent",
            {"location": "上海", "budget_cny_per_person": 150},
        ),
        SubTask(
            "T4",
            "ItineraryAgent：生成闭环逐日行程（依赖天气/酒店/美食）",
            "ItineraryAgent",
            {"departure_city": "上海", "destination_city": "杭州", "days": 3},
            depends_on=["T1", "T2", "T3"],
        ),
    ]
    layers = [["T1", "T2", "T3"], ["T4"]]  # 第一层并行，第二层行程科
    return {
        "user_goal": user_query,
        "total_goal": "上海→苏州→杭州 3 天联游（天气+酒店+美食+行程）",
        "subtasks": [vars(t) for t in subtasks],
        "execution_order": ["T1", "T2", "T3", "T4"],
        "pending_layers": layers,
    }


def build_instruction(task: Dict[str, Any], prior: Dict[str, str]) -> str:
    parts = [task["description"]]
    if task.get("params"):
        parts.append(f"参数: {json.dumps(task['params'], ensure_ascii=False)}")
    for dep in task.get("depends_on", []):
        if dep in prior:
            parts.append(f"参考 {dep}：{prior[dep]}")
    return "\n".join(parts)


# ── 4. 固定图编排（六步节点；真实项目见 orchestrator.py + nodes.py）────────

class FixedGraphBookDemo:
    def __init__(self) -> None:
        self.workers = SubAgentDemo()

    def handle(self, user_query: str) -> GraphRunState:
        state = GraphRunState(user_query=user_query)
        print("=" * 60)
        print(f"📥 用户：{user_query.strip()}")
        print(f"🔗 {GRAPH_CHAIN}")
        print("=" * 60)

        # Ch2 预调查
        print("\n▶ pre_survey [Ch2]")
        state.pre_survey = {
            "given_facts": ["城市：上海、苏州、杭州", "偏好：安静酒店 ≤800元/晚"],
            "facts_to_lookup": ["天气", "酒店", "美食"],
        }
        print(json.dumps(state.pre_survey, ensure_ascii=False, indent=2))

        # Ch3 记忆检索
        print("\n▶ retrieve_memory [Ch3]")
        state.memories = [{"type": "preference", "content": "偏好安静酒店"}]
        print(json.dumps(state.memories, ensure_ascii=False, indent=2))

        # Ch4 任务规划
        print("\n▶ build_plan [Ch4]")
        state.execution_plan = build_demo_plan(user_query)
        print(json.dumps(state.execution_plan["subtasks"], ensure_ascii=False, indent=2))

        # Ch5+ 按层执行子 Agent
        print("\n▶ execute_layer [Ch5+]")
        ep = state.execution_plan
        subtasks = {t["task_id"]: t for t in ep["subtasks"]}
        for i, layer in enumerate(ep["pending_layers"], 1):
            print(f"  ⚙️ 第 {i} 层: {layer}")
            for tid in layer:
                task = subtasks[tid]
                print(f"  ▶ {tid} → {task['agent']}")
                instr = build_instruction(task, state.subtask_results)
                state.subtask_results[tid] = self.workers.run(task["agent"], instr)
                print(f"  ✓ {tid} 完成")

        # 汇聚
        print("\n▶ aggregate [汇聚]")
        lines = [f"您好！多智能体协作结果（{user_query.strip()[:30]}…）：\n"]
        for i, tid in enumerate(ep["execution_order"], 1):
            lines.append(f"{i}. {state.subtask_results[tid]}")
        state.final_response = "\n".join(lines)
        print("=" * 60)
        print(state.final_response)
        print("=" * 60)

        # Ch3 写回记忆
        print("\n▶ save_memory [Ch3 写回]（演示：已记录本轮对话）")
        return state


# ── 主程序 ──────────────────────────────────────────────────────────────────

DEMO_QUERY = (
    "我下周去上海、苏州、杭州，查天气、订安静酒店、推荐美食，最后出个行程。"
    "酒店每晚不超过800元。"
)


if __name__ == "__main__":
    FixedGraphBookDemo().handle(DEMO_QUERY)


# 运行结果
# 📥 用户：我下周去上海、苏州、杭州，查天气、订安静酒店、推荐美食，最后出个行程。酒店每晚不超过800元。
# 🔗 pre_survey → retrieve_memory → build_plan → execute_layer → aggregate → save_memory
# ============================================================
#
# ▶ pre_survey [Ch2]
# {
#   "given_facts": [
#     "城市：上海、苏州、杭州",
#     "偏好：安静酒店 ≤800元/晚"
#   ],
#   "facts_to_lookup": [
#     "天气",
#     "酒店",
#     "美食"
#   ]
# }
#
# ▶ retrieve_memory [Ch3]
# [
#   {
#     "type": "preference",
#     "content": "偏好安静酒店"
#   }
# ]
#
# ▶ build_plan [Ch4]
# [
#   {
#     "task_id": "T1",
#     "description": "WeatherAgent：查询三地逐日天气预报与出行建议",
#     "agent": "WeatherAgent",
#     "params": {
#       "cities": [
#         "上海",
#         "苏州",
#         "杭州"
#       ],
#       "date": "2026-06-18"
#     },
#     "depends_on": []
#   },
#   {
#     "task_id": "T2",
#     "description": "HotelAgent：推荐安静酒店，≤800元/晚",
#     "agent": "HotelAgent",
#     "params": {
#       "cities": [
#         "上海",
#         "苏州",
#         "杭州"
#       ],
#       "budget_cny_per_night_max": 800
#     },
#     "depends_on": []
#   },
#   {
#     "task_id": "T3",
#     "description": "RestaurantAgent：推荐三地本地美食及餐厅",
#     "agent": "RestaurantAgent",
#     "params": {
#       "location": "上海",
#       "budget_cny_per_person": 150
#     },
#     "depends_on": []
#   },
#   {
#     "task_id": "T4",
#     "description": "ItineraryAgent：生成闭环逐日行程（依赖天气/酒店/美食）",
#     "agent": "ItineraryAgent",
#     "params": {
#       "departure_city": "上海",
#       "destination_city": "杭州",
#       "days": 3
#     },
#     "depends_on": [
#       "T1",
#       "T2",
#       "T3"
#     ]
#   }
# ]
#
# ▶ execute_layer [Ch5+]
#   ⚙️ 第 1 层: ['T1', 'T2', 'T3']
#   ▶ T1 → WeatherAgent
#       └─ [WeatherAgent] WeatherAgent：查询三地逐日天气预报与出行建议
# 参数: {"cities": ["上海", "苏州", "杭州"], "date"…
#   ✓ T1 完成
#   ▶ T2 → HotelAgent
#       └─ [HotelAgent] HotelAgent：推荐安静酒店，≤800元/晚
# 参数: {"cities": ["上海", "苏州", "杭州"], "budget_c…
#   ✓ T2 完成
#   ▶ T3 → RestaurantAgent
#       └─ [RestaurantAgent] RestaurantAgent：推荐三地本地美食及餐厅
# 参数: {"location": "上海", "budget_cny_per_per…
#   ✓ T3 完成
#   ⚙️ 第 2 层: ['T4']
#   ▶ T4 → ItineraryAgent
#       └─ [ItineraryAgent] ItineraryAgent：生成闭环逐日行程（依赖天气/酒店/美食）
# 参数: {"departure_city": "上海", "dest…
#   ✓ T4 完成
#
# ▶ aggregate [汇聚]
# ============================================================
# 您好！多智能体协作结果（我下周去上海、苏州、杭州，查天气、订安静酒店、推荐美食，最后…）：
#
# 1. 【天气科】上海/苏州/杭州：多云 24–30°C，适宜出行。
# 2. 【酒店科】安静型酒店 2–3 家，≤800元/晚。
# 3. 【美食科】本帮菜、苏式面、龙井虾仁各 3 家推荐。
# 4. 【行程科】Day1 沪→苏；Day2 拙政园+平江路；Day3 杭西湖；返沪。
# ============================================================
#
# ▶ save_memory [Ch3 写回]（演示：已记录本轮对话）


